import streamlit as st
import pandas as pd
from PIL import Image

###################################
from utils.get_title import predict_one_sample
from utils.get_abstract import get_abstract
###################################

from utils.functionforDownloadButtons import download_button

###################################
from utils.UIhelper import get_model, generate_sidebar_elements, style_button_row


# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

def writer():
    st.set_page_config(page_icon=Image.open('./static/布偶猫-稀有色.png'), page_title="Cat-File")
    img = Image.open('./static/布偶猫-稀有色.png')
    device, tokenizer, model = get_model("vocab/vocab.txt", "output_dir/checkpoint-139805")
    c32, c33, c34 = st.columns([3.5,3,3.5])
    c33.image(img, width=200)
    args, batch_size, generate_max_len, repetition_penalty, top_k, top_p = generate_sidebar_elements()
    def _max_width_():
        max_width_str = f"max-width: 1800px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>    
        """,
            unsafe_allow_html=True,
        )
    _max_width_()


    st.title("News title & abstract Generator" if args.language else "新闻标题&摘要生成器")
    st.info(
    f"""
    **INFO:** This is a tool for generating titles and summaries in batches, please upload files in the specified format. You can download [demo.csv](https://filedropper.com/d/s/18PPvpjkFtgXg5lmxwchw4xiLmCdbA) to view the format template.
    """ if args.language else
    f"""
    **注意:**  这是批量生成标题和摘要的工具，请按照指定格式上传文件。你可以下载[demo.csv](https://filedropper.com/d/s/18PPvpjkFtgXg5lmxwchw4xiLmCdbA)查看格式模板。
    """)

    c29, c30, c31 = st.columns([1, 6, 1])
    c32, c33, c34 = st.columns([4,3,3.5])
    with c30:

        uploaded_file = st.file_uploader(
            "",
            key="1",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'" if args.language else "要激活“宽模式”，请到菜单> 设置>打开“宽模式”" ,
        )

        if uploaded_file is not None:
            file_container = st.expander("Check your uploaded .csv" if args.language else "检查你上传的 .csv文件")
            shows = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            uploaded_file.seek(0)
            file_container.write(shows)

        else:
            st.info(
                f"""
                    👆 Only UTF-8 files in CSV format are allowed. Please download the template: [demo.csv](https://filedropper.com/d/s/18PPvpjkFtgXg5lmxwchw4xiLmCdbA)
                """ if args.language else
                f"""
                👆 只允许使用CSV格式的UTF-8文件。请下载模板: [demo.csv](https://filedropper.com/d/s/18PPvpjkFtgXg5lmxwchw4xiLmCdbA)
                """
            )

            st.stop()

    # st.balloons()
    if c33.button("Click to batch generate" if args.language else " 点 击 批 量 生 成", on_click=style_button_row, kwargs={
    'clicked_button_ix': 3, 'n_buttons': 4}):
        rows = shows.shape[0]
        cols = shows.shape[1]
        title_abs_dict = dict()

        with st.spinner('Please wait for batch processing...' if args.language else
                        "请等待批量处理…"):
            placeholder = st.empty()
            my_bar = st.progress(0)
            # placeholder2 = st.empty()
            for i in range(rows):
                placeholder.text("The text {} is currently being processed, with {} remaining".format(i+1, rows-i-1) if args.language else
                                 "文本 {} 目前正在处理中，还剩下 {} 个文本未处理".format(i+1, rows-i-1))
                content = shows.iloc[i,0]
                my_bar.progress((i+1)//rows * 100)
                st.text("text"+str(i+1)+"："+content if args.language else
                        "文本"+str(i+1)+"："+content)
                titles = predict_one_sample(model, tokenizer, device, args, content)
                for j, item in enumerate(titles):
                    if "标题"+str(j) not in title_abs_dict.keys():
                        title_abs_dict["标题"+str(j)] = []
                        title_abs_dict["标题"+str(j)].append(item)
                    else:
                        title_abs_dict["标题"+str(j)].append(item)

                abstracts = get_abstract(content, args.summary_nums)
                for j, item in enumerate(abstracts):
                    if "摘要"+str(j) not in title_abs_dict.keys():
                        title_abs_dict["摘要"+str(j)] = []
                        title_abs_dict["摘要"+str(j)].append(item)
                    else:
                        title_abs_dict["摘要"+str(j)].append(item)
        placeholder.success('Done! ' if args.language else
                            "已完成！")


        df2 = pd.DataFrame(
            title_abs_dict
        )
        # st.table(df2)
        result = pd.concat([shows.iloc[:,:1], df2], axis=1)
        # st.table(result)
        st.subheader("Click the button below to download results 👇 " if args.language else
                     "点击下方按钮进行下载结果 👇 ")

        c29, c30, c31 = st.columns([3.75, 2.5, 3.75])

        with c30:

            CSVButton = download_button(
                result,
                "File.csv",
                "Click to download" if args.language else "  点 击 下 载",
            )

if __name__ == "__main__":
    writer()