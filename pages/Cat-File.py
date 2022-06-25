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



    st.title("File title & abstract Generator")
    st.info(
    f"""
    **INFO:** Batch generate title and summary tool, please upload files in accordance with the specified format. You can download [demo.csv](https://filedropper.com/d/s/18PPvpjkFtgXg5lmxwchw4xiLmCdbA) and type it in its format.
    """)

    c29, c30, c31 = st.columns([1, 6, 1])
    c32, c33, c34 = st.columns([3.5,3,3.5])
    with c30:

        uploaded_file = st.file_uploader(
            "",
            key="1",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
        )

        if uploaded_file is not None:
            file_container = st.expander("Check your uploaded .csv")
            shows = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            uploaded_file.seek(0)
            file_container.write(shows)

        else:
            st.info(
                f"""
                    👆 Only CSV files in UTF-8 format are allowed. Sample to try: [demo.csv](https://filedropper.com/d/s/18PPvpjkFtgXg5lmxwchw4xiLmCdbA)
                    """
            )

            st.stop()

    # st.balloons()
    if c33.button("Click to batch generate", on_click=style_button_row, kwargs={
    'clicked_button_ix': 3, 'n_buttons': 4}):
        rows = shows.shape[0]
        cols = shows.shape[1]
        title_abs_dict = dict()

        with st.spinner('Please wait for batch processing...'):
            placeholder = st.empty()
            my_bar = st.progress(0)
            # placeholder2 = st.empty()
            for i in range(rows):
                placeholder.text("The text {} is currently being processed, with {} remaining".format(i+1, rows-i-1))
                content = shows.iloc[i,0]
                my_bar.progress((i+1)//rows * 100)
                st.text("text"+str(i+1)+"："+content)
                titles = predict_one_sample(model, tokenizer, device, args, content)
                for j, item in enumerate(titles):
                    if "标题"+str(j) not in title_abs_dict.keys():
                        title_abs_dict["标题"+str(j)] = []
                        title_abs_dict["标题"+str(j)].append(item)
                    else:
                        title_abs_dict["标题"+str(j)].append(item)

                abstracts = get_abstract(content)
                for j, item in enumerate(abstracts):
                    if "摘要"+str(j) not in title_abs_dict.keys():
                        title_abs_dict["摘要"+str(j)] = []
                        title_abs_dict["摘要"+str(j)].append(item)
                    else:
                        title_abs_dict["摘要"+str(j)].append(item)
        placeholder.success('Done! ')


        df2 = pd.DataFrame(
            title_abs_dict
        )
        # st.table(df2)
        result = pd.concat([shows.iloc[:,:1], df2], axis=1)
        # st.table(result)
        st.subheader("Click the button below to download 👇 ")

        c29, c30, c31 = st.columns([3.75, 2.5, 3.75])

        with c30:

            CSVButton = download_button(
                result,
                "File.csv",
                "Click to download",
            )

if __name__ == "__main__":
    writer()