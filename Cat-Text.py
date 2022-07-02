# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: Cat-Text.py.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2022/2/19 14:19
"""
    文件说明:

"""
import streamlit as st
import time
from utils.get_title import predict_one_sample
from utils.get_abstract import get_abstract
from PIL import Image

from utils.UIhelper import get_model, generate_sidebar_elements



def writer():
    st.set_page_config(
        page_title="Cat-Text",
        page_icon=Image.open('./static/布偶猫-稀有色.png'),
    )
    img = Image.open('./static/布偶猫-稀有色.png')
    device, tokenizer, model = get_model("vocab/vocab.txt", "output_dir/checkpoint-139805")
    c32, c33, c34 = st.columns([3.5,3,3.5])
    c33.image(img, width=200)

    args, batch_size, generate_max_len, repetition_penalty, top_k, top_p = generate_sidebar_elements()
    st.title("News title & summary Generator" if args.language else "新闻标题&摘要生成器")
    st.info("**INFO:** This is a tool for processing single news text. You can generate news titles and summaries by setting the parameters in the left sidebar. Enter your news in the text box below!" if args.language
            else "**注意:** 这是一个处理单个新闻文本的工具。您可以通过设置左侧栏中的参数来生成新闻标题和摘要。请在下面的文本框中输入您的新闻!")
    c35, c36, c37 = st.columns([4.25,3,3.5])
    button1 = c36.empty()
    st.markdown(
        """
        ### Enter the news text
        """ if args.language else
        """
        ### 输入新闻文本
        """
    )
    content = st.text_area("100 words minimum 2000 words Maximum " if args.language else "最多2000字最少100字", max_chars=2000, placeholder="Enter your news text here" if args.language else "请在这输入你的新闻文本")
    c35, c36, c37 = st.columns([4.25,3,3.5])
    if c36.button("click to generate" if args.language else "点击生成"):
        content.strip("\n")
        if len(content)<100:
            st.error("News text needs more than 100 words" if args.language else "新闻文本需要多于100字")
        else:
            st.header("Title Generation" if args.language else "标题生成")
            with st.spinner('Please wait while title is being generated...' if args.language else "正在生成标题，请稍候…"):
                start_time = time.time()
                titles = predict_one_sample(model, tokenizer, device, args, content)
                end_time = time.time()
            st.success("Title generation complete, take {}s".format(end_time - start_time) if args.language else "标题生成完成，耗时 {}s".format(end_time - start_time))
            for i, title in enumerate(titles):
                st.caption("title-result {}".format(i + 1) if args.language else "标题结果 {}".format(i + 1))
                st.info(title)

            st.header("Summary extraction" if args.language else "摘要抽取")
            with st.spinner('Please wait while summary is being extracted...' if args.language else "正在抽取摘要，请稍候…"):
                start_time = time.time()
                abstracts = get_abstract(content, args.summary_nums)
                end_time = time.time()
            st.success("Summary generation complete, take {}s".format(end_time - start_time) if args.language else "摘要生成完成，耗时 {}s".format(end_time - start_time))

            for i, abstract in enumerate(abstracts):
                st.caption("summary-result {}".format(i + 1) if args.language else "摘要结果 {}".format(i + 1))
                st.info(abstract)
    else:
        st.stop()


if __name__ == '__main__':
    writer()
