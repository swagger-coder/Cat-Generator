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
    st.title("Text title & abstract Generator")
    st.info("**INFO:** This is a tool for processing individual news text.Enter your news in the text box below.")
    st.markdown(
        """
        ### Enter the news text
        """
    )
    content = st.text_area("Two thousand words Max", max_chars=2000)
    c35, c36, c37 = st.columns([3.5,3,3.5])
    if c36.button("click to generate"):
        content.strip("\n")
        st.header("Title Generation")
        with st.spinner('Please wait while title is being generated...'):
            start_time = time.time()
            titles = predict_one_sample(model, tokenizer, device, args, content)
            end_time = time.time()
        st.success("Title generation complete, take {}s".format(end_time - start_time))
        for i, title in enumerate(titles):
            st.caption("title-result {}".format(i + 1))
            st.info(title)

        st.header("Abstract extraction")
        with st.spinner('Please wait while abstract is being extracted...'):
            start_time = time.time()
            abstracts = get_abstract(content)
            end_time = time.time()
        st.success("Abstract generation complete, take {}s".format(end_time - start_time))

        for i, abstract in enumerate(abstracts):
            st.caption("abstract result {}".format(i + 1))
            st.info(abstract)
    else:
        st.stop()


if __name__ == '__main__':
    writer()
