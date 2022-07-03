import argparse

import pandas as pd
import streamlit as st
from utils.model import GPT2LMHeadModel
from transformers import BertTokenizer

import os
import torch

@st.cache(allow_output_mutation=True)
def get_model(vocab_path, model_path):
    device_ids = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = str(device_ids)
    device = torch.device("cuda" if torch.cuda.is_available() and int(device_ids) >= 0 else "cpu")
    tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return device, tokenizer, model

def generate_sidebar_elements(flag = True):
    en = flag
    option = st.sidebar.selectbox("", ('English', '中文') if en else ('中文', 'English'))
    if option == "English":
        en = True
    else:
        en = False
    # st.sidebar.subheader("Parameter configuration" if en else "配置参数")
    paras = {"parameter" if en else "参数": ["Title_nums",
                           "Summary_nums",
                           "Generate_max_len",
                           "Repetition_penalty",
                           "Top_k",
                           "Top_p"],
             "explanation" if en else "解释": ["set the number of titles to generate" if en else "设置生成标题的个数",
                             "set the number of summaries to generate" if en else "设置生成摘要的个数",
                             "set the max length of the title" if en else "设置生成标题的最大长度",
                             "set parameter for repetition penalty" if en else "设置重复处罚率",
                             "set the number of tokens with the highest probability of retention when decoding" if en else "设置解码时保留概率最高的多少个标记",
                             "set the flag at which the cumulative retention probability is greater than when decoding" if en else "设置解码时保留概率累加大于多少的标记"]
    }
    df = pd.DataFrame(paras)
    with st.sidebar.expander("Check parameters explanation" if en else "查看参数解释"):
        st.table(df)

    batch_size = st.sidebar.slider("Title_nums", min_value=0, max_value=10, value=3)
    summary_nums = st.sidebar.slider("Summary_nums", min_value=0, max_value=5, value=3)
    generate_max_len = st.sidebar.number_input("Generate_max_len", min_value=0, max_value=64, value=32, step=1)
    repetition_penalty = st.sidebar.number_input("Repetition_penalty", min_value=0.0, max_value=10.0, value=1.2,
                                                 step=0.1)
    top_k = st.sidebar.slider("Top_k", min_value=0, max_value=10, value=3, step=1)
    top_p = st.sidebar.number_input("Top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=batch_size, type=int, help='生成标题的个数')
    parser.add_argument('--summary_nums', type=int, default=summary_nums, help='生成摘要数量')
    parser.add_argument('--generate_max_len', default=generate_max_len, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=repetition_penalty, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=top_k, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=top_p, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--language', type=bool, default=en, help='语言')

    args = parser.parse_args()
    return args, batch_size, generate_max_len, repetition_penalty, top_k, top_p



def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
