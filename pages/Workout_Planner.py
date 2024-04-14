import streamlit as st

import kagglehub
import torch
torch.cuda.is_available()

weights_dir = kagglehub.model_download(
    "google/gemma/pyTorch/2b-it")

import os
tokenizer_path = os.path.join(weights_dir, "tokenizer.model")

ckpt_path = os.path.join(weights_dir, "gemma-2b-it.ckpt")


import sys

sys.path.append("D:\FutureHacks6")

from gemma_pytorch.gemma.config import get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM

import torch

model_config = get_config_for_2b()

model_config.tokenizer = tokenizer_path

model_config.quant = "quant" in "2b-it"


torch.set_default_dtype(model_config.get_dtype())

device = torch.device("cpu")

model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)

model = model.to(device)

def inference(days:int=5, target:str="all around", length:str = "1 hour", rep_range = "8 to 10"):
    prompt = f"Generate me a weekly workout plan. "
    if days:
        prompt += f"Use {days} workout days and {7-days} rest days. Space out rest days and workout days evenly. "
    if target:
        prompt += f"I want to target the {target}. "
    if length:
        prompt += f"Each workout should be {length} long. "
    if rep_range:
        prompt += f"Aim for the {rep_range} rep range."
    return model.generate(
        prompt,
        device = device,
        output_len = 500,
    )

# def on_submit(out):
#     st.write("Here's your workout!")
#     st.write(out)




# with st.form("my_form"):
#     with st.container():
#         st.title('FIT.LY')
#         st.markdown("Generate Your Own Personalized Workout!")
#         # empty line
#         st.write("")
#         st.write("")

#         # Create a row for the input fields
#         row_input = st.columns((1, 1, 1, 1))

#         # Day input at column 1
#         with row_input[0]:
#             day = st.selectbox('Days', [1, 2, 3, 4, 5, 6, 7])

#         # Target input at column 2
#         with row_input[1]:
#             target = st.text_input('Target Muscle Group', max_chars=10)

#         # Length input at column 3
#         with row_input[2]:
#             length = st.slider('Length (min)', 15, 180, 60, 1)

#         # Rep range input at column 4
#         with row_input[3]:
#             rep_range = st.selectbox('Rep Range', ['Low (5-8)', 'Medium (8-12)', 'High (12-15)'])
#     submitted = st.form_submit_button("Submit", on_click=on_submit(inference(day,target,length,rep_range)))


def on_submit(day, target, length, rep_range):
    print("submitted")
    st.title("FIT.LY")
    st.markdown("Enjoy your workout!")
    st.write(inference(day,target,length, rep_range))


# Create a form
with st.form("my_form"):
    with st.container():
        st.title('FIT.LY')
        st.markdown("Generate Your Own Personalized Workout!")
        # empty line
        st.write("")
        st.write("")

        # Create a row for the input fields
        row_input = st.columns((1, 1, 1, 1))

        # Day input at column 1
        with row_input[0]:
            day = st.selectbox('Days', [1, 2, 3, 4, 5, 6, 7])

        # Target input at column 2
        with row_input[1]:
            target = st.text_input('Target Muscle Group', max_chars=10)

        # Length input at column 3
        with row_input[2]:
            length = st.slider('Length (min)', 15, 180, 60, 1)

        # Rep range input at column 4
        with row_input[3]:
            rep_range = st.selectbox('Rep Range', ['Low (5-8)', 'Medium (8-12)', 'High (12-15)'])
    submitted = st.form_submit_button("Submit", on_click=lambda: on_submit(day, target, length, rep_range))