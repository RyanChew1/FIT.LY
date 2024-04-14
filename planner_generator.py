# %%
import kagglehub
import torch
torch.cuda.is_available()

# %%
weights_dir = kagglehub.model_download(
    "google/gemma/pyTorch/2b-it")

# %%
import os
tokenizer_path = os.path.join(weights_dir, "tokenizer.model")

ckpt_path = os.path.join(weights_dir, "gemma-2b-it.ckpt")

# %%

# %%
import sys

sys.path.append("gemma_pytorch")

# %%
from gemma_pytorch.gemma.config import get_config_for_2b
from gemma_pytorch.gemma.model import GemmaForCausalLM

# %%
import torch

model_config = get_config_for_2b()
model_config

# %%
model_config.tokenizer = tokenizer_path

# %%
model_config.quant = "quant" in "2b-it"

# %%
torch.set_default_dtype(model_config.get_dtype())

# %%
device = torch.device("cuda")
device

# %%
model = GemmaForCausalLM(model_config)
model.load_weights(ckpt_path)

# %%
model = model.to(device)

# %% [markdown]
# # INFERENCE

# %%
model.generate("Give me a 5 day workout plan, 2 rest days. Targeting chest. Make each workout around 30 mins long",
device = device,
output_len = 100)

# %%
def inference(days:int=5, target:str="all around", length:str = "1 hour", rep_range = "8 to 10"):
    prompt = f"Generate me a weekly workout plan. "
    if days:
        prompt += f"Use {days} workout days and {7-days} rest days. "
    if target:
        prompt += f"I want to target the {target}. "
    if length:
        prompt += f"Each workout should be {length} long. "
    if rep_range:
        prompt += f"Aim for the {rep_range} rep range."
    return model.generate(
        prompt,
        device = device,
        output_len = 1000,
    )

# %%
print(inference())

# %%



