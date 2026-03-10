from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch

model_name = "FPHam/Hemingway_Rewrite_13b_GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    disable_exllama=False
)

def rewrite(text):
    prompt = f"Rewrite the following text in the style of Ernest Hemingway:\n\n{text}\n\nRewritten:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

paragraph = "The sun dipped behind the hills as the caravan marched onward."
print(rewrite(paragraph))