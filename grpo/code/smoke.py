import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda")

msgs = [{"role": "user", "content": "What is 17 * 23? Put the final number in <answer></answer>."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(prompt, return_tensors="pt").to("cuda")

out = model.generate(**ids, max_new_tokens=200, do_sample=False)
new_tokens = out[0, ids["input_ids"].shape[1]:]   # drop the prompt, keep only the answer
print(tok.decode(new_tokens, skip_special_tokens=True))
