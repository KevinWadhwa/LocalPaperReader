"""Merge the GRPO LoRA adapter (step80) into the merged SFT base, writing a standalone
HF model for the generator GGUF. The verifier uses sft_v2_4b/merged directly (adapter OFF),
so only the generator needs this merge. CPU is fine; this is just a weight fold, no GPU.

    python parsescripts/merge_grpo.py
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "runs/sft_v2_4b/merged"
ADAPTER = "runs/grpo_v2_4b/step80"
# Write the 15GB merged model to local /tmp, not scratch: scratch has a tight per-user
# NFS quota (the merge OOM'd it once), while /tmp is on the node's 1.3TB local disk.
OUT = "/tmp/kw1525_gen_v2_4b_merged"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()                       # fold LoRA into the base weights
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"generator (SFT+GRPO merged) written to {OUT}")
