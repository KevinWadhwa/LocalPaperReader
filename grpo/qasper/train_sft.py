import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from qasper.sft_targets import build_sft_examples


def to_dataset(split, tok):
    pairs, _ = build_sft_examples(split, tok)
    return Dataset.from_list([{"prompt": e.prompt, "completion": t + "<|im_end|>"} for e, t in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="runs/sft_1.7b")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    config = SFTConfig(
        output_dir=args.out, num_train_epochs=2, learning_rate=1e-4,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_steps=10,
        max_length=None, completion_only_loss=True, bf16=True, gradient_checkpointing=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", report_to="none",
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=model, args=config, processing_class=tok, peft_config=lora,
                         train_dataset=to_dataset("train", tok), eval_dataset=to_dataset("dev", tok))
    trainer.train()
    trainer.save_model(f"{args.out}/final")

    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(f"{args.out}/merged")
    tok.save_pretrained(f"{args.out}/merged")


if __name__ == "__main__":
    main()