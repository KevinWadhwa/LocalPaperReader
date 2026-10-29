import argparse
from collections import Counter

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from qasper.abstain import subtype, twins
from qasper.sft_targets import build_sft_examples, build_target

ABSTAIN_SHARE = 0.17                                                     # of the total weight: the dev and GRPO mix
UNANSWERABLE_SHARE = 0.40                                                # of the abstain weight, about dev's
KINDED = ("evidence_missing", "table_missing", "table_caption_shown")    # abstains that have a question kind


def weighted_pairs(split, tok, use_twins=True):
    """(example, target, weight). Answers weigh 1. Abstains (contaminated labels dropped) share ABSTAIN_SHARE of the
    weight: UNANSWERABLE_SHARE to unanswerable, the rest to each question kind in proportion to its answers, so
    every kind has the same abstain rate. Within a cell every example weighs the same."""
    pairs, _ = build_sft_examples(split, tok)
    answers = [(e, t) for e, t in pairs if e.target == "answer"]
    cells = {}                                                           # question kind (None: unanswerable) -> (e, t, source)
    for e, t in pairs:
        s = subtype(e)
        if s == "unanswerable":
            cells.setdefault(None, []).append((e, t, s))
        elif s in KINDED:
            cells.setdefault(e.question.kind, []).append((e, t, s))
    if use_twins:
        for _, w in twins([e for e, _ in answers], split, tok, drop_long=True):
            cells.setdefault(w.question.kind, []).append((w, build_target(w)[0], "twin"))

    kinds = Counter(e.question.kind for e, _ in answers)
    abstain_mass = ABSTAIN_SHARE / (1 - ABSTAIN_SHARE) * len(answers)
    mass = {None: UNANSWERABLE_SHARE * abstain_mass}
    mass.update({k: (1 - UNANSWERABLE_SHARE) * abstain_mass * n / len(answers) for k, n in kinds.items()})

    rows, by_source = [(e, t, 1.0) for e, t in answers], Counter()
    for c, ps in sorted(cells.items(), key=str):
        w = mass.get(c, 0.0) / len(ps)
        rows += [(e, t, w) for e, t, _ in ps]
        for _, _, s in ps:
            by_source[s] += w
        print(f"  {str(c):12s} n={len(ps):5d}  weight {w:.3f}  total {w * len(ps):6.1f}  "
              f"{dict(Counter(s for _, _, s in ps))}")
    abst = sum(by_source.values())
    print(f"  {len(answers)} answers, {len(rows) - len(answers)} abstains with total weight {abst:.1f}: "
          f"abstain share {abst / (abst + len(answers)):.3f}")
    print("  abstain weight by source: " + ", ".join(f"{s} {v / abst:.1%}" for s, v in by_source.most_common()))
    return rows


def to_dataset(split, tok, weighted=True, use_twins=True):
    if weighted:
        rows = weighted_pairs(split, tok, use_twins)
    else:
        rows = [(e, t, 1.0) for e, t in build_sft_examples(split, tok)[0]]
    return Dataset.from_list([{"prompt": e.prompt, "completion": t + "<|im_end|>", "weight": w} for e, t, w in rows])


class WeightedCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        batch["weight"] = torch.tensor([e["weight"] for e in examples])
        return batch


class WeightedSFTTrainer(SFTTrainer):
    """SFTTrainer's loss (sum over completion tokens / tokens in the batch) with each example's tokens scaled by
    its weight. The count stays unweighted, so a batch of low-weight examples gives a small loss."""

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "weight" not in self._signature_columns:
            self._signature_columns.append("weight")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weight = inputs.pop("weight")
        inputs["use_cache"] = False
        outputs = model(**inputs, fused_lm_head=True)
        per_token_loss = -outputs.log_probs * weight.to(outputs.log_probs).unsqueeze(1)
        num_tokens = outputs.label_mask.sum() if num_items_in_batch is None else num_items_in_batch
        loss = per_token_loss.sum() / torch.as_tensor(num_tokens).clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="runs/sft_v2_1.7b")
    ap.add_argument("--no_twins", action="store_true")                  # ablation: cleaning and reweighting only
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    config = SFTConfig(
        output_dir=args.out, num_train_epochs=2, learning_rate=1e-4,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_steps=10,
        max_length=None, completion_only_loss=True, bf16=False, tf32=True, gradient_checkpointing=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", report_to="none",
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    trainer = WeightedSFTTrainer(model=model, args=config, processing_class=tok, peft_config=lora,
                                 data_collator=WeightedCollator(pad_token_id=tok.pad_token_id),
                                 train_dataset=to_dataset("train", tok, use_twins=not args.no_twins),
                                 eval_dataset=to_dataset("dev", tok, weighted=False))
    trainer.train()
    trainer.save_model(f"{args.out}/final")

    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(f"{args.out}/merged")
    tok.save_pretrained(f"{args.out}/merged")


if __name__ == "__main__":
    main()
