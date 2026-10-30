import argparse
import json
import os

import numpy as np
import torch

from qasper.taskdata import build_examples
from qasper.abstain import subtype, twins
from qasper.evaluate import load_model
from qasper.reward import ABSTAIN

LAYER = 20                                  # hidden_states[20]: the output of decoder layer 20 of 28
SHOWN = ("answer", "ungrounded")            # outcomes that pass the rules; the verifier decides on these


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/sft_v2_1.7b/merged")           # adapter off: the frozen SFT
    ap.add_argument("--adapter", default="runs/grpo_v2_1.7b/final")         # adapter on: the generator
    ap.add_argument("--split", default="dev", choices=["dev", "test"])      # test: once, after the verifier is fixed
    ap.add_argument("--eval_dir", default="runs/qasper_eval/grpo_v2")       # evaluate.py outputs of the generator
    ap.add_argument("--sft_eval_dir", default="runs/qasper_eval/sft_v2")    # and of the SFT: its stored abstain scores
    ap.add_argument("--batch_size", type=int, default=4)                    # 4: the batches evaluate.py scored abstains in
    ap.add_argument("--out", default="runs/verifier")
    return ap.parse_args()


def read(path):
    return [json.loads(l) for l in open(path)]


class Tap:
    """hidden_states[LAYER] (decoder layer LAYER's output) at position `pos` only: output_hidden_states would hold
    every layer's states for the whole sequence."""
    def __init__(self, model):
        self.pos, self.h = None, None
        model.get_base_model().model.layers[LAYER - 1].register_forward_hook(self)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        self.h = h[:, self.pos].detach().float().cpu().numpy()


@torch.no_grad()
def sft_pass(model, tok, examples, batch_size, tap):
    """evaluate.abstain_logps with the same batches, also returning the layer-LAYER state at the last prompt token."""
    comp = tok(f"<answer>{ABSTAIN}</answer><|im_end|>", add_special_tokens=False,
               return_tensors="pt")["input_ids"].to(model.device)
    C = comp.shape[1]
    order = sorted(range(len(examples)), key=lambda i: examples[i].n_tokens)
    lps, states = [None] * len(examples), [None] * len(examples)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok([examples[i].prompt for i in idx], return_tensors="pt", padding=True).to(model.device)
        seqs = torch.cat([batch["input_ids"], comp.expand(len(idx), -1)], dim=1)
        am = torch.cat([batch["attention_mask"], torch.ones_like(comp).expand(len(idx), -1)], dim=1)
        pos = (am.cumsum(dim=1) - 1).clamp(min=0)
        tap.pos = -C - 1                                                 # last prompt token: prompts are left-padded
        logits = model(input_ids=seqs, attention_mask=am, position_ids=pos, logits_to_keep=C + 1).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, seqs[:, -C:].unsqueeze(-1)).squeeze(-1).sum(dim=1)
        for j, i in enumerate(idx):
            lps[i], states[i] = lp[j].item(), tap.h[j]
        print(f"sft {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return np.array(lps), np.stack(states)


def answer_end_ids(tok, prompt, output):
    """Prompt and output tokenised separately and joined, as at generation, up to the token holding the last
    character of </answer>. None if there is no </answer>; the flag marks a final token that runs past it."""
    end = output.rfind("</answer>")
    if end < 0:
        return None, False
    end += len("</answer>")
    enc = tok(output, add_special_tokens=False, return_offsets_mapping=True)
    t = next(j for j, (_, b) in enumerate(enc["offset_mapping"]) if b >= end)
    return tok(prompt)["input_ids"] + enc["input_ids"][:t + 1], enc["offset_mapping"][t][1] > end


@torch.no_grad()
def generator_pass(model, tok, examples, outputs, batch_size, dim, tap):
    """Layer-LAYER state at the end of </answer>, the generator teacher-forced over prompt + its own output.
    NaN rows where the output has no </answer>."""
    seqs, overrun = zip(*(answer_end_ids(tok, e.prompt, o) for e, o in zip(examples, outputs)))
    order = sorted((i for i, s in enumerate(seqs) if s), key=lambda i: len(seqs[i]))
    states = np.full((len(seqs), dim), np.nan, dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(seqs[i])) + seqs[i] for i in idx], device=model.device)
        am = torch.tensor([[0] * (L - len(seqs[i])) + [1] * len(seqs[i]) for i in idx], device=model.device)
        pos = (am.cumsum(dim=1) - 1).clamp(min=0)
        tap.pos = -1
        model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=1)
        states[idx] = tap.h
        print(f"generator {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    print(f"answer-end states for {len(order)} of {len(seqs)} rows; final token runs past </answer> in {sum(overrun)}")
    return states


def check_stored(args, recs, lp):
    """The recomputed abstain score must match the SFT eval's: same prompts, same model, same baseline."""
    sft_tag = args.model.rstrip("/").split("/")[-1]
    files = [f"{args.sft_eval_dir}/{sft_tag}_{args.split}_{s}.jsonl" for s in ("outputs", "twins")]
    if not all(map(os.path.exists, files)):
        print("no stored SFT eval for this split: abstain-score check skipped")
        return
    old = read(files[0]) + read(files[1])
    assert [r["question_id"] for r in old] == [r["question_id"] for r in recs]
    old = np.array([r["logp_abstain"] for r in old])
    r = np.corrcoef(old, lp)[0, 1]
    print(f"recomputed vs stored SFT abstain score: r = {r:.6f}, max |diff| {np.abs(old - lp).max():.2e}")
    if r < 0.999:
        raise SystemExit("recomputed abstain scores don't match the stored SFT eval: check the prompts and the model")


def main():
    args = parse_args()
    tok, model = load_model(args.model, args.adapter)
    examples, _ = build_examples(args.split, tok)
    pairs = twins(examples, args.split, tok)
    assert all(e.question.paper_id == t.question.paper_id for e, t in pairs)     # a twin stays in its original's fold
    rows = examples + [t for _, t in pairs]
    tag = args.model.rstrip("/").split("/")[-1] + "_" + args.adapter.strip("/").replace("/", "_")   # as evaluate.py
    recs = read(f"{args.eval_dir}/{tag}_{args.split}_outputs.jsonl") + read(f"{args.eval_dir}/{tag}_{args.split}_twins.jsonl")
    assert [(r["question_id"], r["subtype"]) for r in recs] == [(e.question.question_id, subtype(e)) for e in rows], \
        "rebuilt prompts and twins don't match the eval"

    tap = Tap(model)
    with model.disable_adapter():
        lp, h_prompt = sft_pass(model, tok, rows, args.batch_size, tap)
    check_stored(args, recs, lp)
    h_answer = generator_pass(model, tok, rows, [r["output"] for r in recs], args.batch_size, h_prompt.shape[1], tap)

    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/{tag}_{args.split}.npz"
    np.savez(path, question_id=np.array([e.question.question_id for e in rows]),
             paper_id=np.array([e.question.paper_id for e in rows]), twin=np.arange(len(rows)) >= len(examples),
             target=np.array([r["target"] for r in recs]), subtype=np.array([r["subtype"] for r in recs]),
             kind=np.array([r["kind"] for r in recs]), outcome=np.array([r["outcome"] for r in recs]),
             shown=np.array([r["outcome"] in SHOWN for r in recs]),
             raw_f1=np.array([r.get("raw_f1", np.nan) for r in recs]),
             logp_abstain=lp, h_prompt=h_prompt, h_answer=h_answer)
    print(f"saved {path}: {len(rows)} rows ({len(examples)} questions, {len(pairs)} twins), "
          f"{sum(r['outcome'] in SHOWN for r in recs)} pass the rules")


if __name__ == "__main__":
    main()
