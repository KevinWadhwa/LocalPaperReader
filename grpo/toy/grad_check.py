"""Same-batch gradient check: my GRPO update vs TRL's, on one identical rollout.

One rollout is sampled with core.rollout, then the LoRA gradient is computed on the same
weights and the same tokens by:
  mine   core.train.accumulate_grads (my grpo_loss), run twice -> run-to-run noise floor
  trl    GRPOTrainer._compute_loss from the installed TRL, fed the same tensors

Both get the same advantages, so this tests the loss and backward pass only. TRL's advantage
eps (1e-4 vs 1e-8) is reported separately. Each check runs with beta as given and with
beta=0, so a mismatch can be pinned on the policy term or the KL term.

--bf16 1 matches train_trl.py (TRL's forward runs under bf16 autocast); --bf16 0 turns
autocast off so both sides run at the same precision.
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import use_adapter

from core.rollout import rollout
from core.train import prepare, accumulate_grads
from toy.task import make_problem, make_dataset, build_messages, format_prompt
from toy.reward import reward
from toy.train_trl import accuracy_reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="runs/trl_d3/lora",
                   help="trained LoRA to start from; 'none' = fresh LoRA (B=0, so KL and A grads are all zero)")
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--questions", type=int, default=16)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/grad_check")
    return p.parse_args()


def build_trainer(args):
    """A GRPOTrainer set up like train_trl.py. Its PEFT model is shared by both sides."""
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    n_micro = args.questions * args.k // args.micro_batch
    config = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=n_micro,
        num_generations=args.k, num_iterations=1,
        beta=args.beta, epsilon=args.eps_clip, loss_type="grpo", scale_rewards="group",
        max_completion_length=args.max_new_tokens,
        gradient_checkpointing=False,                                    # same forward as mine
        bf16=bool(args.bf16), report_to="none", save_strategy="no", seed=args.seed,
    )
    dataset = Dataset.from_list(
        [{"prompt": build_messages(q), "answer": a} for q, a in make_dataset(0, 16, args.difficulty)]
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[accuracy_reward],
                          args=config, train_dataset=dataset, peft_config=lora)
    trainer.current_gradient_accumulation_steps = n_micro               # normally set by the train loop
    tok.padding_side = "left"                                            # for my rollout
    return tok, trainer


def load_adapter(model, path):
    state = load_file(os.path.join(path, "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    missing = [key for key in result.missing_keys if "lora_" in key]
    assert not result.unexpected_keys and not missing, (result.unexpected_keys, missing)


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
                      for p in params])


def mine_grad(model, params, batch, adv, logp_old, logp_ref, args):
    model.zero_grad(set_to_none=True)
    model.train()
    stats = accumulate_grads(model, batch, adv, logp_old, logp_ref, args)
    return flat_grad(params), stats


def trl_ref_logps(trainer, model, batch, args):
    """Ref log-probs the way TRL computes them (its forward, adapters off)."""
    seqs, amask, P = batch["sequences"], batch["attention_mask"], batch["prompt_len"]
    with torch.no_grad(), use_adapter(model, adapter_name=None):
        ref, _, _ = trainer._get_per_token_logps_and_entropies(
            model, seqs, amask, seqs.shape[1] - P, batch_size=args.micro_batch)
    return ref


def trl_grad(trainer, model, params, batch, adv, ref):
    seqs, amask = batch["sequences"], batch["attention_mask"]
    cmask, P = batch["comp_mask"], batch["prompt_len"]
    model.zero_grad(set_to_none=True)
    model.train()
    mb = trainer.args.per_device_train_batch_size
    for i in range(0, seqs.shape[0], mb):
        s = slice(i, i + mb)
        inputs = {
            "prompt_ids": seqs[s, :P], "prompt_mask": amask[s, :P],
            "completion_ids": seqs[s, P:], "completion_mask": cmask[s],
            "advantages": adv[s], "ref_per_token_logps": ref[s],
        }
        trainer._compute_loss(model, inputs).backward()                 # already divided by n_micro
    return flat_grad(params)


def compare(name, a, b):
    cos = F.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / a.norm().clamp(min=1e-30)).item()
    print(f"  {name:28s} cos={cos:.6f}  rel_l2={rel:.3e}  |a|={a.norm().item():.4e}  |b|={b.norm().item():.4e}")
    return cos, rel


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok, trainer = build_trainer(args)
    model = trainer.model
    if args.adapter != "none":
        load_adapter(model, args.adapter)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"adapter={args.adapter} bf16_autocast={bool(args.bf16)} trainable={sum(p.numel() for p in params)}")

    # One rollout, shared by both sides
    model.eval()
    problems = [make_problem(rng, args.difficulty) for _ in range(args.questions)]
    batch = rollout(model, tok, problems, args.k, format_prompt, reward,
                    args.temperature, args.max_new_tokens)
    adv, logp_old, logp_ref = prepare(model, batch, args)
    cmask = batch["comp_mask"].bool()
    r = batch["rewards"]
    print(f"batch: {tuple(batch['sequences'].shape)} reward={r.mean().item():.3f} "
          f"mixed_groups={((r.sum(1) > 0) & (r.sum(1) < args.k)).float().mean().item():.3f} "
          f"mean_len={batch['comp_mask'].sum(1).float().mean().item():.1f}")

    # Inputs that differ before the loss
    trl_adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1).to(adv.device)
    ref_trl = trl_ref_logps(trainer, model, batch, args)
    print("inputs:")
    print(f"  advantage, TRL eps 1e-4      max|diff|={(adv - trl_adv).abs().max().item():.3e}  (not fed to TRL below)")
    print(f"  ref log-probs, mine vs TRL   max|diff|={(logp_ref - ref_trl)[cmask].abs().max().item():.3e}  "
          f"mean|diff|={(logp_ref - ref_trl)[cmask].abs().mean().item():.3e}")

    results = {}
    for beta in [args.beta, 0.0]:
        args.beta = beta
        trainer.beta = beta
        g1, stats = mine_grad(model, params, batch, adv, logp_old, logp_ref, args)
        g2, _ = mine_grad(model, params, batch, adv, logp_old, logp_ref, args)
        gt = trl_grad(trainer, model, params, batch, adv, ref_trl)
        print(f"beta={beta}  (mine: kl={stats['kl']:.4f} ratio={stats['ratio']:.4f} "
              f"entropy={stats['entropy']:.4f} loss={stats['loss']:.3e})")
        _, noise = compare("mine vs mine (noise floor)", g1, g2)
        cos, rel = compare("mine vs TRL", g1, gt)
        results[beta] = (cos, rel, noise)

    print("verdict:")
    for beta, (cos, rel, noise) in results.items():
        ok = cos > 0.999 and rel < 1e-2
        print(f"  beta={beta}: {'MATCH' if ok else 'MISMATCH'}  (cos={cos:.6f}, rel_l2={rel:.2e}, noise floor {noise:.2e})")


if __name__ == "__main__":
    main()
