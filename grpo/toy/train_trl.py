import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from toy.task import make_dataset, build_messages
from toy.reward import reward, parse_answer


def accuracy_reward(completions, answer, **kwargs):
    """The real reward: the same function your own trainer uses."""
    return [reward(c[0]["content"], a) for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs):
    """Weight 0: only logged, so the format rate can be compared. No effect on training."""
    return [float(parse_answer(c[0]["content"]) is not None) for c in completions]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/trl_d3")
    args = p.parse_args()

    # Same task: 16 questions per step, so steps * 16 questions in total
    problems = make_dataset(args.seed, args.steps * 16, args.difficulty)
    dataset = Dataset.from_list(
        [{"prompt": build_messages(q), "answer": a} for q, a in problems]
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    # Neutralise Qwen's built-in sampling defaults, so TRL samples from the true policy
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.generation_config.repetition_penalty = 1.0

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        # optimiser: match yours
        learning_rate=args.lr, lr_scheduler_type="constant",
        weight_decay=0.0, max_grad_norm=1.0,
        # batch: 8 answers per chunk x 16 chunks = 128 answers = 16 questions x K=8
        per_device_train_batch_size=8, gradient_accumulation_steps=16,
        num_generations=8, num_iterations=1,
        # sampling: true policy, no filters
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        max_completion_length=256,
        # loss: match yours
        beta=args.beta, epsilon=0.2, loss_type="grpo", scale_rewards="group",
        # bookkeeping
        reward_weights=[1.0, 0.0],
        logging_steps=1, report_to="none", save_strategy="no",
        seed=args.seed, bf16=True,
    )

    print(config.to_json_string())

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[accuracy_reward, format_reward],
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(os.path.join(args.out, "lora"))
    with open(os.path.join(args.out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)


if __name__ == "__main__":
    main()