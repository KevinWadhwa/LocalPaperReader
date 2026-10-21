import json
from collections import Counter
from dataclasses import dataclass

from qasper.data import load_split
from qasper.context import build_context, target_label
from qasper.prompt import format_prompt, MAX_PROMPT_TOKENS


@dataclass
class Example:
    question: object
    paper: object
    p_to_idx: dict
    target: str
    gold_ps: list
    prompt: str
    n_tokens: int


def build_examples(split, tok, k=16, few_shot=False, drop_long=None):
    if drop_long is None:
        drop_long = split == "train"
    papers, questions = load_split(f"data/qasper/qasper-{split}-v0.3.json")
    ranking = json.load(open(f"data/qasper/retrieval_{split}.json"))["ranking"]
    examples, counts = [], Counter()
    for q in questions:
        paper = papers[q.paper_id]
        context, p_to_idx = build_context(paper, ranking[q.question_id], k)
        target, gold_ps = target_label(q, paper, p_to_idx)
        if target is None:
            counts["skipped"] += 1
            continue
        prompt = format_prompt(tok, context, q.question, few_shot)
        n_tokens = len(tok(prompt)["input_ids"])
        if drop_long and n_tokens > MAX_PROMPT_TOKENS:
            counts["too long"] += 1
            continue
        examples.append(Example(q, paper, p_to_idx, target, gold_ps, prompt, n_tokens))
        counts[target] += 1
    return examples, counts


if __name__ == "__main__":
    import statistics
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    for split in ["train", "dev"]:
        examples, counts = build_examples(split, tok)
        lengths = [e.n_tokens for e in examples]
        print(f"{split}: {dict(counts)}  prompt tokens: median {statistics.median(lengths):.0f}, max {max(lengths)}")
    print("\n--- example prompt ---\n" + examples[0].prompt)