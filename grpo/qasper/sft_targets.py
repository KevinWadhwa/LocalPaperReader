import re
from collections import Counter

from qasper.bm25 import tokenize
from qasper.reward import reward, norm_ws, ABSTAIN, MAX_QUOTES, MIN_WORDS, MAX_WORDS
from qasper.taskdata import build_examples

MIN_COVERAGE = 0.4
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
KIND_ORDER = {"extractive": 0, "abstractive": 1, "yes_no": 2}


def units(text):
    out = []
    for s in SENT_SPLIT.split(norm_ws(text)):
        if out and len(out[-1].split()) < MIN_WORDS:
            out[-1] += " " + s
        else:
            out.append(s)
    if len(out) > 1 and len(out[-1].split()) < MIN_WORDS:
        last = out.pop()
        out[-1] += " " + last
    return out


def fit(unit, need):
    words = unit.split()
    if len(words) <= MAX_WORDS:
        return unit
    best = max(range(len(words) - MAX_WORDS + 1),
               key=lambda s: len(need & set(tokenize(" ".join(words[s:s + MAX_WORDS])))))
    return " ".join(words[best:best + MAX_WORDS])


def choose_quotes(cands, need):
    chosen, covered = [], set()
    while len(chosen) < MAX_QUOTES:
        best = max(cands, key=lambda c: len(set(tokenize(c[2])) & (need - covered)))
        gain = set(tokenize(best[2])) & (need - covered)
        if not gain:
            break
        chosen.append(best)
        covered |= gain
        if covered >= need:
            break
    return chosen, covered


def build_target(e):
    if e.target == "abstain":
        return f"<answer>{ABSTAIN}</answer>", "abstain"
    options = sorted((a for a in e.question.answers if a.kind in KIND_ORDER and a.text),
                     key=lambda a: KIND_ORDER[a.kind])
    reason = "no usable answer"
    for a in options:
        answer_words = set(tokenize(a.text))
        need = set(tokenize(e.question.question)) if a.kind == "yes_no" else answer_words
        cands = [(p, j, fit(u, need)) for p in e.gold_ps
                 for j, u in enumerate(units(e.paper.paragraphs[e.p_to_idx[p]]))]
        if not cands:
            continue
        chosen, covered = choose_quotes(cands, need)
        if not chosen:
            reason = "no overlapping sentence"
            continue
        if a.kind == "abstractive" and len(covered) / max(len(answer_words), 1) < MIN_COVERAGE:
            reason = "low coverage"
            continue
        numbers = {w for w in answer_words if any(ch.isdigit() for ch in w)}
        if a.kind == "abstractive" and not numbers <= covered:
            reason = "ungrounded number"
            continue
        out = "\n".join(f'<quote p="{p}">{t}</quote>' for p, _, t in sorted(chosen))
        out += f"\n<answer>{norm_ws(a.text)}</answer>"
        _, outcome = reward(out, e.question, e.paper, e.p_to_idx, e.target, e.gold_ps)
        if outcome == "answer":
            return out, a.kind
        reason = f"reward: {outcome}"
    return None, reason


def build_sft_examples(split, tok):
    examples, _ = build_examples(split, tok)
    pairs, counts = [], Counter()
    for e in examples:
        target, why = build_target(e)
        counts[why if target is None else "built " + why] += 1
        if target is not None:
            pairs.append((e, target))
    return pairs, counts


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    pairs, counts = build_sft_examples("train", tok)
    for k, v in sorted(counts.items()):
        print(f"{v:5d}  {k}")
    n_abstain = sum(e.target == "abstain" for e, _ in pairs)
    print(f"\n{len(pairs)} SFT examples, abstention {n_abstain / len(pairs):.1%}")
    for e, t in pairs[:3]:
        print(f"\n--- {e.question.question}\n{t}")