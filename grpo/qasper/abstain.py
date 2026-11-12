import json

from qasper.bm25 import tokenize
from qasper.context import build_context, contains_answer
from qasper.data import table_only
from qasper.prompt import format_prompt, MAX_PROMPT_TOKENS
from qasper.taskdata import Example

NEAR_COPY = 0.7             # a shown paragraph holding this share of a gold paragraph's words counts as the gold
TWIN_COVERAGE = 0.6         # twin rejected if one paragraph covers this share of an abstractive answer's words


def gold_idx(q):
    return {i for a in q.answers if a.kind not in ("unanswerable", "empty") for i in a.evidence_idx}


def extractive_in(q, paras):
    texts = [a.text for a in q.answers if a.kind == "extractive" and a.text]
    return any(contains_answer(" ".join(paras), t) for t in texts)


def near_copy(q, paper, paras):
    golds = [set(tokenize(paper.paragraphs[i])) for i in gold_idx(q)]
    return any(len(g & set(tokenize(s))) >= NEAR_COPY * max(len(g), 1) for g in golds for s in paras)


def subtype(e):
    """answer / unanswerable / evidence_missing / table_missing / table_caption_shown,
    or contaminated: an abstain label whose context still holds the answer."""
    q = e.question
    if e.target == "answer":
        return "answer"
    if q.label == "unanswerable":
        return "unanswerable"
    paras = [e.paper.paragraphs[i] for i in e.p_to_idx.values()]
    if extractive_in(q, paras) or (not table_only(q) and near_copy(q, e.paper, paras)):
        return "contaminated"
    if gold_idx(q) & set(e.p_to_idx.values()):
        return "table_caption_shown"
    return "table_missing" if table_only(q) else "evidence_missing"


def twin(e, ranking, tok, drop_long=False, few_shot=False, k=16):
    """Abstain twin of an answer example: same question, gold paragraphs replaced by the next-ranked ones.
    None if the new context could still answer it. k: paragraphs shown, as in build_examples."""
    q = e.question
    gold = gold_idx(q)
    context, p_to_idx = build_context(e.paper, [i for i in ranking[q.question_id] if i not in gold], k)
    paras = [e.paper.paragraphs[i] for i in p_to_idx.values()]
    if extractive_in(q, paras) or near_copy(q, e.paper, paras):
        return None
    for a in q.answers:
        need = set(tokenize(a.text)) if a.kind == "abstractive" and a.text else set()
        if need and any(len(need & set(tokenize(s))) >= TWIN_COVERAGE * len(need) for s in paras):
            return None
    prompt = format_prompt(tok, context, q.question, few_shot)
    n_tokens = len(tok(prompt)["input_ids"])
    if drop_long and n_tokens > MAX_PROMPT_TOKENS:
        return None
    return Example(q, e.paper, p_to_idx, "abstain", [], prompt, n_tokens)


def twins(examples, split, tok, drop_long=False, few_shot=False, k=16):
    """(answer example, its abstain twin) for every answer example that has a clean twin."""
    ranking = json.load(open(f"data/qasper/retrieval_{split}.json"))["ranking"]
    pairs = [(e, twin(e, ranking, tok, drop_long, few_shot, k)) for e in examples if e.target == "answer"]
    return [(e, t) for e, t in pairs if t is not None]
