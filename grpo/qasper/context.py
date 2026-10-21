from qasper.data import table_only
from qasper.retrieval import tokenize

CAPTION_SECTION = "Figures and tables"

CAPTION_LABEL = "Table/figure caption; contents not shown"

def build_context(paper, ranked_idx, k=10):
    shown = sorted(ranked_idx[:k])
    blocks, p_to_idx = [f"Title: {paper.title}"], {}
    for p, idx in enumerate(shown, start=1):
        section = paper.sections[idx]
        label = CAPTION_LABEL if section == CAPTION_SECTION else (section or "Unnamed section")
        blocks.append(f"[P{p}] ({label}) {paper.paragraphs[idx]}")
        p_to_idx[p] = idx
    return "\n\n".join(blocks), p_to_idx


def contains_answer(text, answer):
    need = set(tokenize(answer))
    return bool(need) and need <= set(tokenize(text))


def target_label(question, paper, p_to_idx):
    if question.label in ("disputed", "empty"):
        return None, []
    if question.label == "unanswerable":
        return "abstain", []
    answers = [a for a in question.answers if a.kind not in ("unanswerable", "empty")]
    gold = {i for a in answers for i in a.evidence_idx}
    if not gold:
        return None, []
    gold_ps = [p for p, idx in p_to_idx.items() if idx in gold]
    if not gold_ps:
        return "abstain", []
    if table_only(question):
        captions = [paper.paragraphs[p_to_idx[p]] for p in gold_ps]
        ok = any(a.kind == "extractive" and contains_answer(c, a.text)
                 for a in answers for c in captions)
        return ("answer", gold_ps) if ok else ("abstain", [])
    return "answer", gold_ps