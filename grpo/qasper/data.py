import json
import statistics
from collections import Counter
from dataclasses import dataclass


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    paragraphs: list    # abstract, then body paragraphs, then figure/table captions
    sections: list      # section name for each paragraph


@dataclass
class Answer:
    kind: str           # extractive / abstractive / yes_no / unanswerable / empty
    text: str | None    # gold answer text (None if unanswerable)
    evidence: list      # evidence strings as given by the annotator
    evidence_idx: list  # positions of the evidence in paper.paragraphs (text or caption)
    has_float: bool     # some evidence is a table/figure
    quotable: bool      # some evidence is real paragraph text the model can quote


@dataclass
class Question:
    question_id: str
    paper_id: str
    question: str
    answers: list       # one Answer per annotator
    label: str          # answerable / unanswerable / disputed / empty
    kind: str           # yes_no / extractive / abstractive / unanswerable / disputed / empty


def find_paragraph(evidence, para_index):
    """Position of an evidence string among the paragraphs (captions included), or None."""
    if evidence.startswith("FLOAT SELECTED"):
        evidence = evidence.split(":", 1)[1].strip()
    return para_index.get(evidence)


def parse_answer(a, para_index):
    if a["unanswerable"]:
        kind, text = "unanswerable", None
    elif a["extractive_spans"]:
        kind, text = "extractive", ", ".join(a["extractive_spans"])
    elif a["free_form_answer"]:
        kind, text = "abstractive", a["free_form_answer"]
    elif a["yes_no"] is not None:
        kind, text = "yes_no", "Yes" if a["yes_no"] else "No"
    else:
        kind, text = "empty", None
    ev = a["evidence"]
    idx = sorted({i for i in (find_paragraph(e, para_index) for e in ev) if i is not None})
    has_float = any(e.startswith("FLOAT SELECTED") for e in ev)
    quotable = any(not e.startswith("FLOAT SELECTED") and e in para_index for e in ev)
    return Answer(kind, text, ev, idx, has_float, quotable)


def label_of(answers):
    kinds = [a.kind for a in answers if a.kind != "empty"]
    if not kinds:
        return "empty"
    n_unans = sum(k == "unanswerable" for k in kinds)
    if n_unans == len(kinds):
        return "unanswerable"
    if n_unans == 0:
        return "answerable"
    return "disputed"


def question_kind(answers, label):
    if label != "answerable":
        return label
    kinds = [a.kind for a in answers if a.kind not in ("unanswerable", "empty")]
    return Counter(kinds).most_common(1)[0][0]


def load_split(path):
    raw = json.load(open(path))
    papers, questions = {}, []
    for pid, p in raw.items():
        paras, secs = [], []
        if p["abstract"].strip():
            paras.append(p["abstract"])
            secs.append("Abstract")
        for sec in p["full_text"]:
            for text in sec["paragraphs"]:
                if text.strip():
                    paras.append(text)
                    secs.append(sec["section_name"] or "")
        for fig in p.get("figures_and_tables", []):
            caption = fig.get("caption", "").strip()
            if caption:
                paras.append(caption)
                secs.append("Figures and tables")
        papers[pid] = Paper(pid, p["title"], p["abstract"], paras, secs)

        para_index = {}
        for i, text in enumerate(paras):
            para_index.setdefault(text, i)

        for qa in p["qas"]:
            answers = [parse_answer(x["answer"], para_index) for x in qa["answers"]]
            label = label_of(answers)
            questions.append(Question(qa["question_id"], pid, qa["question"], answers,
                                      label, question_kind(answers, label)))
    return papers, questions


def usable_for_training(q):
    if q.label == "unanswerable":
        return True
    if q.label != "answerable":
        return False
    return any(a.quotable for a in q.answers)


def table_only(q):
    """Answerable, but every annotator's evidence is a table or figure."""
    answered = [a for a in q.answers if a.kind not in ("unanswerable", "empty")]
    return q.label == "answerable" and all(
        a.evidence and all(e.startswith("FLOAT SELECTED") for e in a.evidence) for a in answered)


def summarise(name, papers, questions, tok=None):
    idx_of = {}
    for pid, p in papers.items():
        idx_of[pid] = {}
        for i, text in enumerate(p.paragraphs):
            idx_of[pid].setdefault(text, i)
    print(f"\n=== {name}: {len(papers)} papers, {len(questions)} questions")
    print("labels:", dict(Counter(q.label for q in questions)))
    print("question kinds:", dict(Counter(q.kind for q in questions)))
    kinds = Counter(a.kind for q in questions for a in q.answers)
    total = sum(kinds.values())
    print("answer kinds:", {k: f"{v / total:.1%}" for k, v in kinds.most_common()})
    answered = [a for q in questions for a in q.answers if a.kind not in ("unanswerable", "empty")]
    n_ev = sum(len(a.evidence) for a in answered)
    n_float = sum(e.startswith("FLOAT SELECTED") for a in answered for e in a.evidence)
    n_found = sum(len(a.evidence_idx) for a in answered)
    n_float_found = sum(e.startswith("FLOAT SELECTED") and find_paragraph(e, idx_of[q.paper_id]) is not None
                        for q in questions for a in q.answers for e in a.evidence)
    print(f"evidence strings: {n_ev}, tables/figures {n_float / n_ev:.1%}, found (text or caption) {n_found / n_ev:.1%}")
    print(f"table/figure evidence matched to a caption: {n_float_found / max(n_float, 1):.1%}")
    print(f"table-only questions: {sum(table_only(q) for q in questions)}")
    print(f"usable for training: {sum(usable_for_training(q) for q in questions)} of {len(questions)}")
    if tok is not None:
        lengths = [len(tok("\n\n".join(p.paragraphs))["input_ids"]) for p in papers.values()]
        q = statistics.quantiles(lengths, n=10)
        print(f"paper tokens: median {statistics.median(lengths):.0f}, 10th {q[0]:.0f}, 90th {q[-1]:.0f}, max {max(lengths)}")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    for split in ["train", "dev"]:
        papers, questions = load_split(f"data/qasper/qasper-{split}-v0.3.json")
        summarise(split, papers, questions, tok)