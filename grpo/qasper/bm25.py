import re

from rank_bm25 import BM25Okapi

STOPWORDS = set("""a an the of in on for to and or is are was were be been by with as at from that this
these those it its which what how why who whom when where do does did their they them than then into
about over under between can could should would will may might not no yes""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


class PaperIndex:
    """BM25 over one paper's paragraphs."""

    def __init__(self, paper):
        self.paper = paper
        self.bm25 = BM25Okapi([tokenize(p) or ["_"] for p in paper.paragraphs])

    def top_k(self, question, k):
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return order[:k]


def recall_at_k(papers, questions, ks, tok=None):
    """For answerable questions with text evidence: how often retrieval finds it."""
    indexes = {pid: PaperIndex(p) for pid, p in papers.items()}
    rows = [q for q in questions if q.label == "answerable" and any(a.evidence_idx for a in q.answers)]
    results = {}
    for k in ks:
        hit_any, hit_all, n_tokens = 0, 0, []
        for q in rows:
            got = set(indexes[q.paper_id].top_k(q.question, k))
            gold = [set(a.evidence_idx) for a in q.answers if a.evidence_idx]
            hit_any += any(g & got for g in gold)
            hit_all += any(g <= got for g in gold)
            if tok is not None:
                paper = papers[q.paper_id]
                n_tokens.append(sum(len(tok(paper.paragraphs[i])["input_ids"]) for i in got))
        results[k] = (hit_any / len(rows), hit_all / len(rows),
                      sum(n_tokens) / len(n_tokens) if n_tokens else None)
    return len(rows), results


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from qasper.data import load_split
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    papers, questions = load_split("data/qasper/qasper-dev-v0.3.json")
    n, results = recall_at_k(papers, questions, [1, 3, 5, 8, 10, 15, 20], tok)
    print(f"dev answerable questions with evidence: {n}")
    print(f"{'k':>4} {'any-hit':>8} {'all-hit':>8} {'context tokens':>15}")
    for k, (a, b, t) in results.items():
        print(f"{k:>4} {a:>8.1%} {b:>8.1%} {t:>15.0f}")