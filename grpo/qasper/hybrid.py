from collections import defaultdict

from qasper.bm25 import PaperIndex
from qasper.dense import DenseRetriever, recall_at_k


class HybridRetriever:
    """Reciprocal rank fusion of BM25 and dense rankings."""

    def __init__(self, papers, c=60):
        self.papers = papers
        self.bm25 = {pid: PaperIndex(p) for pid, p in papers.items()}
        self.dense = DenseRetriever(papers)
        self.c = c

    def top_k(self, paper_id, question, k):
        n = len(self.papers[paper_id].paragraphs)
        fused = defaultdict(float)
        for ranks in (self.bm25[paper_id].top_k(question, n),
                      self.dense.top_k(paper_id, question, n)):
            for position, i in enumerate(ranks):
                fused[i] += 1.0 / (self.c + position + 1)
        return sorted(fused, key=fused.get, reverse=True)[:k]


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from qasper.data import load_split

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    papers, questions = load_split("data/qasper/qasper-dev-v0.3.json")
    retriever = HybridRetriever(papers)

    n, results = recall_at_k(retriever, papers, questions, [1, 3, 5, 8, 10, 15, 20], tok)
    print(f"dev answerable questions with evidence: {n}")
    print(f"{'k':>4} {'any-hit':>8} {'all-hit':>8} {'context tokens':>15}")
    for k, (a, b, t) in results.items():
        print(f"{k:>4} {a:>8.1%} {b:>8.1%} {t:>15.0f}")