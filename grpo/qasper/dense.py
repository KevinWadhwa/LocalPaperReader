import torch
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Embedding search over each paper's paragraphs."""

    def __init__(self, papers, model_name="BAAI/bge-small-en-v1.5", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.emb = {
            pid: self.model.encode(p.paragraphs, batch_size=64,
                                   normalize_embeddings=True, convert_to_tensor=True)
            for pid, p in papers.items()
        }

    def top_k(self, paper_id, question, k):
        q = self.model.encode([QUERY_PREFIX + question],
                              normalize_embeddings=True, convert_to_tensor=True)
        scores = (self.emb[paper_id] @ q.T).squeeze(1)
        k = min(k, scores.shape[0])
        return torch.topk(scores, k).indices.tolist()


def recall_at_k(retriever, papers, questions, ks, tok=None):
    """For answerable questions with located evidence: how often the top k contains it."""
    rows = [q for q in questions if q.label == "answerable" and any(a.evidence_idx for a in q.answers)]
    results = {}
    for k in ks:
        hit_any, hit_all, n_tokens = 0, 0, []
        for q in rows:
            got = set(retriever.top_k(q.paper_id, q.question, k))
            gold = [set(a.evidence_idx) for a in q.answers if a.evidence_idx]
            hit_any += any(g & got for g in gold)
            hit_all += any(g <= got for g in gold)
            if tok is not None:
                paper = papers[q.paper_id]
                n_tokens.append(sum(len(tok(paper.paragraphs[i])["input_ids"]) for i in got))
        results[k] = (hit_any / len(rows), hit_all / len(rows),
                      sum(n_tokens) / len(n_tokens) if n_tokens else float("nan"))
    return len(rows), results


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from qasper.data import load_split

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    papers, questions = load_split("data/qasper/qasper-dev-v0.3.json")
    retriever = DenseRetriever(papers)

    n, results = recall_at_k(retriever, papers, questions, [1, 3, 5, 8, 10, 15, 20], tok)
    print(f"dev answerable questions with evidence: {n}")
    print(f"{'k':>4} {'any-hit':>8} {'all-hit':>8} {'context tokens':>15}")
    for k, (a, b, t) in results.items():
        print(f"{k:>4} {a:>8.1%} {b:>8.1%} {t:>15.0f}")