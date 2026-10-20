import numpy as np
from sentence_transformers import CrossEncoder

class RerankRetriever:
    def __init__(self, papers, first_stage, model_name="BAAI/bge-reranker-base",
                 n_candidates=50, device="cuda", batch_size=64):
        self.first_stage = first_stage
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.paragraphs = {p.paper_id: p.paragraphs for p in papers.values()}
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        #self.model.model.half()
        self.cache = {}

    def top_k(self, paper_id, question, k):
        key = (paper_id, question)
        if key not in self.cache:
            candidates = self.first_stage.top_k(paper_id, question, self.n_candidates)
            paras = self.paragraphs[paper_id]
            pairs = [(question, paras[i]) for i in candidates]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            order = np.argsort(-scores)
            self.cache[key] = [candidates[j] for j in order]
        return self.cache[key][:k]

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from qasper.data import load_split
    from qasper.dense import DenseRetriever, recall_at_k

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    papers, questions = load_split("data/qasper/qasper-dev-v0.3.json")
    first_stage = DenseRetriever(papers)
    retriever = RerankRetriever(papers, first_stage)

    n, results = recall_at_k(retriever, papers, questions, [1, 3, 5, 8, 10, 15, 20], tok)
    print(f"dev answerable questions with evidence: {n}")
    print(f"{'k':>4} {'any-hit':>8} {'all-hit':>8} {'context tokens':>15}")
    for k, (a, b, t) in results.items():
        print(f"{k:>4} {a:>8.1%} {b:>8.1%} {t:>15.0f}")