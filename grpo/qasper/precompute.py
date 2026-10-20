import json
from qasper.data import load_split
from qasper.hybrid import HybridRetriever

def precompute(split, n=50, w=0.3, c=5):
    papers, questions = load_split(f"data/qasper/qasper-{split}-v0.3.json")
    retriever = HybridRetriever(papers, w=w, c=c)
    ranking = {q.question_id: retriever.top_k(q.paper_id, q.question, n) for q in questions}
    path = f"data/qasper/retrieval_{split}.json"
    with open(path, "w") as f:
        json.dump({"method": f"hybrid w={w} c={c}", "n": n, "ranking": ranking}, f)
    print(f"{split}: {len(ranking)} questions -> {path}")

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        precompute(split)