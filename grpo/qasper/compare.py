"""Compare BM25, dense, BM25+dense fusion and reranking on QASPER dev, by k, by token budget and by evidence type."""
from transformers import AutoTokenizer

from qasper.bm25 import PaperIndex
from qasper.data import load_split
from qasper.dense import DenseRetriever
from qasper.rerank import RerankRetriever

N = 50
KS = [1, 3, 5, 10, 20]
BUDGETS = [250, 500, 1000, 2000]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
papers, questions = load_split("data/qasper/qasper-dev-v0.3.json")
rows = [q for q in questions if q.label == "answerable" and any(a.evidence_idx for a in q.answers)]
lens = {pid: [len(tok(t)["input_ids"]) for t in p.paragraphs] for pid, p in papers.items()}

bm25 = {pid: PaperIndex(p) for pid, p in papers.items()}
dense = DenseRetriever(papers)
rerank = RerankRetriever(papers, dense, model_name="BAAI/bge-reranker-v2-m3", n_candidates=N)


def fuse(a, b, w, c):
    """Weighted reciprocal rank fusion of two rankings of the same paragraphs; w is the weight on a."""
    score = {i: w / (c + pos + 1) for pos, i in enumerate(a)}
    for pos, i in enumerate(b):
        score[i] += (1 - w) / (c + pos + 1)
    return sorted(score, key=score.get, reverse=True)


# full rankings over every paragraph, as HybridRetriever fuses them
rankings = {"bm25": [], "dense": []}
for q in rows:
    n = len(papers[q.paper_id].paragraphs)
    rankings["bm25"].append(bm25[q.paper_id].top_k(q.question, n))
    rankings["dense"].append(dense.top_k(q.paper_id, q.question, n))
# w=0.5 c=60 is the current HybridRetriever
for w in (0.2, 0.3, 0.5):
    for c in (5, 60):
        rankings[f"hybrid w={w} c={c}"] = [fuse(b, d, w, c)
                                           for b, d in zip(rankings["bm25"], rankings["dense"])]
# v2-m3 over the dense top N
rankings["rerank"] = [rerank.top_k(q.paper_id, q.question, N) for q in rows]


def any_hit(q, got):
    return any(set(a.evidence_idx) & got for a in q.answers if a.evidence_idx)


def take_budget(ranking, lengths, budget):
    """Top paragraphs in rank order until the next one would overflow the budget."""
    got, used = set(), 0
    for i in ranking:
        if used + lengths[i] > budget:
            break
        got.add(i)
        used += lengths[i]
    return got


def has_caption(q):
    secs = papers[q.paper_id].sections
    return any(secs[i] == "Figures and tables" for a in q.answers for i in a.evidence_idx)


def table(title, cols, idx, cell, fmt=">8.1%"):
    print(f"\n{title}  (n={len(idx)})")
    print(f"{'method':<18}" + "".join(f"{c:>8}" for c in cols))
    for name, ranks in rankings.items():
        vals = [sum(cell(rows[j], ranks[j], c) for j in idx) / len(idx) for c in cols]
        print(f"{name:<18}" + "".join(format(v, fmt) for v in vals))


hit_k = lambda q, r, k: any_hit(q, set(r[:k]))
hit_b = lambda q, r, b: any_hit(q, take_budget(r, lens[q.paper_id], b))
toks_k = lambda q, r, k: sum(lens[q.paper_id][i] for i in r[:k])

all_idx = list(range(len(rows)))
cap_idx = [j for j in all_idx if has_caption(rows[j])]
text_idx = [j for j in all_idx if not has_caption(rows[j])]

table("any-hit @ k", KS, all_idx, hit_k)
table("avg context tokens @ k", KS, all_idx, toks_k, fmt=">8.0f")
table("any-hit @ token budget", BUDGETS, all_idx, hit_b)
table("any-hit @ k, gold includes a caption", KS, cap_idx, hit_k)
table("any-hit @ k, text-only gold", KS, text_idx, hit_k)
