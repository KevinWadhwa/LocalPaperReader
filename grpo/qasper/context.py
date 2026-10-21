CAPTION_LABEL = "Table/figure caption; contents not shown"

def build_context(paper, ranked_idx, k=10):
    shown = sorted(ranked_idx[:k])
    blocks, p_to_idx = [f"Title: {paper.title}"], {}
    for p, idx in enumerate(shown, start=1):
        section = paper.sections[idx]
        label = CAPTION_LABEL if section == "Figures and tables" else section
        blocks.append(f"[P{p}] ({label}) {paper.paragraphs[idx]}")
        p_to_idx[p] = idx
    return "\n\n".join(blocks), p_to_idx