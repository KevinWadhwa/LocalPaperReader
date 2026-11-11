"""PDF -> Paper with pdfplumber only: no GROBID, no Java, no server.

pdf_parse.py needs a GROBID server on localhost:8070, which can't be bundled into a double-click
app. pdfplumber/pdfminer.six are MIT like this project; PyMuPDF would be shorter and handle
two-column order for us but is AGPL-3.0.

Produces only what context.build_context uses: title, paragraphs, one section label each.
Does NOT reproduce QASPER's placeholders (BIBREFn, FIGREFn, ...) -- resolving a citation to a
bibliography entry needs a trained model, which is what GROBID was for. Citations stay literal.
That's safe: quote_check substring-matches paragraph text and never reads placeholders, and
retrieval is BM25+embeddings where author names carry more signal than a BIBREF token. The
returned display map is therefore empty and restore() is a pass-through here.

The fidelity risk is paragraph GRANULARITY, not placeholders: a merged or split paragraph changes
what [P1]/[P2] contain. Measure with parsescripts/check_parse.py before trusting this.

    PYTHONPATH=. python qasper/pdf_parse_v2.py paper.pdf
"""
import argparse
import re
import statistics
import unicodedata
from collections import Counter

from qasper.data import Paper
# Imported, not copied, so a fix to either parser's normalisation fixes both.
from qasper.pdf_parse import MIN_WORDS, _clean

# Captions go LAST under this exact name: that's what data.py:93-96 does building a Paper from
# QASPER JSON, and what context.py keys CAPTION_LABEL off.
CAPTION_SECTION = "Figures and tables"
CAPTION_RE = re.compile(r"^(Figure|Table|Fig\.|Tab\.)\s*\d+\s*[:.]", re.I)
# QASPER excludes the bibliography; left in, it adds hundreds of junk short paragraphs.
STOP_RE = re.compile(r"^\s*(references|bibliography|acknowledgements?|acknowledgments?)\s*$", re.I)
HEAD_MAX_WORDS = 12
MARGIN_FRAC = 0.07          # top/bottom band where running heads and page numbers live


def _norm(s):
    """NFKD so PDF ligatures (fi, fl) become ASCII, then rejoin words hyphenated across a line
    break. QASPER's text has neither artefact."""
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", s)


def _lines(page, margin):
    """page.chars -> one entry per visual line, keeping font size and a bold flag.

    pdfplumber gives characters, not lines, so grouping is ours. extract_text() would be shorter
    but discards the sizes, which are the only signal for heading-vs-body.
    """
    chars = [c for c in page.chars if margin < c["top"] < page.height - margin]
    rows = {}
    for c in chars:
        rows.setdefault(round(c["top"] / 3), []).append(c)   # 3pt bucket tolerates baseline wobble
    out = []
    for key in sorted(rows):
        cs = sorted(rows[key], key=lambda c: c["x0"])
        text = "".join(c["text"] for c in cs)
        if not text.strip():
            continue
        fonts = " ".join(c.get("fontname", "") for c in cs).lower()
        out.append({"top": min(c["top"] for c in cs), "x0": min(c["x0"] for c in cs),
                    "size": statistics.median([c["size"] for c in cs]),
                    "bold": "bold" in fonts or "black" in fonts, "text": text})
    return out


def _split_columns(lines, page_width):
    """Lines in READING order. Two-column papers otherwise sort a left line together with the
    right line beside it and the paragraphs interleave into word salad. Emits all of the left
    column before any of the right."""
    if not lines:
        return []
    mid = page_width / 2
    left = [l for l in lines if l["x0"] < mid * 0.9]
    right = [l for l in lines if l["x0"] > mid * 1.05]
    if len(left) >= 4 and len(right) >= 4 and len(left) + len(right) > 0.8 * len(lines):
        body_top = min(l["top"] for l in left + right)
        full = [l for l in lines if l not in left and l not in right]   # titles, wide tables
        return (sorted([l for l in full if l["top"] < body_top], key=lambda l: l["top"])
                + sorted(left, key=lambda l: l["top"]) + sorted(right, key=lambda l: l["top"])
                + sorted([l for l in full if l["top"] >= body_top], key=lambda l: l["top"]))
    return sorted(lines, key=lambda l: l["top"])


def _drop_running_heads(pages_lines):
    """Drop the same short line repeated on most pages: the margin crop misses running heads that
    sit inside the text box."""
    counts = Counter()
    for lines in pages_lines:
        for l in lines:
            t = l["text"].strip()
            if len(t.split()) <= 8:
                counts[t] += 1
    threshold = max(2, len(pages_lines) // 2)
    repeated = {t for t, n in counts.items() if n >= threshold}
    return [[l for l in lines if l["text"].strip() not in repeated] for lines in pages_lines]


def _blocks(lines, body_size):
    """Group lines into blocks, breaking on a gap bigger than a line or a font-size change."""
    blocks, cur, prev = [], [], None
    for l in lines:
        gap = (l["top"] - prev["top"]) if prev else 0
        if cur and (gap > body_size * 1.8 or abs(l["size"] - prev["size"]) > 0.6):
            blocks.append(cur)
            cur = []
        cur.append(l)
        prev = l
    if cur:
        blocks.append(cur)
    return blocks


def _is_heading(block, body_size):
    """Short, and either larger than body text, bold, or numbered ("3.1 Method")."""
    text = " ".join(l["text"] for l in block).strip()
    if not text or len(text.split()) > HEAD_MAX_WORDS:
        return False
    size = statistics.median([l["size"] for l in block])
    return (size > body_size + 0.5
            or (block[0]["bold"] and len(text.split()) <= 8)
            or bool(re.match(r"^\d+(\.\d+)*\.?\s+\S", text)))


def _section_name(text):
    """"3.1 Approach" -> "Approach", so labels read like QASPER's."""
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", text).strip(" .:")


def parse_pdf_v2(path, paper_id=None):
    """PDF -> (Paper, {}). Empty display map: no placeholders to restore. Signature mirrors
    pdf_parse.parse_pdf so callers can swap backends."""
    import pdfplumber                      # local: only this backend needs the dependency

    pages_lines, widths = [], []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages_lines.append(_lines(page, page.height * MARGIN_FRAC))
            widths.append(page.width)
    pages_lines = _drop_running_heads(pages_lines)

    all_lines = [l for lines, w in zip(pages_lines, widths) for l in _split_columns(lines, w)]
    if not all_lines:
        raise RuntimeError(f"no extractable text in {path} (scanned image? needs OCR)")

    # Body size = modal font size; headings and footnotes are the minority. Smaller-than-body
    # lines are footnotes, dropped as noise.
    size_mode = Counter(round(l["size"] * 2) / 2 for l in all_lines).most_common(1)[0][0]
    all_lines = [l for l in all_lines if l["size"] >= size_mode - 0.6]

    title = _norm(all_lines[0]["text"]).strip() if all_lines else (paper_id or "")
    paragraphs, sections, captions, abstract_parts = [], [], [], []
    section = "Abstract"

    for block in _blocks(all_lines[1:], size_mode):
        text = _clean(_norm(" ".join(l["text"] for l in block)))
        if not text:
            continue
        if STOP_RE.match(text):
            break
        if CAPTION_RE.match(text):
            captions.append(text)
            continue
        if _is_heading(block, size_mode):
            name = _section_name(text)
            section = "Abstract" if re.match(r"^abstract\b", name, re.I) else (name or "Unnamed section")
            continue
        if len(text.split()) < MIN_WORDS:        # orphan equation lines, stray labels
            continue
        if section == "Abstract":
            abstract_parts.append(text)
        paragraphs.append(text)
        sections.append(section)

    for caption in captions:                     # no MIN_WORDS: QASPER keeps "Table 1: Data."
        paragraphs.append(caption)
        sections.append(CAPTION_SECTION)

    paper = Paper(paper_id or "pdf", title, " ".join(abstract_parts), paragraphs, sections)
    return paper, {}


def main():
    ap = argparse.ArgumentParser(description="parse a PDF to a QASPER-style Paper without GROBID")
    ap.add_argument("pdf")
    ap.add_argument("--paper_id", default=None)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    paper, _ = parse_pdf_v2(args.pdf, args.paper_id)
    print(f"title     : {paper.title}")
    print(f"abstract  : {paper.abstract[:200]}{'...' if len(paper.abstract) > 200 else ''}")
    print(f"paragraphs: {len(paper.paragraphs)}   sections: {len(set(paper.sections))}   "
          f"captions: {sum(1 for s in paper.sections if s == CAPTION_SECTION)}")
    for i, (s, p) in enumerate(zip(paper.sections, paper.paragraphs)):
        if i >= args.show:
            break
        print(f"\n  [P{i + 1}] ({s}) {p[:220]}{'...' if len(p) > 220 else ''}")


if __name__ == "__main__":
    main()
