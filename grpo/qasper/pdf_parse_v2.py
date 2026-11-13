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


# These PDFs encode word gaps as x-offsets, not space chars: all three test papers have ZERO
# literal spaces, so joining page.chars gives "Wecrowdsourceddogmatism...". pdfplumber infers the
# gaps given a tolerance, hence extract_words rather than page.chars.
X_TOL = 1.0


def _gutter(page):
    """x of the two-column gutter, or None for a single-column page.

    The emptiest 10pt bin in the middle third, and only if it is well below average density, so
    full-width pages (wide figures, tables) are left unsplit rather than cut in half. Measured on
    the test papers: a stable ~305 on two-column pages, None where a float spans the page.
    """
    xs = [c["x0"] for c in page.chars]
    if not xs:
        return None
    bins = Counter(int(x // 10) * 10 for x in xs)
    lo, hi = int(page.width * 0.35 // 10) * 10, int(page.width * 0.65 // 10) * 10
    count, b = min((bins.get(k, 0), k) for k in range(lo, hi + 10, 10))
    return b + 5 if count < (len(xs) / (page.width / 10)) * 0.35 else None


def _column_lines(page, x0, x1, margin, top=None):
    """Lines of one column, each with median font size and a bold flag. extract_words does the
    space inference and left-to-right order; it also carries the font metadata that heading
    detection needs and extract_text discards. `top` overrides where the column starts, to skip
    front matter; the BOTTOM stays at the margin either way."""
    box = page.crop((x0, margin if top is None else top, x1, page.height - margin))
    # upright=False drops rotated text, i.e. the arXiv id stamped down the left margin, which
    # otherwise leaks stray single chars into paragraphs and section names.
    words = [w for w in box.extract_words(extra_attrs=["size", "fontname", "upright"],
                                          x_tolerance=X_TOL) if w.get("upright", True)]
    rows = {}
    for w in words:
        rows.setdefault(round(w["top"] / 3), []).append(w)   # 3pt bucket tolerates baseline wobble
    out = []
    for key in sorted(rows):
        ws = sorted(rows[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        fonts = " ".join(w.get("fontname", "") for w in ws).lower()
        out.append({"top": min(w["top"] for w in ws), "x0": min(w["x0"] for w in ws),
                    "size": statistics.median([w["size"] for w in ws]),
                    # "-medi" because this corpus's bold is NimbusRomNo9L-Medi: no "bold"/"black"
                    "bold": any(k in fonts for k in ("bold", "black", "-medi")), "text": text})
    return out


def _page_lines(page, margin, top=None):
    """A page's lines in READING order: left column fully, then right. Otherwise a left-column
    line sorts with the right-column line beside it and paragraphs interleave mid-sentence.
    `top` starts the page below already-consumed front matter."""
    g = _gutter(page)
    if g is None:
        return _column_lines(page, 0, page.width, margin, top)
    left = _column_lines(page, 0, g, margin, top)
    right = _column_lines(page, g, page.width, margin, top)
    # A thin column means the gutter was spurious (a full-width float, or front matter that
    # straddles it and comes back truncated on both sides), so read the page whole instead.
    if len(left) < 4 or len(right) < 4:
        return _column_lines(page, 0, page.width, margin, top)
    return left + right


def _front_matter(page, margin):
    """Page 1's full-width head: (title, abstract, body_top).

    The title/author/abstract region spans the page while the body below it is two-column, so a
    single column split truncates the title mid-word ("Identifying Dogmatism in Soci"). Read it
    full-width: title = the largest-font line, abstract = everything between the Abstract heading
    and the next heading. body_top is where the two-column body starts.
    """
    full = _column_lines(page, 0, page.width, margin)
    if not full:
        return "", "", margin
    # Title = every line sharing the largest font size, since a long title wraps.
    top_size = max(l["size"] for l in full)
    title = _norm(" ".join(l["text"] for l in full if abs(l["size"] - top_size) < 0.3)).strip()

    head = next((l for l in full if re.match(r"^abstract\b", l["text"], re.I)), None)
    if head is None:                                    # no Abstract heading: nothing to consume
        return title, "", margin
    # The abstract sits in ONE column while the introduction already runs in the other, so
    # reading full-width splices the two together: take it from the heading's own column.
    g = _gutter(page)
    x0, x1 = (0, page.width) if g is None else ((0, g) if head["x0"] < g else (g, page.width))
    col = [l for l in _column_lines(page, x0, x1, margin) if l["top"] > head["top"]]
    if not col:
        return title, "", margin
    # The abstract is a single typographic block, so it runs while the size matches its own first
    # line and ends at the first line of a different size -- which is the next section heading.
    # Keyed on size rather than a heading word list (which would miss anything unnumbered or
    # non-English) or on boldness (some papers set the whole abstract bold).
    size = col[0]["size"]
    parts = []
    for l in col:
        if abs(l["size"] - size) > 0.4:
            return title, _clean(_norm(" ".join(parts))), l["top"]
        parts.append(l["text"])
    return title, _clean(_norm(" ".join(parts))), col[-1]["top"]


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

    pages_lines = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            margin = page.height * MARGIN_FRAC
            if i == 0:
                title, abstract, body_top = _front_matter(page, margin)
                pages_lines.append(_page_lines(page, margin, top=body_top))
            else:
                pages_lines.append(_page_lines(page, margin))
    pages_lines = _drop_running_heads(pages_lines)

    all_lines = [l for lines in pages_lines for l in lines]
    if not all_lines:
        raise RuntimeError(f"no extractable text in {path} (scanned image? needs OCR)")

    # Body size = modal font size; headings and footnotes are the minority. Smaller-than-body
    # lines are footnotes, dropped as noise.
    size_mode = Counter(round(l["size"] * 2) / 2 for l in all_lines).most_common(1)[0][0]
    all_lines = [l for l in all_lines if l["size"] >= size_mode - 0.6]

    title = title or (paper_id or "")
    paragraphs, sections, captions = [], [], []
    section = "Unnamed section"
    if abstract:                                  # abstract is paragraph 0, as in QASPER
        paragraphs.append(abstract)
        sections.append("Abstract")

    for block in _blocks(all_lines, size_mode):
        text = _clean(_norm(" ".join(l["text"] for l in block)))
        if not text:
            continue
        if STOP_RE.match(text):
            break
        if CAPTION_RE.match(text):
            captions.append(text)
            continue
        if _is_heading(block, size_mode):
            section = _section_name(text) or "Unnamed section"
            continue
        if len(text.split()) < MIN_WORDS:        # orphan equation lines, stray labels
            continue
        paragraphs.append(text)
        sections.append(section)

    for caption in captions:                     # no MIN_WORDS: QASPER keeps "Table 1: Data."
        paragraphs.append(caption)
        sections.append(CAPTION_SECTION)

    paper = Paper(paper_id or "pdf", title, abstract, paragraphs, sections)
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
