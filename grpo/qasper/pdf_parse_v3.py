"""PDF to a QASPER-style Paper using pdfplumber alone (no GROBID, Java or server). The model was trained on
QASPER's text, which comes from LaTeX sources, so the parser rebuilds that from the PDF's typography: one
paragraph per LaTeX paragraph, nested section names, BIBREFn citations, TABREFn / FIGREFn / SECREFn references,
DISPLAYFORMn equations and captions last. restore() turns the placeholders back into the paper's text for display.

    PYTHONPATH=. python qasper/pdf_parse_v3.py paper.pdf [--show 8]
"""
# Conventions, measured on qasper-train-v0.3 (888 papers, 47,687 paragraphs). QASPER's text is S2ORC's parse of
# the LaTeX source, so the parser rebuilds LaTeX structure from the PDF's typography:
#
#   paragraphs  one per LaTeX paragraph. In the PDF a paragraph starts at a first-line indent, after a short
#               last line, after a vertical gap or at a list marker, and runs on across column and page
#               breaks. Run-in heads stay inline ("Data collection. Subreddits are ..."). v2 split only on
#               vertical gaps, which two-column LaTeX almost never has, so it merged most paragraphs.
#   sections    heading text without its number; subsections nest as "Parent ::: Child". All-caps headings
#               are recased ("RELATED WORK" -> "Related Work"), keeping acronyms the body writes in caps.
#   citations   BIBREFn, n in order of first citation (monotone in 77% of QASPER papers), so no bibliography
#               model is needed. Brackets go: "(Kim et al., 2019; Lee, 2018)" -> "BIBREF0, BIBREF1".
#               Narrative author-year drops the names: "Kim et al. (2019) report" -> "BIBREF0 report".
#               Numeric keeps them: "Kim et al. [3]" -> "Kim et al. BIBREF0". The bibliography entry, when
#               one matches, goes in the display map; it is not used to accept or reject a citation.
#   refs        "Table 2" -> "Table TABREFn", likewise FIGREF, SECREF ("Section", "Appendix") and EQREF; n is a
#               shared counter in order of first mention, like QASPER's (arbitrary) label numbers.
#   equations   a displayed equation becomes "DISPLAYFORM0" at the END of the paragraph that introduces it
#               (numbering restarts per paragraph); the "where ..." after it starts a new paragraph.
#   dropped     footnotes and their markers, table and figure contents, running heads, page numbers, front
#               matter (CCS concepts, keywords), and the bibliography. Acknowledgements and appendices stay.
#   captions    last, under "Figures and tables", citations left raw, as in QASPER.
#   characters  ligatures expanded, line-break hyphens rejoined, ’ -> ' in body text (QASPER's body has straight
#               apostrophes; its captions come from the PDF and keep ’).
#
# Not reproduced: inline math (QASPER has $...$ or INLINEFORMn from the LaTeX; a PDF has only glyphs, which
# are kept as extracted).
#
# The display map is {placeholder: (original text, raw reference or None)}; restore() uses it.
import argparse
import re
import statistics
import unicodedata
from collections import Counter

from qasper.data import Paper

CAPTION_SECTION = "Figures and tables"
MIN_WORDS = 5               # as pdf_parse.py: shorter blocks are orphan labels and equation fragments
SPACE_RATIO = 0.1           # gap / font size: letters in a word measure < 0.04 apart, real spaces > 0.16
CELL_GAP = 0.8              # gap (in ems) that splits a row into cells: the gutter, or table columns

CAPTION_RE = re.compile(r"^(?:Figure|Fig\.|FIGURE|FIG\.|Table|TABLE|Tab\.)\s*(?:\d+|[IVXL]+)(?:\s*[:.|]|\s*$)")
HEAD_NUM_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)+\.?|[IVX]+\.|[A-Z]\.|[A-Z](?=\s+[A-Z]))\s+(?=\S)")
REFS_RE = re.compile(r"^(?:(?:bibliographical )?references?(?: and notes)?|bibliography|literature cited|works cited)$",
                     re.I)
ABSTRACT_RE = re.compile(r"^\s*abstract\b[\s.:—–-]*", re.I)
FRONT_RE = re.compile(r"^(?:index terms|keywords|key words|ccs concepts|general terms|categories and subject "
                      r"descriptors|acm reference format|contents|table of contents|list of figures|list of tables)\b", re.I)
LIST_RE = re.compile(r"^(?:[•●▪◦‣∙·–—-]|\(?\d{1,2}[.)]|\(?[a-z][.)]|\((?:i{1,3}|iv|v|vi{0,3})\))\s")
EQNUM_RE = re.compile(r"\((?:\d+|[A-Z]\.\d+)[a-z]?\)$")
TERMINAL_RE = re.compile(r"[.!?:;\"”')\]]$")
BOLD_RE = re.compile(r"bold|black|heavy|demi|medi|cmbx|sfbx|cmb\d|-bd\b|semibold", re.I)
ITALIC_RE = re.compile(r"ital|oblique|cmti|sfti|-it\b", re.I)
MATH_RE = re.compile(r"cmmi|cmsy|cmex|msbm|msam|cmbsy|cmmib|eufm|eusm|rsfs|math|symbol|stix|txmi|txsy|txex|"
                     r"pxmi|pxsy|ntxmi|ntxsy|mtmi|mtsy|euex|eurm|wasy|lmmi|lmsy", re.I)
MARKER_CHARS = set("0123456789*∗†‡§¶⋆,")
TEX = {"∈": "\\in ", "≤": "\\le ", "≥": "\\ge ", "×": "\\times ", "→": "\\rightarrow ", "←": "\\leftarrow ",
       "∑": "\\sum ", "∞": "\\infty ", "·": "\\cdot ", "≠": "\\ne ", "≈": "\\approx ", "∼": "\\sim ", "−": "-",
       "′": "^{\\prime }", "∗": "*"}
SMALL_WORDS = {"a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "by", "at", "from", "per",
               "via", "vs", "as", "into", "over"}


# ----------------------------------------------------------------------------------------------- lines

def _font(c):
    return c.get("fontname", "").split("+")[-1]


class _Line:
    """One row of text inside one column (col -1: spans the gutter). cells > 1: the row has table-like gaps."""
    __slots__ = ("page", "col", "chunk", "x0", "x1", "top", "bottom", "size", "maxsize", "text", "bold",
                 "italic", "math", "caps", "lead_bold", "fn_start", "cells", "kind", "textlike", "tex")

    def __repr__(self):
        return f"<{self.kind} p{self.page} c{self.col} x{self.x0:.0f}-{self.x1:.0f} y{self.top:.0f} {self.text[:40]!r}>"


def _page_words(page):
    words = page.extract_words(x_tolerance_ratio=SPACE_RATIO, y_tolerance=2, keep_blank_chars=False,
                               extra_attrs=["upright"], return_chars=True)
    out = []
    for w in words:
        if not w["upright"]:                    # the arXiv id stamped down the margin
            continue
        chars = [c for c in w["chars"] if c["text"].strip()]
        if not chars:
            continue
        w["chars"] = chars
        w["size"] = statistics.median(c["size"] for c in chars)
        for c in chars:
            c["base"] = -c["matrix"][5]             # baseline, y down
        w["base"] = statistics.median(c["base"] for c in chars)
        out.append(w)
    return out


def _rows(words, body):
    """Words -> rows sharing a baseline. The baseline is the text matrix's, not the glyph box's, so fonts with
    different descents agree and sub/superscripts (shifted by under half a line) stay in their row."""
    rows = []
    for w in sorted(words, key=lambda w: w["base"]):
        if rows and w["base"] - rows[-1][len(rows[-1]) // 2]["base"] <= 0.45 * body:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def _cells(row, body):
    """Split a row (sorted words) at gaps wider than CELL_GAP ems."""
    cells = [[row[0]]]
    for a, b in zip(row, row[1:]):
        if b["x0"] - a["x1"] > CELL_GAP * body:
            cells.append([b])
        else:
            cells[-1].append(b)
    return cells


def _texchar(t):
    r"""'α' -> '\alpha ' (S2ORC keeps the space after a command), '∈' -> '\in '."""
    name = unicodedata.name(t, "")
    if name.startswith("GREEK SMALL LETTER "):
        return "\\" + name[19:].lower().replace("lamda", "lambda") + " "
    if name.startswith("GREEK CAPITAL LETTER "):
        return "\\" + name[21:].capitalize().replace("Lamda", "Lambda") + " "
    return TEX.get(t, t)


def _texword(w, size, base):
    """One word -> text with its math as $...$, like QASPER's LaTeX: the span from the first to the last
    math-font or sub/superscript character, with x_1 / x^{ab} for smaller characters off the baseline."""
    level = [0 if c["size"] >= 0.85 * size else -1 if c["base"] > base + 0.08 * size
             else 1 if c["base"] < base - 0.2 * size else 0 for _, c in w]
    math = [bool(MATH_RE.search(_font(c))) or "GREEK" in unicodedata.name(t[:1], "") for t, c in w]
    idx = [k for k in range(len(w)) if (math[k] and w[k][0] not in ",.;:") or level[k]]
    if not any(math) or not idx:
        return "".join(t for t, _ in w)
    a, b = idx[0], idx[-1]
    out, k = "", a
    while k <= b:                                   # runs on the baseline, and runs off it: subscripts, then
        j = k                                       # superscripts (stacked ones interleave left to right)
        while j <= b and (level[j] == 0) == (level[k] == 0):
            j += 1
        for lv, mark in ((0, ""), (-1, "_"), (1, "^")):
            run = "".join(_texchar(w[i][0]) for i in range(k, j) if level[i] == lv)
            if run:
                out += mark + (run if lv == 0 or len(run) == 1 else "{" + run + "}")
        k = j
    return "".join(t for t, _ in w[:a]) + "\x02" + out + "\x03" + "".join(t for t, _ in w[b + 1:])


def _render(chars, tex=False):
    """Characters of one line -> text. Spaces come from gaps (these PDFs have no space characters); raised
    footnote markers after a word are dropped; small caps get their case back. tex=True also writes inline
    math as $...$, merging neighbours joined by operators: "$x = (x_1, ..., x_m$)"."""
    chars = sorted(chars, key=lambda c: c["x0"])
    size = statistics.median(c["size"] for c in chars)
    base = statistics.median(c["base"] for c in chars if c["size"] >= 0.9 * size)
    words, cur, prev, in_marker = [], [], None, False
    for i, c in enumerate(chars):
        t = c["text"]
        if t.startswith("(cid:"):
            continue
        if prev is not None and c["x0"] - prev["x1"] > SPACE_RATIO * max(c["size"], prev["size"]):
            if cur:
                words.append(cur)
            cur = []
        raised = c["size"] < 0.85 * size and c["base"] < base - 0.2 * size
        if raised and t in MARKER_CHARS:
            before = cur[-1] if cur else None
            if in_marker or (before and not MATH_RE.search(_font(before[1]))
                             and (before[0].isalpha() or before[0] in ".,;:)”\"'")):
                in_marker = True
                prev = c
                continue
        in_marker = False
        cur.append((t, c))
        prev = c
    if cur:
        words.append(cur)
    out = []
    for w in words:
        letters = [(t, c) for t, c in w if t.isalpha()]
        if len(letters) >= 2 and all(t.isupper() for t, _ in letters):
            big = max(c["size"] for _, c in letters)
            if big / min(c["size"] for _, c in letters) > 1.12:     # small caps: true capitals are bigger
                w = [(t.lower() if t.isalpha() and c["size"] < 0.92 * big else t, c) for t, c in w]
        out.append(_texword(w, size, base) if tex else "".join(t for t, _ in w))
    text = re.sub("\x03((?:[ =+\\-−<>,()|\\d]|\\.\\.\\.)*)\x02", r"\1", " ".join(out))
    text = re.sub("(\x02[^\x03]*\\([^\x03)]*)\x03\\)", "\\1)\x03", text)       # "$f(x$)" -> "$f(x)$"
    text = re.sub("\x03( ?[=<>≤≥] ?-?[\\d.]+)(?![\\w.])", "\\1\x03", text)        # "$L$ = 6" -> "$L = 6$"
    return unicodedata.normalize("NFKC", text.replace("\x02", "$").replace("\x03", "$"))


def _make_line(page_no, col, words, n_cells, body_font):
    chars = [c for w in words for c in w["chars"]]
    alpha = [c for c in chars if c["text"].isalnum()] or chars
    fonts = [_font(c) for c in alpha]
    l = _Line()
    l.page, l.col, l.chunk, l.kind, l.cells, l.textlike = page_no, col, None, None, n_cells, False
    l.x0, l.x1 = min(w["x0"] for w in words), max(w["x1"] for w in words)
    l.top, l.bottom = min(w["top"] for w in words), max(w["bottom"] for w in words)
    l.size = statistics.median(c["size"] for c in chars)
    l.maxsize = max(c["size"] for c in alpha)
    is_bold = lambda f: f != body_font and bool(BOLD_RE.search(f))
    l.bold = sum(map(is_bold, fonts)) / len(fonts)
    l.italic = sum(bool(ITALIC_RE.search(f)) for f in fonts) / len(fonts)
    l.math = sum(bool(MATH_RE.search(f)) and f != body_font for f in fonts) / len(fonts)
    letters = [c["text"] for c in chars if c["text"].isalpha()]
    l.caps = sum(t.isupper() for t in letters) / len(letters) if letters else 0.0
    first = min(chars, key=lambda c: c["x0"])
    base = statistics.median(c["base"] for c in chars if c["size"] >= 0.9 * l.size)
    l.fn_start = (first["text"] in MARKER_CHARS and first["size"] < 0.85 * l.size
                  and first["base"] < base - 0.2 * l.size)
    l.text = _render(chars)
    l.tex = _render(chars, tex=True) if l.math else l.text       # body paragraphs use this one
    lead = []                                      # leading bold words: a run-in head "Data collection."
    for w in sorted(words, key=lambda w: w["x0"]):
        al = [c for c in w["chars"] if c["text"].isalnum()]
        if al and all(is_bold(_font(c)) for c in al):
            lead.append(w["text"])
        else:
            break
    l.lead_bold = " ".join(lead) if len(lead) < len(words) else ""
    return l


# ------------------------------------------------------------------------------------------- geometry

def _geometry(segments, body, width):
    """Text columns [(left, right), ...] from where full body-size lines start and end. Two columns when a
    second start position, far from the first, is common too."""
    lines = [(x0, x1) for x0, x1, size in segments if abs(size - body) <= 0.6 and x1 - x0 >= 0.2 * width]
    if not lines:
        return [(0, width)]
    starts = Counter(round(x0) for x0, _ in lines)
    support = lambda x, cnt: sum(n for s, n in cnt.items() if abs(s - x) <= 1)
    lefts = [max(starts, key=lambda s: support(s, starts))]
    far = [s for s in starts if abs(s - lefts[0]) > 0.3 * width]
    if far:
        second = max(far, key=lambda s: support(s, starts))
        if support(second, starts) >= 0.25 * support(lefts[0], starts):
            lefts.append(second)
    cols = []
    for left in sorted(lefts):
        ends = Counter(round(x1) for x0, x1 in lines if abs(x0 - left) <= 1.5)
        # the margin is the leftmost well-used start: list text can outnumber the lines at the margin itself
        outer = [s for s in starts if left - 3 * body <= s < left - 1 and support(s, starts) >= 0.2 * support(left, starts)]
        cols.append((min(outer, default=left), max(ends, key=lambda e: support(e, ends))))
    if len(cols) == 2 and cols[0][1] >= cols[1][0]:
        cols = [(cols[0][0], cols[1][1])]
    return cols


def _assign(rows, cols, body):
    """Rows -> (col, words, n_cells). In a two-column layout each cell lies in one column or crosses the
    gutter (col -1: title, wide float, caption). Only the crossing cells leave their columns, so a body
    line that shares a baseline with a figure label stays in its column."""
    out = []
    if len(cols) == 1:
        for row in rows:
            out.append((0, row, len(_cells(row, body))))
        return out
    mid = (cols[0][1] + cols[1][0]) / 2
    for row in rows:
        cells = []
        for cell in _cells(row, body):
            x0, x1 = cell[0]["x0"], cell[-1]["x1"]
            if x1 <= mid + 2 or x0 >= mid - 2:
                cells.append(cell)
                continue
            # a narrow gutter can be under CELL_GAP: split at a real gap that straddles it
            cut = next((k for k in range(1, len(cell)) if cell[k - 1]["x1"] <= mid + 1 and cell[k]["x0"] >= mid - 1
                        and cell[k]["x0"] - cell[k - 1]["x1"] > 0.6 * cell[k]["size"]), None)   # not a word space
            cells.extend([cell[:cut], cell[cut:]] if cut else [cell])
        side = [0 if c[-1]["x1"] <= mid + 2 else 1 if c[0]["x0"] >= mid - 2 else -1 for c in cells]
        for col in (-1, 0, 1):
            ws = [w for c, s in zip(cells, side) if s == col for w in c]
            if ws:
                out.append((col, ws, sum(s == col for s in side) + (sum(s != col for s in side) if col == -1 else 0)))
    return out


def _reading_order(lines):
    """A page's lines in reading order: full-width lines cut the page into bands, and each band is read
    left column, then right column. `chunk` names the (page, band, column) a line was read in."""
    spans = sorted((l for l in lines if l.col == -1), key=lambda l: l.top)
    cols = sorted((l for l in lines if l.col >= 0), key=lambda l: l.top)
    out, band, i = [], 0, 0
    for s in spans + [None]:
        stop = s.top if s is not None else float("inf")
        group = []
        while i < len(cols) and cols[i].top < stop:
            group.append(cols[i])
            i += 1
        if group:
            band += 1
        for col in (0, 1):
            for l in group:
                if l.col == col:
                    l.chunk = (l.page, band, col)
                    out.append(l)
        if s is not None:
            if group or not out or out[-1].col != -1:
                band += 1
            s.chunk = (s.page, band, -1)
            out.append(s)
    return out


# ------------------------------------------------------------------------------------------- text glue

_TOKEN_RE = re.compile(r"[^\W\d_][\w'-]*", re.U)


def _vocab(lines):
    return {t.lower() for l in lines for t in _TOKEN_RE.findall(l.text) if not t.endswith("-")}


def _join(texts, vocab):
    """Join line texts; a word hyphenated across the break is rejoined unless the hyphen is real: the joined
    form is never used and either the hyphenated form is, both halves are words, or the second half is
    capitalised ("Danescu-Niculescu-Mizil")."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        m = re.search(r"([\w'-]*[^\W\d_])-$", out)
        n = re.match(r"[^\W\d_][\w'-]*", t)
        if m and n:
            head, tail = m.group(1), n.group(0)
            joined, hyph = (head + tail).lower(), f"{head}-{tail}".lower()
            keep = joined not in vocab and (tail[0].isupper() or "-" in head or hyph in vocab
                                            or (head.lower() in vocab and tail.lower() in vocab))
            out = out + t if keep else out[:-1] + t
        elif out.endswith(("–", "—", "/")) and t[:1].isalnum():
            out += t
        else:
            out = f"{out} {t}" if out else t
    return out


def _tidy(s):
    s = s.replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^\s*[•●▪◦‣∙]\s*", "", s)                # list bullets: QASPER drops them
    return s


def _recase(name, acronyms):
    """'TEXT-BASED INDUSTRY MODELING' -> 'Text-based Industry Modeling'; words the body writes in caps stay."""
    letters = [c for c in name if c.isalpha()]
    if len(letters) < 2 or sum(c.isupper() for c in letters) < 0.9 * len(letters):
        return name
    out = []
    for k, w in enumerate(name.split()):
        core = re.sub(r"[^\w-]", "", w)
        if core in acronyms or not any(c.isalpha() for c in w):
            out.append(w)
        elif k > 0 and w.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(w[:1] + w[1:].lower())
    return " ".join(out)


# -------------------------------------------------------------------------------------------- citations

_PARTICLE = r"(?:\b(?:van|von|der|den|de|del|della|di|da|dos|du|la|le|ten|ter|al|el|bin|y)\s+)*"
_NAME1 = _PARTICLE + r"[A-ZÀ-ÖØ-ÞĀ-ſ][\w'’\-]*"
_AUTH = rf"{_NAME1}(?:(?:,\s*{_NAME1})*,?\s+(?:and|&)\s+{_NAME1})?(?:,?\s+et\s+al\.?)?"
_YEAR = r"(?:19|20)\d{2}[a-z]?"
_YEARS = rf"{_YEAR}(?:\s*,\s*(?:{_YEAR}|[a-z]\b))*"
NARRATIVE_RE = re.compile(rf"(?P<auth>{_AUTH})\s*\((?P<years>{_YEARS})\)")
PAREN_RE = re.compile(r"\(([^()]{4,400})\)")
CITE_PART_RE = re.compile(rf"^\s*(?P<pre>(?:e\.g\.|i\.e\.|see also|see|cf\.|also|for example|for instance|inter alia)"
                          rf",?\s+)?(?P<auth>(?:[A-Z][\w'’\-]+\s+){{0,2}}{_AUTH}),?\s+(?P<years>{_YEARS})(?P<post>,?\s*(?:p\.|pp\.|inter alia|"
                          rf"among others|for a review|for an overview|for details|chapter|ch\.|section|sec\.)[^;]*)?\s*$")
NUMERIC_RE = re.compile(r"\[(\s*\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*\s*)\]")
_STOP_NAMES = {"In", "The", "See", "And", "For", "Table", "Figure", "Section", "Fig", "Eq", "Equation", "Appendix",
               "Chapter", "This", "We", "Our", "Since", "From", "On", "At", "As", "To", "By", "With", "Of", "Both"}


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


class _Bib:
    """Bibliography entries, for checking and resolving citations, and BIBREF numbering by first use."""

    def __init__(self, entries):
        self.entries = entries
        self.norm = [_strip_accents(e) for e in entries]
        nums = [re.match(r"^\[(\d+)\]|^(\d+)\.\s", e) for e in entries]
        self.numbered = len(entries) >= 3 and sum(bool(m) for m in nums) >= 0.6 * len(entries)
        self.by_num = {int(m.group(1) or m.group(2)): i for i, m in enumerate(nums) if m}
        self.order, self.display = {}, {}

    def find(self, surname, year):
        s, best = _strip_accents(surname), None       # the entry naming the surname earliest: first author
        for i, e in enumerate(self.norm):
            j, k = e.find(s), e.find(year[:4])
            if 0 <= j < k and (len(year) == 4 or year in e) and (best is None or j < best[0]):
                best = (j, i)
        return best[1] if best else None

    def placeholder(self, key, original, raw):
        if key not in self.order:
            self.order[key] = len(self.order)
        ph = f"BIBREF{self.order[key]}"
        self.display.setdefault(ph, (original, raw))
        return ph


def _surname(auth):
    for w in re.split(r"\s+", re.sub(r"\bet\s+al\.?", "", auth).strip()):
        if w and w[0].isupper() and w.strip(",") not in _STOP_NAMES:
            return w.strip(",")
    return None


def _years(years):
    """'2019a, b' -> ['2019a', '2019b']; '2018, 2019' -> both."""
    out, last = [], None
    for part in re.split(r"\s*,\s*", years):
        if re.fullmatch(_YEAR, part):
            out.append(part)
            last = part[:4]
        elif last and re.fullmatch(r"[a-z]", part):
            out.append(last + part)
    return out


def _cite_keys(auth, years, bib):
    """[(key, bib index or None)] for one author group, or None if it isn't a citation."""
    name = _surname(auth)
    if not name:
        return None
    keys = []
    for y in _years(years):
        i = bib.find(name, y)                       # for the display map only: unmatched citations still count
        keys.append(((_strip_accents(name), y), i))
    return keys or None


def _citations(text, bib, numeric):
    raw = lambda i: bib.entries[i] if i is not None else None

    def narrative(m):
        keys = _cite_keys(m.group("auth"), m.group("years"), bib)
        if not keys:
            return m.group(0)
        return ", ".join(bib.placeholder(k, m.group(0), raw(i)) for k, i in keys)

    def paren(m):
        out, any_cite, extra = [], False, False
        for part in m.group(1).split(";"):
            pm = CITE_PART_RE.match(part)
            keys = _cite_keys(pm.group("auth"), pm.group("years"), bib) if pm else None
            if not keys:
                out.append(part.strip())
                extra = True
                continue
            any_cite = True
            phs = ", ".join(bib.placeholder(k, part.strip(), raw(i)) for k, i in keys)
            pre, post = (pm.group("pre") or "").strip(), (pm.group("post") or "").strip()
            extra = extra or bool(pre or post)
            out.append(" ".join(x for x in (pre, phs) if x) + (post if post.startswith(",") else (" " + post if post else "")))
        if not any_cite:
            return m.group(0)
        return f"({'; '.join(out)})" if extra else ", ".join(out)

    def num(m):
        ns = []
        for part in re.split(r"\s*[,;]\s*", m.group(1).strip()):
            a = re.split(r"\s*[–-]\s*", part)
            if len(a) == 2 and a[0].isdigit() and a[1].isdigit() and 0 < int(a[1]) - int(a[0]) < 30:
                ns.extend(range(int(a[0]), int(a[1]) + 1))
            elif part.isdigit():
                ns.append(int(part))
            else:
                return m.group(0)
        if bib.numbered and not all(n in bib.by_num for n in ns):
            return m.group(0)
        out = []
        for n in ns:
            i = bib.by_num.get(n)
            out.append(bib.placeholder(n, f"[{n}]", raw(i)))
        return " " + ", ".join(out)

    text = NARRATIVE_RE.sub(narrative, text)
    text = PAREN_RE.sub(paren, text)
    if numeric:
        text = NUMERIC_RE.sub(num, text)
    text = re.sub(r"(BIBREF\d+)\s*[;,]\s*(?=BIBREF\d+)", r"\1, ", text)
    text = re.sub(r"(BIBREF\d+)\s+([.,;:])", r"\1\2", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ----------------------------------------------------------------------------------------- cross-refs

_MORE = r"((?:\s*(?:,|and|&|–|-|to)\s*\d+\b)*)"
XREF_RES = [
    ("TABREF", re.compile(rf"\b(Tables?|Tab\.|Tabs\.)\s+(\d+)\b{_MORE}")),
    ("FIGREF", re.compile(rf"\b(Figures?|Figs?\.)\s+(\d+)\b{_MORE}")),
    ("SECREF", re.compile(r"\b(Sections?|Secs?\.|§|Appendix|Appendices)\s*((?:\d+|[A-H])(?:\.\d+)*)\b"
                          r"((?:\s*(?:,|and|&|–|-|to)\s*(?:\d+|[A-H])(?:\.\d+)*\b)*)")),
    ("EQREF", re.compile(r"\b(Equations?|Eqs?\.|Eqns?\.)\s*\(?(\d+)\)?((?:\s*(?:,|and|&|–|-|to)\s*\(?\d+\)?)*)")),
]


class _XRefs:
    """'Table 2' -> 'Table TABREFn' etc.; n from one counter shared by all kinds, in order of first mention."""

    def __init__(self):
        self.order, self.display = {}, {}

    def __call__(self, text):
        for kind, rx in XREF_RES:
            def sub(m, kind=kind):
                labels = [m.group(2)] + re.findall(r"(?:\d+|[A-H])(?:\.\d+)*", m.group(3) or "")
                phs = []
                for lab in labels:
                    if (kind, lab) not in self.order:
                        self.order[(kind, lab)] = len(self.order)
                    ph = f"{kind}{self.order[(kind, lab)]}"
                    self.display.setdefault(ph, (lab, None))
                    phs.append(ph)
                if kind == "EQREF" and "(" in m.group(0):
                    return f"{m.group(1)} ( " + " , ".join(phs) + " )"
                seps = re.findall(r"\s*(?:,|and|&|–|-|to)\s*", m.group(3) or "")
                return f"{m.group(1)} {phs[0]}" + "".join(s + p for s, p in zip(seps, phs[1:]))
            text = rx.sub(sub, text)
        return text


# ------------------------------------------------------------------------------------------ the parser

def _stats(lines, cols_of, body):
    """Paragraph indent and line pitch from body-size lines. Indent = the most common small positive
    offset of a FULL line from the column edge; pitch = the most common top-to-top step."""
    offs, steps, prev = Counter(), Counter(), None
    for l in lines:
        if abs(l.size - body) > 0.6 or l.cells > 1 or l.col == -1:
            prev = None
            continue
        left, right = cols_of(l)
        d = l.x0 - left
        if 0.25 * body <= d <= 3 * body and l.x1 >= right - 2:
            offs[round(d)] += 1
        if prev is not None and prev.chunk == l.chunk:
            s = l.top - prev.top
            if 0.9 * body <= s <= 2.2 * body:
                steps[round(s * 2) / 2] += 1
        prev = l
    full = sum(1 for l in lines if l.col != -1 and abs(l.size - body) <= 0.6)
    top = offs.most_common(1)
    indent = top[0][0] if top and top[0][1] >= max(3, 0.01 * full) else None
    pitch = steps.most_common(1)[0][0] if steps else 1.2 * body
    return indent, pitch


def _heading_level(l, body, cols_of, numbered_styles, in_appendix):
    """(level, name) if `l` looks like a section heading, else None. Headings are short, start the column
    or are centred in it, and stand out: larger, bold, or set in (small) caps with a number."""
    text = l.text.strip()
    words = text.split()
    m = HEAD_NUM_RE.match(text)
    # the gap after a section number ("2   Dogmatism data") can be wide enough to count as a cell
    if not words or len(words) > 16 or l.cells > (2 if m else 1) or CAPTION_RE.match(text) or l.fn_start \
            or l.lead_bold or text.endswith("-") or l.math >= 0.5:
        return None
    if not (text[0].isupper() or text[0].isdigit()) or text.endswith((",", ";")):
        return None
    caps = l.caps >= 0.9 and sum(c.isalpha() for c in text) >= 3
    smallcaps = caps and l.maxsize >= 1.1 * l.size
    size = l.maxsize if smallcaps else l.size   # small caps: the true capitals carry the heading size
    big, bold = size >= body + 0.8, l.bold >= 0.85
    num = m.group("num") if m else None
    if num and len(num.rstrip(".")) == 1 and num[0].isalpha() and not num.endswith(".") and not in_appendix:
        num = None                                  # "A Study of ..." is a title, not appendix A
    if num and num.rstrip(".").isdigit() and int(num.rstrip(".")) > 30:
        return None                                 # a stray number line, not a section
    if not (big or bold or smallcaps or (caps and num) or (num and l.italic >= 0.85 and (
            re.search(r"\d\.\d", num) or (num.endswith(".") and num[0].isalpha())))):
        return None
    if not big and size < body - 1.3:
        return None
    left, right = cols_of(l)
    centred = abs((l.x0 + l.x1) / 2 - (left + right) / 2) <= 0.08 * (right - left)
    if not (abs(l.x0 - left) <= 3 or centred):
        return None
    name = text[m.end():] if num else text
    if not num and text.endswith(".") and len(words) > 5:
        return None
    if num and not (name.strip()[:1].isupper() or name.strip()[:1].isdigit()):
        return None
    name = name.strip(" .:")
    style = (round(size * 2) / 2, bold, smallcaps)
    if num:
        n = num.rstrip(".")
        if re.fullmatch(r"[IVX]+", n):
            level = 1
        elif re.fullmatch(r"[A-Z]", n):
            level = 1 if in_appendix else 2
        else:
            level = n.count(".") + 1
        numbered_styles[style] = level
    else:
        level = numbered_styles.get(style, 1 if big or smallcaps else 2)
    return level, name


def parse_pdf(path, paper_id=None, grobid_url=None, debug=False):
    """PDF -> (Paper, display map). `grobid_url` is accepted and ignored, so this is a drop-in for
    pdf_parse.parse_pdf. debug=True returns the classified lines instead."""
    import pdfplumber                               # local: only this backend needs the dependency

    with pdfplumber.open(path) as pdf:
        pages = [(p.width, p.height, _page_words(p)) for p in pdf.pages]
    chars = [c for _, _, ws in pages for w in ws for c in w["chars"]]
    if not chars:
        raise RuntimeError(f"no extractable text in {path} (scanned image? needs OCR)")
    body = Counter(round(c["size"] * 2) / 2 for c in chars).most_common(1)[0][0]
    body_font = Counter(_font(c) for c in chars if abs(c["size"] - body) <= 0.5).most_common(1)[0][0]

    # ---- rows -> column geometry -> lines. Geometry comes from the pages of the same parity (odd and even
    # margins can differ); a page whose lines clearly fit a different NUMBER of columns keeps its own
    # (a two-column abstract page in a one-column thesis).
    page_rows = [_rows(ws, body) for _, _, ws in pages]
    width = pages[0][0]
    segs = [[(c[0]["x0"], c[-1]["x1"], statistics.median(w["size"] for w in c)) for row in rows for c in _cells(row, body)]
            for rows in page_rows]
    geom_all = _geometry([x for v in segs for x in v], body, width)
    parity = []
    for k in (0, 1):
        v = [x for seg in segs[k::2] for x in seg]
        g = _geometry(v, body, width) if len(v) >= 150 else geom_all
        same = len(g) != len(geom_all) or all(abs(a[0] - b[0]) <= 3 for a, b in zip(g, geom_all))
        parity.append(geom_all if same else g)
    def fit(g, v):                                  # body lines that run exactly edge to edge of a column of g
        return sum(abs(size - body) <= 0.6 and any(abs(x0 - a) <= 2 and abs(x1 - b) <= 2 for a, b in g)
                   for x0, x1, size in v)
    geom = []
    for i, v in enumerate(segs):
        base, own = parity[i % 2], _geometry(v, body, width)
        geom.append(own if len(own) != len(base) and fit(own, v) >= max(10, 3 * fit(base, v)) else base)

    lines = []
    for i, rows in enumerate(page_rows):
        page_lines = [_make_line(i, col, ws, n, body_font) for col, ws, n in _assign(rows, geom[i], body)]
        lines.extend(_reading_order([l for l in page_lines if l.text.strip()]))

    def cols_of(l):
        cols = geom[l.page]
        if l.col == -1 or len(cols) == 1:
            return (cols[0][0], cols[-1][1])
        return cols[l.col]

    # ---- furniture: outside the text block, a bare page number, or the same short line on many pages
    body_lines = [l for l in lines if abs(l.size - body) <= 0.6 and l.cells == 1 and len(l.text) > 20]
    tops, bots = sorted(l.top for l in body_lines), sorted(l.bottom for l in body_lines)
    lo, hi = tops[int(0.01 * len(tops))] - 0.3 * body, bots[int(0.99 * (len(bots) - 1))] + 0.3 * body
    key = lambda l: re.sub(r"\d+", "#", l.text.lower()).strip()
    repeated = Counter(k for _, k in {(l.page, key(l)) for l in lines if len(l.text.split()) <= 12})
    for l in lines:
        outside = (l.bottom < lo or l.top > hi) and not (l.page == 0 and l.size > body + 1)   # not a title up top
        if outside or re.fullmatch(r"\d{1,3}", l.text.strip()) \
                or (len(pages) >= 4 and repeated[key(l)] >= max(3, 0.4 * len(pages))):
            l.kind = "furniture"
    lines = [l for l in lines if l.kind != "furniture"]

    indent, pitch = _stats(lines, cols_of, body)
    vocab = _vocab(lines)
    acronyms = {t for l in lines if l.caps < 0.5 for t in re.findall(r"\b[A-Z][A-Z0-9-]*[A-Z0-9]\b", l.text)}

    # ---- text-like lines: body size, one cell, inside the column, and starting where text starts: the
    # column edge, the paragraph indent, a list marker, or an aligned block (a quotation, an indented list).
    # Figure labels and table cells start anywhere else.
    def basic(l):
        left, right = cols_of(l)
        gap_ok = abs(l.x0 - left) <= 2 + (indent or 0) and (l.x1 >= right - 2 or len(l.text.split()) >= 5)
        edge_to_edge = abs(l.x0 - left) <= 2 + (indent or 0) and l.x1 >= right - 2
        return (body - 1.3 <= l.maxsize and l.size <= body + 0.6             # maxsize: a line mostly in small caps
                and l.x0 >= left - 0.5 * body and l.x1 <= right + 2 and not l.fn_start    # a hanging "1." sits left
                and (l.cells == 1 or (gap_ok and l.cells == 2)                 # a gap around inline math
                     or (edge_to_edge and l.cells == 3)                        # a loosely justified full line
                     or (l.cells == 2 and LIST_RE.match(l.text) is not None)))  # "•   item"

    for k, l in enumerate(lines):
        if not basic(l):
            continue
        left, right = cols_of(l)
        edge = abs(l.x0 - left) <= 2 or (indent is not None and abs(l.x0 - left - indent) <= 2)
        full = l.x1 >= right - 2
        if l.math >= 0.5 and not (edge and full):
            continue
        listed = LIST_RE.match(l.text) is not None and l.x0 - left <= 4 * body
        aligned = False
        for nb in (lines[k - 1] if k else None, lines[k + 1] if k + 1 < len(lines) else None):
            if nb is not None and nb.chunk == l.chunk and basic(nb) and abs(nb.x0 - l.x0) <= 1.5 \
                    and abs(nb.top - l.top) < 2.5 * body and l.x0 - left <= 4 * body \
                    and max(nb.x1 - nb.x0, l.x1 - l.x0) >= 0.5 * (right - left):
                aligned = True
        above = lines[k - 1] if k else None        # a hanging indent: "• item ... / continuation"
        hang = above is not None and above.textlike == 1 and above.chunk == l.chunk \
            and 0 < l.top - above.top <= 1.45 * pitch and l.x0 - left <= 4 * body
        l.textlike = 1 if edge or listed or aligned else 2 if hang else 0     # 2: only via the line above, no chain

    # ---- front matter: title, abstract; on the abstract's page everything above it is dropped
    first = [l for l in lines if l.page == 0] or lines[:1]
    abs_i = next((i for i, l in enumerate(lines) if l.page <= 1 and ABSTRACT_RE.match(l.text)
                  and (len(l.text.split()) <= 2 or l.bold >= 0.3 or re.match(r"(?i)abstract\s*[—–:.-]", l.text))), None)
    title_pool = [l for l in (lines[:abs_i] if abs_i else lines) if l.page == 0] or first
    title = ""
    if title_pool:
        big = max(l.maxsize for l in title_pool)
        tl = [l for l in sorted(title_pool, key=lambda l: l.top) if abs(l.maxsize - big) < 0.3]
        group = [tl[0]]
        for l in tl[1:]:
            if l.top - group[-1].bottom <= 1.2 * big:
                group.append(l)
        title = _recase(_join([l.text for l in group], vocab), acronyms) if all(g.caps > 0.9 for g in group) \
            else _join([l.text for l in group], vocab)

    abstract = ""
    if abs_i is not None:
        a = lines[abs_i]
        parts = [ABSTRACT_RE.sub("", a.text, count=1)]
        j = abs_i + 1
        while j < len(lines):
            l = lines[j]
            if FRONT_RE.match(l.text) or _heading_level(l, body, cols_of, {}, False) or CAPTION_RE.match(l.text) \
                    or l.page > a.page + 1:
                break
            if not (l.page == a.page and l.top < a.top - 1) and not l.fn_start:
                parts.append(l.text)
            j += 1
        abstract = _tidy(_join(parts, vocab)).replace("’", "'")
        dropped = {id(l) for l in lines if l.page == a.page and l.top < a.top - 1}
        lines = [l for k, l in enumerate(lines) if k >= j and id(l) not in dropped]
    else:                                           # no "Abstract" label: drop page 1 down to the first heading
        k = next((i for i, l in enumerate(lines) if _heading_level(l, body, cols_of, {}, False)), 0)
        lines = lines[k:]

    # ---- body: walk the lines in reading order
    paragraphs, captions, bib_lines = [], [], []
    stack, section, mode = [], "", "body"
    numbered_styles, last_num = {}, 0
    cur, cur_lines, cur_disp, prev = [], [], 0, None   # paragraph texts, its lines, DISPLAYFORM count, last text line
    fn_chunk = None                                     # chunk whose remaining lines are footnotes

    def flush():
        nonlocal cur, cur_lines, cur_disp
        if cur:
            paragraphs.append((section, cur))
        cur, cur_lines, cur_disp = [], [], 0

    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        text = l.text.strip()

        # footnotes: from a line starting with a raised marker to the bottom of its column
        if l.fn_start and l.col != -1:
            fn_chunk = l.chunk
        elif fn_chunk == l.chunk and l.size >= body - 0.5:
            fn_chunk = None                         # body-size text again: the note (under a table, say) is over
        if fn_chunk == l.chunk and l.chunk is not None and not CAPTION_RE.match(text):
            l.kind = "footnote"
            i += 1
            continue

        # captions: the label line and the lines that continue it at the same size and spacing
        if CAPTION_RE.match(text) and not (prev is not None and cur and prev.chunk == l.chunk
                                          and not TERMINAL_RE.search(prev.text) and l.top - prev.top <= 1.3 * pitch
                                          and abs(l.size - body) <= 0.3):
            group = [l]
            j = i + 1
            while j < n:
                m = lines[j]
                if m.chunk != l.chunk or abs(m.size - l.size) > 0.35 or CAPTION_RE.match(m.text) \
                        or m.top - group[-1].top > 1.6 * l.size + 0.5 or m.x0 < l.x0 - 3 \
                        or (len(group) > 1 and m.x1 > max(g.x1 for g in group) + 3):
                    break
                group.append(m)
                j += 1
            for g in group:
                g.kind = "caption"
            captions.append(_tidy(_join([g.text for g in group], vocab)))
            i = j
            continue

        h = _heading_level(l, body, cols_of, numbered_styles, mode in ("refs", "appendix"))
        m = re.match(r"^(\d+)\.?\s", text)
        if h and m:                                 # "2. Size of ..." in body-size bold after section 4 is a list item
            if l.size < body + 0.8 and int(m.group(1)) <= last_num:
                h = None
            else:
                last_num = int(m.group(1))
        if h:
            group = [l]                             # a heading may wrap: same style, close below
            j = i + 1
            while j < n:
                m = lines[j]
                if m.chunk != l.chunk or abs(m.size - l.size) > 0.3 or abs(m.bold - l.bold) > 0.3 \
                        or l.x1 - l.x0 < 0.6 * (cols_of(l)[1] - cols_of(l)[0]) \
                        or abs(m.caps - l.caps) > 0.3 or m.top - group[-1].bottom > 0.6 * l.size \
                        or HEAD_NUM_RE.match(m.text) or len(" ".join(g.text for g in group + [m]).split()) > 20:
                    break
                group.append(m)
                j += 1
            level, name = h
            if len(group) > 1:
                name = _join([name] + [g.text for g in group[1:]], vocab).strip(" .:")
            name = _recase(name, acronyms)
            for g in group:
                g.kind = "heading"
            flush()
            if REFS_RE.match(name):
                mode = "refs"
            elif FRONT_RE.match(name):
                mode = "skip"
            else:
                if mode == "refs":
                    level = 1 if not re.match(r"^[A-Z]\.\d", text) else level
                mode = "appendix" if mode in ("refs", "appendix") or re.match(
                    r"(?i)^(appendix|appendices|supplementary|supplemental)", name) else "body"
                stack = stack[:level - 1] + [name]
                section = " ::: ".join(stack)
            prev = None
            i = j
            continue

        if mode != "refs" and re.match(r"^\[1\]\s", text) and any(
                re.match(r"^\[2\]\s", m.text) for m in lines[i + 1:i + 8]):
            flush()
            mode = "refs"
        if mode == "skip" or (FRONT_RE.match(text) and not cur):
            mode = "skip"
            l.kind = "front"
            i += 1
            continue
        if mode == "refs":
            l.kind = "bib"
            bib_lines.append(l)
            i += 1
            continue

        if l.textlike:
            l.kind = "text"
            left, right = cols_of(l)
            new = not cur
            if not new:
                same = prev is not None and prev.chunk == l.chunk
                pl, pr = cols_of(prev) if prev is not None else (left, right)
                ind = indent is not None and abs(l.x0 - left - indent) <= 1.5
                if ind and same and abs(prev.x0 - pl - indent) <= 1.5 and prev.x1 >= pr - 3:
                    ind = False                     # a block indented by exactly one indent, not a new paragraph
                term = prev is not None and TERMINAL_RE.search(prev.text) is not None
                prev_short = prev is not None and term and (
                    prev.x1 < max(x.x1 for x in cur_lines[-3:] if x.chunk == prev.chunk) - 1.0 * body
                    or (prev.x1 < pr - 1.5 * body and abs(prev.x0 - pl) <= 1.5))
                block_prev = prev is not None and prev.x0 - pl > (indent or 0) + 2.5
                block_now = l.x0 - left > (indent or 0) + 2.5
                item = LIST_RE.match(cur_lines[0].text) is not None if cur_lines else False
                if same and l.top - prev.top > 1.45 * pitch and not (prev.x1 >= pr - 2 and not term):
                    new = True                      # a gap, unless after a full unfinished line (tall glyphs)
                elif LIST_RE.match(text) and (term or text[0] in "•●▪◦‣∙·"):
                    new = True                      # "1. ..." only after a finished line; a bullet always
                elif item and l.x0 > cur_lines[0].x0 + 2 and not prev_short:
                    new = False                     # hanging indent: the item continues
                elif ind or prev_short:
                    new = True
                elif l.lead_bold and abs(l.x0 - left) <= 1.5 and prev.x1 < pr - 1.5 * body:
                    new = True                      # "Model-A is ..." items: a bold label after a short line
                elif same and block_prev != block_now:
                    new = True
            runin = l.lead_bold if re.search(r"[.:]$", l.lead_bold) and len(l.lead_bold.split()) <= 8 \
                and l.x0 - left <= 1.5 + (indent or 0) else ""
            if new or runin:
                flush()
            if runin and stack and i and lines[i - 1].kind == "heading":    # the last line of a wrapped bold heading
                stack[-1] += " " + runin
                section = " ::: ".join(stack)
                l.tex = " ".join(l.tex.split()[len(runin.split()):])
            elif runin.endswith("."):               # "Datasets. The statistics ..." is also section "... ::: Datasets."
                section = " ::: ".join(stack + [runin])     # the words stay in the text: nothing quotable is lost
            cur.append(l.tex)
            cur_lines.append(l)
            prev = l
            i += 1
            continue

        # not text: a run of non-text lines is a float body (next to a caption), a displayed equation
        # (math, right after text in the same column), or noise
        group = [l]
        j = i + 1
        while j < n and lines[j].chunk == l.chunk and not lines[j].textlike and not lines[j].fn_start \
                and not CAPTION_RE.match(lines[j].text) \
                and not _heading_level(lines[j], body, cols_of, dict(numbered_styles), mode == "appendix"):
            group.append(lines[j])
            j += 1
        nxt = lines[j] if j < n else None
        near_caption = (nxt is not None and CAPTION_RE.match(nxt.text) and nxt.chunk == l.chunk) or \
                       (i > 0 and lines[i - 1].kind == "caption" and lines[i - 1].chunk == l.chunk)
        mathy = any(EQNUM_RE.search(g.text) and g.x1 >= cols_of(g)[1] - 3 for g in group) or \
            statistics.mean(g.math for g in group) >= 0.2 or any(re.search(r"[=∑∏∫≤≥∈→←⊂∀∃≈∝]", g.text) for g in group)
        is_eq = (not near_caption and mathy and len(group) <= 8 and prev is not None and cur
                 and prev.chunk == l.chunk and max(g.size for g in group) >= body - 1.3)
        for g in group:
            g.kind = "equation" if is_eq else "float"
        if is_eq:
            cur.append(f"DISPLAYFORM{cur_disp}")
            cur_disp += 1
            flush()                                 # the "where ..." after an equation starts a new paragraph
        i = j

    flush()
    if debug:
        return lines

    # ---- bibliography -> entries: numbering, a hanging indent, or a gap starts an entry
    entries, prevb = [], None
    label = re.compile(r"^\[[^\]]{1,40}\]")              # "[12] ..." or "[Cheng et al.2015] ..."
    numbered_bib = sum(bool(label.match(b.text)) for b in bib_lines) >= 3
    for l in bib_lines:
        left, right = cols_of(l)
        if numbered_bib:
            new = prevb is None or label.match(l.text) is not None
        else:
            pleft, pright = cols_of(prevb) if prevb is not None else (left, right)
            new = prevb is None or (abs(l.x0 - left) <= 1.5 and prevb.x0 - pleft > 1.5) \
                or (l.chunk == prevb.chunk and l.top - prevb.bottom > 0.6 * l.size) \
                or (abs(l.x0 - left) <= 1.5 and abs(prevb.x0 - pleft) <= 1.5 and prevb.x1 < pright - 2 * body)
        if new:
            entries.append([l.text])
        else:
            entries[-1].append(l.text)
        prevb = l
    bib = _Bib([_tidy(_join(e, vocab)) for e in entries])

    # ---- placeholders, in reading order
    numeric = bib.numbered or len(NUMERIC_RE.findall(" ".join(" ".join(ls) for _, ls in paragraphs))) >= 5
    xref = _XRefs()
    out_p, out_s = [], []
    if abstract:
        out_p.append(abstract)
        out_s.append("Abstract")
    for sec, ls in paragraphs:
        text = _tidy(_join(ls, vocab)).replace("’", "'").replace("‘", "'")
        if len(text.split()) < MIN_WORDS:
            continue
        out_p.append(xref(_citations(text, bib, numeric)))
        out_s.append(sec)
    for c in captions:                              # no MIN_WORDS: QASPER keeps "Table 1: Data."
        out_p.append(c)
        out_s.append(CAPTION_SECTION)

    paper = Paper(paper_id or path.rsplit("/", 1)[-1].removesuffix(".pdf"), title, abstract, out_p, out_s)
    return paper, {**bib.display, **xref.display}


# ------------------------------------------------------------------------------------------- display

_PLACEHOLDER_RE = re.compile(r"\b(?:TABREF|FIGREF|SECREF|EQREF|DISPLAYFORM)\d+\b")
_BIBREF_RUN_RE = re.compile(r"\bBIBREF\d+\b(?:,\s*BIBREF\d+\b)*")


def restore(text, display_map):
    """Placeholders back to the paper's text, for display only. A run of author-year BIBREFs becomes one
    bracketed group, "(Firat et al., 2016; Johnson et al., 2016)"; a narrative citation keeps its form,
    "Kim et al. (2019)"; numeric ones become "[3, 5]". Same map format as pdf_parse.restore."""
    def bib(m):
        phs = re.findall(r"BIBREF\d+", m.group(0))
        if not all(ph in display_map for ph in phs):
            return m.group(0)
        cites = [display_map[ph][0] for ph in phs]
        if all(re.fullmatch(r"\[\d+\]", c) for c in cites):
            return "[" + ", ".join(c[1:-1] for c in cites) + "]"
        if len(cites) == 1 and "(" in cites[0]:
            return cites[0]
        return "(" + "; ".join(cites) + ")"

    def other(m):
        ph = m.group(0)
        if ph.startswith("DISPLAYFORM"):
            return "[equation]"
        return display_map[ph][0] if ph in display_map else ph

    return _PLACEHOLDER_RE.sub(other, _BIBREF_RUN_RE.sub(bib, text))


def main():
    ap = argparse.ArgumentParser(description="parse a PDF to a QASPER-style Paper with pdfplumber (no GROBID)")
    ap.add_argument("pdf")
    ap.add_argument("--paper_id", default=None)
    ap.add_argument("--show", type=int, default=5, help="paragraphs to print")
    ap.add_argument("--debug", action="store_true", help="print every line with the kind it was given")
    args = ap.parse_args()
    if args.debug:
        for l in parse_pdf(args.pdf, args.paper_id, debug=True):
            print(f"{str(l.kind):9s} p{l.page} c{l.col:<2d} x{l.x0:5.0f}-{l.x1:<5.0f} y{l.top:5.0f} s{l.size:4.1f} "
                  f"b{l.bold:.1f} m{l.math:.1f} c{l.caps:.1f} n{l.cells} t{int(l.textlike)} | {l.text[:100]}")
        return
    paper, display = parse_pdf(args.pdf, args.paper_id)
    print(f"title     : {paper.title}")
    print(f"abstract  : {paper.abstract[:200]}{'...' if len(paper.abstract) > 200 else ''}")
    print(f"paragraphs: {len(paper.paragraphs)}   sections: {len(set(paper.sections))}   "
          f"captions: {paper.sections.count(CAPTION_SECTION)}   "
          f"citations: {sum(ph.startswith('BIBREF') for ph in display)}")
    for i, (s, p) in enumerate(zip(paper.sections, paper.paragraphs)):
        if i >= args.show:
            break
        print(f"\n  [P{i + 1}] ({s}) {p[:300]}{'...' if len(p) > 300 else ''}")


if __name__ == "__main__":
    main()
