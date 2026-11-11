"""PDF -> Paper via GROBID. Replaces in-text references, table/figure refs and displayed formulas with
QASPER-style placeholders (BIBREFn, TABREFn, FIGREFn, DISPLAYFORMn) and keeps a display map back to the
original text for each placeholder, so the quote check always runs on placeholder text."""
import argparse
import mimetypes
import re
import sys
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET

from qasper.data import Paper

TEI = "{http://www.tei-c.org/ns/1.0}"
MIN_WORDS = 5


def _multipart_body(fields, files):
    """Encode `fields` (name -> str) and `files` (name -> (filename, bytes, content_type)) as multipart/form-data."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for name, (filename, data, content_type) in files.items():
        header = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                  f"Content-Type: {content_type}\r\n\r\n")
        parts.append(header.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def grobid_tei(pdf_path, url="http://localhost:8070", timeout=300):
    try:
        with urllib.request.urlopen(f"{url}/api/isalive", timeout=10) as resp:
            alive = resp.read().decode().strip().lower()
    except urllib.error.URLError as e:
        raise RuntimeError(f"GROBID not reachable at {url}: {e}") from e
    if alive != "true":
        raise RuntimeError(f"GROBID at {url} is not alive (isalive returned {alive!r})")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    content_type = mimetypes.guess_type(pdf_path)[0] or "application/pdf"
    body, boundary = _multipart_body({"includeRawCitations": "1"},
                                     {"input": (pdf_path.rsplit("/", 1)[-1], pdf_bytes, content_type)})
    req = urllib.request.Request(f"{url}/api/processFulltextDocument", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GROBID returned {e.code} for {pdf_path}: {e.read()[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GROBID request failed for {pdf_path}: {e}") from e


def _text(el):
    """All text under `el`, including tails of children, with no XML tags."""
    return "".join(el.itertext())


_BIBREF_RUN_SEP = re.compile(r"(BIBREF\d+)[;,.\s]*(?=BIBREF\d+)")
_BIBREF_BRACKET = re.compile(r"[(\[]\s*((?:BIBREF\d+(?:,\s*)?)+)\s*[)\]]")


def _clean(s):
    s = _BIBREF_RUN_SEP.sub(r"\1, ", s)                      # adjacent refs: "BIBREF4; BIBREF5" -> "BIBREF4, BIBREF5"
    s = _BIBREF_BRACKET.sub(r"\1", s)                         # "(BIBREF4, BIBREF5)" -> "BIBREF4, BIBREF5"
    s = re.sub(r"^\s*[•●▪◦]\s*", "", s)                     # list bullets: QASPER drops them
    s = re.sub(r"\(\s*,?\s*\)", "", s)
    s = re.sub(r"\[\s*,?\s*\]", "", s)
    s = re.sub(r";(?=BIBREF)", "; ", s)                      # unresolved citation run into a ref: "2016;BIBREF15"
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _bib_index(target, id_to_n):
    if not target:
        return None
    return id_to_n.get(target.lstrip("#"))


def _append(parts, text):
    """Append `text`, inserting a space if it would otherwise run into the last non-empty part with no
    boundary (e.g. a resolved ref placeholder immediately followed by an unresolved ref's raw text, with
    an empty tail between them)."""
    prev = next((p for p in reversed(parts) if p), "")
    if prev and text and prev[-1].isalnum() and text[0].isalnum():
        parts.append(" ")
    parts.append(text)


def _formula(f, formula_n, display):
    ph = f"DISPLAYFORM{formula_n[0]}"
    formula_n[0] += 1
    display[ph] = (" ".join(_text(f).split()), None)
    return ph


def _substitute_refs(el, id_to_bibn, fig_n, tab_n, display, formula_n):
    """Render `el`'s content to a string, replacing <ref> and <formula> with placeholders and dropping footnote
    markers. `display` maps placeholder -> (original text, raw bib reference or None)."""
    parts = [el.text or ""]
    for child in el:
        tag = child.tag.split("}")[-1]
        if tag == "ref":
            rtype = child.get("type")
            target = child.get("target")
            original = _text(child)
            if rtype == "bibr":
                n = _bib_index(target, id_to_bibn)
                if n is not None:
                    ph = f"BIBREF{n}"
                    display.setdefault(ph, (original, id_to_bibn.raw.get(target.lstrip("#") if target else "")))
                    _append(parts, ph)
                else:
                    _append(parts, original)
            elif rtype == "table" and target in tab_n:
                ph = f"TABREF{tab_n[target]}"
                display.setdefault(ph, (original, None))
                _append(parts, ph)
            elif rtype == "figure" and target in fig_n:
                ph = f"FIGREF{fig_n[target]}"
                display.setdefault(ph, (original, None))
                _append(parts, ph)
            elif rtype == "foot":
                pass
            else:
                _append(parts, original)
        elif tag == "formula":
            _append(parts, _formula(child, formula_n, display))
        else:
            _append(parts, _text(child))
        parts.append(child.tail or "")
    return "".join(parts)


class _BibIndex(dict):
    """id -> BIBREF number, plus the raw reference strings keyed by id."""
    raw = {}


def _parse_header(root):
    header = root.find(f"{TEI}teiHeader")
    title_el = header.find(f".//{TEI}titleStmt/{TEI}title")
    title = _text(title_el).strip() if title_el is not None else ""
    return title


def _bib_entries(root):
    """xml:id -> (n, raw reference string) in document order of the bibliography."""
    out = {}
    for n, bibl in enumerate(root.findall(f".//{TEI}back//{TEI}listBibl/{TEI}biblStruct")):
        xid = bibl.get("{http://www.w3.org/XML/1998/namespace}id")
        raw_el = bibl.find(f"{TEI}note[@type='raw_reference']")
        if raw_el is not None and raw_el.text:
            raw = raw_el.text.strip()
        else:                                                   # no raw string: rebuild "authors. year. title"
            authors = [" ".join(t.strip() for t in pn.itertext() if t.strip())
                       for pn in bibl.findall(f".//{TEI}author/{TEI}persName")]
            date = bibl.find(f".//{TEI}date[@when]")
            title_el = bibl.find(f".//{TEI}title")
            bits = [", ".join(authors), date.get("when")[:4] if date is not None else "",
                    _text(title_el).strip() if title_el is not None else ""]
            raw = ". ".join(b for b in bits if b)
        if xid:
            out[xid] = (n, raw)
    return out


def _section_parent(number):
    """'3.1' -> '3'; '3' -> None (no parent); non-numeric -> None."""
    if not number or "." not in number:
        return None
    return number.rsplit(".", 1)[0]


def _section_name(head_el, numbers_seen):
    """Nested section name, parent joined with ' ::: ', derived from the head's n= numbering; the bare head
    text if there is no numbering. "" if there is no head at all."""
    if head_el is None:
        return ""
    text = (_text(head_el) or "").strip()
    number = head_el.get("n") if head_el is not None else None
    if not number:
        return text
    parent_num = _section_parent(number)
    if parent_num and parent_num in numbers_seen:
        return f"{numbers_seen[parent_num]} ::: {text}"
    return text


def tei_to_paper(tei_xml, paper_id):
    root = ET.fromstring(tei_xml)
    title = _parse_header(root)

    bib = _bib_entries(root)
    id_to_bibn = _BibIndex({xid: n for xid, (n, _) in bib.items()})
    id_to_bibn.raw = {xid: raw for xid, (_, raw) in bib.items()}

    body = root.find(f"{TEI}text/{TEI}body")
    back = root.find(f"{TEI}text/{TEI}back")

    figures = (body.findall(f".//{TEI}figure") if body is not None else []) + \
              (back.findall(f".//{TEI}figure") if back is not None else [])
    fig_n, tab_n, captions = {}, {}, []
    fi = ti = 0
    for fig in figures:
        xid = fig.get("{http://www.w3.org/XML/1998/namespace}id") or ""
        desc = fig.find(f"{TEI}figDesc")
        caption_text = " ".join(_text(desc).split()) if desc is not None else ""
        head = fig.find(f"{TEI}head")
        label = re.sub(r"\s*:?\s*$", "", " ".join(_text(head).split())) if head is not None else ""
        if caption_text and label and not caption_text.startswith(label):
            caption_text = f"{label}: {caption_text}"              # QASPER: "Table 1: Data Statistics."
        if fig.get("type") == "table":
            tab_n[f"#{xid}"] = ti
            ti += 1
        else:
            fig_n[f"#{xid}"] = fi
            fi += 1
        if caption_text:
            captions.append(caption_text)

    display = {}
    formula_n = [0]

    def render_div_paragraphs(div, numbers_seen, out_paras, out_secs):
        head_el = div.find(f"{TEI}head")
        section_name = _section_name(head_el, numbers_seen)
        number = head_el.get("n") if head_el is not None else None
        if number:
            numbers_seen[number] = section_name
        blocks = []
        for child in div:
            tag = child.tag.split("}")[-1]
            if tag == "p":
                blocks.append(_substitute_refs(child, id_to_bibn, fig_n, tab_n, display, formula_n))
            elif tag == "formula":                             # displayed equation: QASPER appends it to the
                ph = _formula(child, formula_n, display)      # paragraph that introduces it
                if blocks:
                    blocks[-1] += f" {ph}"
                else:
                    blocks.append(ph)
        for text in map(_clean, blocks):
            if len(text.split()) >= MIN_WORDS:
                out_paras.append(text)
                out_secs.append(section_name)
        for sub in div.findall(f"{TEI}div"):
            render_div_paragraphs(sub, numbers_seen, out_paras, out_secs)

    paragraphs, sections = [], []

    abstract_ps = root.findall(f"{TEI}teiHeader//{TEI}abstract//{TEI}p")
    abstract_parts = []
    for p in abstract_ps:
        text = _clean(_substitute_refs(p, id_to_bibn, fig_n, tab_n, display, formula_n))
        if text:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)
    if abstract and len(abstract.split()) >= MIN_WORDS:
        paragraphs.append(abstract)
        sections.append("Abstract")

    numbers_seen = {}
    if body is not None:
        for div in body.findall(f"{TEI}div"):
            render_div_paragraphs(div, numbers_seen, paragraphs, sections)

    if back is not None:
        for div in back.findall(f"{TEI}div"):
            if div.get("type") == "references":
                continue
            render_div_paragraphs(div, numbers_seen, paragraphs, sections)

    for caption in captions:                                    # no length filter: QASPER keeps short captions
        paragraphs.append(caption)
        sections.append("Figures and tables")

    paper = Paper(paper_id or "pdf", title, abstract, paragraphs, sections)
    return paper, display


def parse_pdf(path, paper_id=None, grobid_url="http://localhost:8070"):
    tei_xml = grobid_tei(path, grobid_url)
    return tei_to_paper(tei_xml, paper_id or path.rsplit("/", 1)[-1].removesuffix(".pdf"))


_PLACEHOLDER_RE = re.compile(r"\b(?:TABREF|FIGREF|DISPLAYFORM)\d+\b")
_BIBREF_RUN_RE = re.compile(r"\bBIBREF\d+\b(?:,\s*BIBREF\d+\b)*")


def _citation(s):
    """GROBID's text for one citation of a group, e.g. "(Firat et al. 2016;": unpaired brackets and separators off."""
    s = s.strip().rstrip(";,").strip()
    if s and s[0] in "([" and s.count(s[0]) > s.count({"(": ")", "[": "]"}[s[0]]):
        s = s[1:].strip()
    if s and s[-1] in ")]" and s.count(s[-1]) > s.count({")": "(", "]": "["}[s[-1]]):
        s = s[:-1].strip()
    return s.rstrip(";,").strip()


def restore(text, display_map):
    """Placeholders back to the paper's text, for display only. A run of BIBREFs becomes one bracketed group,
    "(Firat et al. 2016; Johnson et al. 2016)"; a single narrative citation keeps its form, "Kim et al. (2019)"."""
    def bib(m):
        phs = re.findall(r"BIBREF\d+", m.group(0))
        if not all(ph in display_map for ph in phs):
            return m.group(0)
        cites = [_citation(display_map[ph][0]) for ph in phs]
        if len(cites) == 1 and "(" in cites[0]:
            return cites[0]
        return "(" + "; ".join(cites) + ")"

    def other(m):
        return display_map[m.group(0)][0] if m.group(0) in display_map else m.group(0)

    return _PLACEHOLDER_RE.sub(other, _BIBREF_RUN_RE.sub(bib, text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("--paper_id", default=None)
    ap.add_argument("--url", default="http://localhost:8070")
    args = ap.parse_args()

    paper, display = parse_pdf(args.pdf_path, args.paper_id, args.url)
    n_refs = sum(1 for ph in display if ph.startswith("BIBREF"))
    n_captions = sum(1 for s in paper.sections if s == "Figures and tables")
    print(f"paragraphs: {len(paper.paragraphs)}")
    print(f"sections: {len(set(paper.sections))}")
    print(f"captions: {n_captions}")
    print(f"references (BIBREF placeholders): {n_refs}")
    print()
    for i, (p, s) in enumerate(zip(paper.paragraphs[:3], paper.sections[:3])):
        print(f"[{i}] ({s}) {p[:300]}")


if __name__ == "__main__":
    sys.exit(main())
