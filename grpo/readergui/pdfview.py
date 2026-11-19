"""See in PDF: what each answer's links carry, and where Gradio serves the papers and PDF.js."""
import json
import mimetypes
import os
import re
import urllib.parse
from pathlib import Path

from qasper.pdf_parse_v3 import restore
from qasper.reward import norm_ws

PDFJS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfjs")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*[,.;:]?")                    # an ordinary word: no digits, maths or brackets
NEXT = 3                                                             # relevant passages offered after the quotes
mimetypes.add_type("text/javascript", ".mjs")                        # Windows can map .mjs wrongly


def served(path):
    """Where Gradio serves a file (its folder must be a static path)."""
    return "/gradio_api/file=" + urllib.parse.quote(Path(path).as_posix(), safe="/:")


def url(folder, file_id):
    """The kept paper with this file id, found even after a rename. None if it's gone."""
    for f in os.scandir(folder):
        if f.name.endswith(".pdf") and f.is_file() and os.stat(f.path).st_ino == file_id:
            return served(f.path)
    return None


def snippets(text, n=7, most=3):
    """Stretches of 5 to `n` plain words to search for. Citations, numbers and maths break them."""
    runs, run = [], []
    for w in text.split():
        if WORD.fullmatch(w):
            run.append(w)
        else:
            runs.append(run)
            run = []
    runs.append(run)
    return [" ".join(r[i:i + n]) for r in runs for i in range(0, len(r) - 4, n)][:most]


def attach(result, paper, display, file_id):
    """Each quote's See in PDF link: its paragraphs first, then the next relevant passages."""
    if not file_id:
        return
    for a in result.get("aspects") or [result]:
        if a.get("status") != "answer" or not a.get("quotes"):
            continue
        spots, at = [], {}
        for q in a["quotes"]:
            if q["paragraph"] not in at:
                at[q["paragraph"]] = len(spots)
                spots.append({"label": f"Quote {len(spots) + 1}",
                              "find": [q["quote"]] + snippets(q["quote"]) + snippets(q["paragraph"])})
        for idx in (a.get("debug") or {}).get("ranked", []):
            text = restore(norm_ws(paper.paragraphs[idx]), display)
            if len(spots) >= len(at) + NEXT:
                break
            if text not in at:
                at[text] = len(spots)
                spots.append({"label": paper.sections[idx].split(":::")[-1].strip() or "Passage", "find": snippets(text)})
        for q in a["quotes"]:
            q["see"] = "#lpr-see:" + urllib.parse.quote(json.dumps({"id": file_id, "at": at[q["paragraph"]], "spots": spots}))
