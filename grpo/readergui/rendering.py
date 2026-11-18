"""What the interface shows, as HTML: the top bar, the stage messages with their icons, answers and their sources.

Every piece is one line of HTML built from classes that styling.py gives their look. Paper text is escaped here.
"""
import base64
import html
import itertools
import os
import re
import unicodedata

from qasper.reward import ABSTAIN

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icons")     # grpo/icons
ICONS = {f[:-4]: open(os.path.join(ICON_DIR, f), "rb").read() for f in os.listdir(ICON_DIR) if f.endswith(".svg")}
BOOK = "01-flip-through"                            # also the app's own icon, shown still

# stage -> (message, icon file name without .svg; "-dark" is added for dark mode)
PARSING = ("Parsing the PDF…", BOOK)
DOWNLOADING = ("Downloading the paper…", BOOK)
INDEXING = ("Reading and indexing the paper…", BOOK)
ASSESSING = ("Retrieved the most relevant passages. Assessing whether they answer your question…", "02-03-alternate")
WRITING = ("Reasoning out the answer…", "04-teach-it")

TAGLINE = "Answers by quoting the paper, or says it couldn't find the answer."
NO_PAPER = "Open a PDF first."
CONFIDENCE = {"high": "High confidence", "medium": "Medium confidence · check it against the source",
              "low": "Low confidence · a lead, not an answer",
              "defined": "Defined in text", "seen": "Seen in text"}

_icon_uses = itertools.count()


def icon(name, size="", still=False):
    """The icon as two embedded images, light and dark: the styles show the one for the theme. Animated unless
    `still`; the counter makes each use a new image, so its loop starts from the beginning. `size`: a style
    class (lpr-top, lpr-start, lpr-inline) for anything but a stage icon."""
    tag = f"<!--{next(_icon_uses)}-->".encode()
    out = ""
    for cls, suffix in (("lpr-light", ""), ("lpr-dark", "-dark")):
        svg = ICONS[name + suffix]
        if still:
            svg = re.sub(rb"animation:[^;}]*;?", b"", svg)
        classes = " ".join(c for c in ("lpr-icon", cls, size, "" if still else "lpr-anim") if c)
        out += f'<img class="{classes}" alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg + tag).decode()}">'
    return out


# ---------------------------------------------------------------------------------- text from the paper
# pdf_parse_v3 writes maths the way QASPER has it ($...$, \alpha , x_1, x^{ab}: its _texchar / _texword / TEX).
# tidy() turns that back into symbols for display; the model and the quote checks keep the original text.
SYMBOLS = {"in": "∈", "le": "≤", "ge": "≥", "times": "×", "rightarrow": "→", "leftarrow": "←", "sum": "∑",
           "infty": "∞", "cdot": "·", "ne": "≠", "approx": "≈", "sim": "∼", "prime": "′"}
MARK = ("\x02", "\x03")                             # around a quote inside its paragraph: become <mark>
MATH_RE = re.compile(r"\$([^$]{1,300}?)\$")
MONEY_RE = re.compile(r"\d[\d,.]*[kKmMbB]?\s+[A-Za-z]{2,}")     # "$5 and $10" is two prices, not maths


def _symbol(m):
    name = m.group(1)
    if name in SYMBOLS:
        return SYMBOLS[name]
    case = "SMALL" if name.islower() else "CAPITAL"
    try:
        return unicodedata.lookup(f"GREEK {case} LETTER {name.upper().replace('LAMBDA', 'LAMDA')}")
    except KeyError:
        return ""                                   # a command with no symbol here: dropped


def _maths(m):
    s = m.group(1)
    if MONEY_RE.match(s):
        return html.escape(m.group(0))
    s = s.replace("^{\\prime }", "′")
    s = re.sub(r"\\([A-Za-z]+) ?(?=[_^,.;:)\]}]|$)", _symbol, s)      # the space the parser adds after a command
    s = html.escape(re.sub(r"\\([A-Za-z]+)", _symbol, s))
    for mark, tag in (("_", "sub"), ("^", "sup")):
        s = re.sub(re.escape(mark) + r"\{([^{}]*)\}", rf"<{tag}>\1</{tag}>", s)
        s = re.sub(re.escape(mark) + r"(&[#\w]+;|[^\x02\x03{\s])", rf"<{tag}>\1</{tag}>", s)
    return s.replace("{", "").replace("}", "")                      # grouping braces left over


def tidy(text):
    """Paper text for display: escaped, with the parser's LaTeX-style maths as symbols and real sub/superscripts,
    and a marked quote in <mark>."""
    out, last = "", 0
    for m in MATH_RE.finditer(text):
        out += html.escape(text[last:m.start()]) + _maths(m)
        last = m.end()
    out += html.escape(text[last:])
    return re.sub(r"\s+", " ", out).replace(MARK[0], "<mark>").replace(MARK[1], "</mark>")


# ---------------------------------------------------------------------------------- top bar and start screen
def header(title=None, summary="", stage=None):
    """The top bar beside Open PDF: a stage message with its icon, or the open paper's title, its one-line
    summary and the book in the middle, or (nothing open) the invitation to open one."""
    if stage:
        text = f'<div class="lpr-head-stage">{html.escape(stage[0])}{icon(stage[1], "lpr-inline")}</div>'
        book = ""
    elif title:
        text = f'<div class="lpr-title">{html.escape(title)}</div><div class="lpr-sub">{html.escape(summary)}</div>'
        book = icon(BOOK, "lpr-top", still=True)
    else:
        text = f'<div class="lpr-sub">{html.escape(summary or "Open a PDF to start.")}</div>'
        book = ""
    return (f'<div class="lpr-head"><div class="lpr-head-text">{text}</div>'
            f'<div class="lpr-top-slot">{book}</div><div></div></div>')


def start(book=True):
    """The empty conversation: the large book (before a paper is open) and the tagline."""
    return f'<div class="lpr-start-box">{icon(BOOK, "lpr-start", still=True) if book else ""}<div>{html.escape(TAGLINE)}</div></div>'


# ---------------------------------------------------------------------------------- stages
def stage(which, detail=""):
    """A stage's message, an optional smaller second line, then its icon on a line of its own."""
    message, icon_name = which
    detail = f'<div class="lpr-detail">{html.escape(detail)}</div>' if detail else ""
    return f'<div class="lpr-stage">{html.escape(message)}{detail}{icon(icon_name)}</div>'


def looking_in(sections, n_passages):
    """"Looking in: A · B · C (16 passages)": at most six section names, then "and N more"."""
    if not sections:
        return f"Looking in {n_passages} passages"
    more = f" and {len(sections) - 6} more" if len(sections) > 6 else ""
    return f"Looking in: {' · '.join(sections[:6])}{more} ({n_passages} passages)"


# ---------------------------------------------------------------------------------- answers
def oval(text):
    """A reply that is not an answer: the abstention, or a request to open a paper."""
    return f'<div class="lpr-oval">{html.escape(text)}</div>'


def _sources(quotes, refs):
    """All of an answer's quotes behind one "View source": each quoted paragraph once, in its own bubble, with
    every quote from it marked and the references those quotes cite. `refs`: the references of each quote."""
    paragraphs = {}                                 # paragraph text -> [section, marked spans, references]
    for q, cited in zip(quotes, refs):
        text = q["before"] + q["quote"] + q["after"]
        entry = paragraphs.setdefault(text, [(q["section"] or "").split(":::")[-1].strip() or "Unnamed section", [], []])
        entry[1].append((len(q["before"]), len(q["before"]) + len(q["quote"])))
        entry[2] += cited
    if not paragraphs:
        return ""
    names = list(dict.fromkeys(e[0] for e in paragraphs.values()))
    out = ""
    for text, (section, spans, cited) in paragraphs.items():
        marked, last = "", 0
        for start, end in sorted(spans):            # overlapping quotes become one mark
            if start >= last:
                marked += text[last:start] + MARK[0] + text[start:end] + MARK[1]
            elif end > last:
                marked = marked[:-1] + text[last:end] + MARK[1]
            last = max(last, end)
        label = f'<div class="lpr-para-sec">{html.escape(section)}</div>' if len(paragraphs) > 1 else ""
        cited = "".join(f"<div>{html.escape(r['reference'] or r['cited_as'])}</div>" for r in cited)
        cited = f'<div class="lpr-refs">{cited}</div>' if cited else ""
        out += f'<div class="lpr-para">{label}{tidy(marked + text[last:])}{cited}</div>'
    return (f'<details class="lpr-src"><summary><span class="lpr-view">View source</span>'
            f'<span>{html.escape(" · ".join(names))}</span></summary>{out}</details>')


def _answer(a, part=None):
    """One answer: its confidence line (led by the part's name in an overview), the answer, its sources."""
    left = {r["placeholder"]: r for r in a["references"]}
    per_quote = [[left.pop(ph) for ph in q.get("references", []) if ph in left] for q in a["quotes"]]
    if per_quote:
        per_quote[0] += list(left.values())         # cited by the answer alone: with the first quote
    label = CONFIDENCE.get(a.get("confidence"), "")
    if part:
        label = f'<span class="lpr-part">{html.escape(part)}</span>' + (f" · {label}" if label else "")
    line = (f'<div class="lpr-conf"><span class="lpr-dot lpr-{a["confidence"]}"></span><span>{label}</span></div>'
            if label else "")
    answer = (f'<div class="lpr-hex"><span class="lpr-cap"></span><div class="lpr-hex-text">{tidy(a["answer"])}</div>'
              f'<span class="lpr-cap lpr-cap-r"></span></div>')
    return f'<div class="lpr-answer">{line}{answer}{_sources(a["quotes"], per_quote)}</div>'


WORD_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
COMMON = set("a an the of in on for to and or with by is are was were be this that it its as at from than not no".split())


def _repeats(answer, earlier):
    """Whether an overview part says what an earlier part already said: the same numbers, or mostly the same
    words. A display rule only; overview.py keeps every part it found."""
    words = set(WORD_RE.findall(answer.lower())) - COMMON
    numbers = {w for w in words if w[0].isdigit()}
    for other in earlier:
        other_numbers = {w for w in other if w[0].isdigit()}
        if (numbers and numbers == other_numbers) or len(words & other) >= 0.6 * len(words | other):
            return True
    earlier.append(words)
    return False


def render(result):
    """The reply for a reader result: an answer, an overview (its found parts as answers, the missing ones
    named in one line), or the abstention."""
    if result["status"] == "answer":
        out = _answer(result)
    elif result["status"] == "overview" and result["found"]:
        said = []
        parts = "".join(_answer(a, a["label"]) for a in result["aspects"]
                        if a["status"] == "answer" and not _repeats(a["answer"], said))
        missing = ", ".join(m[:1].lower() + m[1:] for m in result["missing"])
        missing = f'<div class="lpr-missing">Not found with enough confidence: {html.escape(missing)}.</div>' if missing else ""
        out = f"<div>{parts}{missing}</div>"
    else:
        out = oval(ABSTAIN)
    return re.sub(r"\s+", " ", out)                 # one line: the chat reads a blank line as the end of the HTML
