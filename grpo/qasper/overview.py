"""Answers the trained judge alone would withhold. General questions ("What are the results?") are detected by rule
and answered as fixed, specific sub-questions. Definition questions use the paper's own wording ("Defined in
text"), a lower judge cutoff, or a concept overview. Origin questions get the paper's own citation for the term
("Seen in text"), never the model's guess. Every answer still passes the same quote rules, and
Reader.ask is untouched. On 50 dev papers, general questions got an answer for 82-98% of papers, against 28-98%
asked directly.
"""
import re
from collections import defaultdict

from qasper.context import build_context
from qasper.prompt import format_prompt
from qasper.reader import BIBREF_RE, repair_quotes
from qasper.reward import ABSTAIN, format_ok, norm_ws, parse, quote_check

# ---- detection: a question is general when, after removing question words and words that refer to the paper,
# everything left belongs to one type's vocabulary. Flags 0.1-0.3% of QASPER dev/test questions, all of them
# genuinely general ("What are the contributions of this paper?").
PHRASES = [r"\b(of|in|from|by|for)? ?(this|the|their|your) (paper|study|work|article|report|research|document|manuscript|thesis"
           r"|project|dissertation|preprint|publication)\b",
           r"\bthe authors?\b", r"\bin (short|brief|summary|a nutshell|general)\b", r"\bcan you\b", r"\bcould you\b",
           r"\btell me\b", r"\bgive me\b", r"\bat a high level\b", r"\bhigh level\b"]
STOP = set("""what which how are is was were be been being the a an of in on for to and or this that these those
it its they their them we our us you your me i my please do does did briefly brief short quick quickly overall main
key major most important primary principal there here some any all with from by as at into its paper study article
document research report authors author give tell list explain describe summarise summarize summary overview outline
tldr tl dr gist whats s were has have had so where project dissertation preprint publication""".split())
NEUTRAL = {"summarise", "summarize", "summary", "overview", "outline", "tldr", "tl", "dr", "gist", "describe", "explain"}
VOCAB = {
    "results": set("""results result findings finding find found outcomes outcome conclude concluded conclusion
        conclusions show shows showed shown achieve achieved achievements discover discovered takeaways takeaway learn
        learned""".split()),
    "methodology": set("""method methods methodology methodologies approach approaches design designed setup set up
        experimental experiments experiment procedure procedures implementation implemented conduct conducted carried
        out done performed work works perform use used uses using take taken""".split()),
    "limitations": set("""limitation limitations weakness weaknesses shortcoming shortcomings drawback drawbacks caveat
        caveats future directions open remaining remain remains problems issues fail fails failure failures suggest
        suggested propose proposed method approach""".split()),
    "about": set("""about purpose objective objectives aim aims goal goals contribution contributions contribute
        propose proposed topic focus idea point motivation""".split()),
}
TIE_ORDER = ["about", "results", "methodology", "limitations"]      # who wins a tie on shared words


def detect(question):
    """The overview type of a general question ("results", "methodology", "limitations", "about"), else None."""
    q = question.lower().replace("’", "'")
    q = re.sub(r"\bfuture work\b", "future", q)                     # 'work' is a methodology word everywhere else
    for p in PHRASES:
        q = re.sub(p, " ", q)
    toks = re.findall(r"[a-z]+", q)
    if not toks or len(toks) > 12:
        return None
    rest = [t for t in toks if t not in STOP]
    if not rest:                                                     # only question words, paper words, summarise...
        if any(t in NEUTRAL for t in toks):
            return "about"
        return "methodology" if "how" in toks else None              # "How did they do it?"
    hits = {k: sum(t in v for t in rest) for k, v in VOCAB.items() if all(t in v for t in rest)}
    if not hits:
        return None
    return max(hits, key=lambda k: (hits[k], -TIE_ORDER.index(k)))


# ---- the fixed sub-questions per type, worded like the specific questions the judge was calibrated on.
# Dropped after the experiment: "What is the overall research design?" (passed the judge for 14-28% of papers) and
# "What weaknesses or failure cases are reported?" (mostly repeated the stated limitations).
TEMPLATES = {
    "results": {"heading": "Results", "passages": "section", "aspects": [
        ("Main finding", "What is the main finding of the paper?"),
        ("Quantitative results", "What quantitative results are reported?"),
        ("Comparison with other methods", "How does the proposed approach compare with the baselines or previous methods?")]},
    "methodology": {"heading": "Methodology", "passages": "similarity", "aspects": [
        ("Methods and models", "What methods or models are used?"),
        ("Experimental setup", "What is the experimental setup, including data, metrics and settings?")]},
    "limitations": {"heading": "Limitations", "passages": "section", "aspects": [
        ("Stated limitations", "What limitations do the authors state?"),
        ("Future work", "What future work do the authors propose?")]},
    "about": {"heading": "Overview", "passages": "section", "aspects": [
        ("Research objective", "What is the research objective?"),
        ("Approach", "What approach does the paper take?"),
        ("Main contribution", "What is the main contribution?"),
        ("Main findings", "What are the main findings?")]},
    "concept": {"heading": None, "passages": "section", "aspects": [     # fallback; {x} = the term, also the heading
        ("Definition", "How is {x} defined?"),
        ("How it works", "How does {x} work?"),
        ("Role", "What is the role of {x}?"),
        ("Origin", "Where was {x} originally proposed?")]},
}
SECTIONS = {                                                         # matched anywhere in a section name, lower case
    "results": ["abstract", "result", "experiment", "evaluation", "finding", "discussion", "conclusion", "analysis",
                "performance"],
    "limitations": ["limitation", "discussion", "conclusion", "future", "error", "analysis", "shortcoming"],
    "about": ["abstract", "introduction", "conclusion"],
    "concept": ["abstract", "introduction", "method", "approach", "background", "model", "architecture", "framework",
                "preliminar", "definition", "related", "overview", "proposed"],
}
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

# ---- language: overview passages skip prose in another language (e.g. a thesis's French summary)
EN_WORDS = set("the of and to in is that for with as on are this by be we it from an which or at not can our these "
               "their has have was were".split())
OTHER_WORDS = {
    "fr": set("le la les des du un une et est que qui dans pour sur avec par pas sont ces nous cette être au aux ont "
              "mais plus leur".split()),
    "de": set("der die das und ist nicht mit den dem ein eine zu von auf für sich des werden wird auch als bei oder".split()),
    "es": set("el los las del una por con para que es se en como más pero sus este esta son entre".split()),
    "it": set("il della che di per una sono con non nel alla delle gli questo questa anche come".split()),
}


def english(text):
    """False only for clear foreign prose: 20+ words, no maths, foreign common words >= 5 and > 2x English."""
    if "$" in text or "DISPLAYFORM" in text or "INLINEFORM" in text:
        return True
    words = re.findall(r"[a-zà-öø-ÿ]+", text.lower())
    if len(words) < 20:
        return True
    count = defaultdict(int)
    for w in words:
        count[w] += 1
    en = sum(count[w] for w in EN_WORDS)
    other = max(sum(count[w] for w in ws) for ws in OTHER_WORDS.values())
    return not (other >= 5 and other > 2 * en)


# ---- concept fallback: "What is X?" about a term in many paragraphs, tried only when the normal answer is withheld
_A = r"(?:an? |the )?"
CONCEPT_PATTERNS = [rf"^(?:what|who) (?:is|are|was|were) {_A}(?P<x>.+?)\??$", rf"^what'?s {_A}(?P<x>.+?)\??$",
                    rf"^(?:define|explain|describe) {_A}(?P<x>.+?)\.?$", rf"^what does {_A}(?P<x>.+?) mean\??$",
                    rf"^what is meant by {_A}(?P<x>.+?)\??$", rf"^how does {_A}(?P<x>.+?) work\??$",
                    rf"^what (?:is|are) the (?:definition|meaning)s? of {_A}(?P<x>.+?)\??$"]
CONCEPT_BAD = set("""of in for on with by to from their they them this these those that used using between and or do does
did is are was were be been what which who how why when where whose its it his her our we you your than as at into about
over under""".split())
GENERIC = set("""baseline baselines dataset datasets data model models method methods task tasks metric metrics result
results approach approaches corpus corpora feature features experiment experiments annotator annotators crowdworkers
system systems benchmark benchmarks setting settings language languages""".split())   # want a list, not a definition
NOT_A_TERM = set("""best existing previous other reported specifically specific main proposed different various
keyword keywords""".split())                                         # "the best reported system" asks for a list
CONCEPT_MIN_SHARE = 0.10                                            # X in at least this share of paragraphs
CONCEPT_FALLBACK = True                                              # False: never try the fallback


def concept_term(question, patterns=CONCEPT_PATTERNS, digits=False):
    """The term X of a definition-style question (or one matching `patterns`), else None."""
    q = " ".join(question.strip().split())
    for p in patterns:
        m = re.match(p, q, re.I)
        if m:
            x = m.group("x").strip(" ?.")
            words = re.split(r"[\s\-]+", x.lower())
            if (1 <= len(words) <= 4 and not set(words) & CONCEPT_BAD and (digits or not re.search(r"\d", x))
                    and not any(w in GENERIC or w.rstrip("s") in GENERIC or w in NOT_A_TERM for w in words)):
                return x
            return None
    return None


def concept_for(reader, question):
    """The term when the concept fallback applies to this question on the indexed paper, else None."""
    if not CONCEPT_FALLBACK:
        return None
    x = concept_term(question)
    return x if x and centrality(reader.paper, x) >= CONCEPT_MIN_SHARE else None


def centrality(paper, x):
    """Share of paragraphs mentioning X (singular or plural)."""
    xl = x.lower()
    forms = {xl, xl[:-1] if xl.endswith("s") else xl + "s"}
    pat = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b")
    return sum(bool(pat.search(p.lower())) for p in paper.paragraphs) / max(len(paper.paragraphs), 1)


def fused(retriever, paper_id, question):
    """The hybrid retriever's fused score for every paragraph (HybridRetriever.top_k ranks by exactly this)."""
    n = len(retriever.papers[paper_id].paragraphs)
    f = defaultdict(float)
    for w, ranks in ((retriever.w, retriever.bm25[paper_id].top_k(question, n)),
                     (1 - retriever.w, retriever.dense.top_k(paper_id, question, n))):
        for pos, i in enumerate(ranks):
            f[i] += w / (retriever.c + pos + 1)
    return f


def _top(scores, k, pool=None):
    idx = list(scores) if pool is None else [i for i in pool if i in scores]
    return sorted(idx, key=lambda i: -scores[i])[:k]


def passage_sets(reader, kind, x=None):
    """The paragraph indices shown for each sub-question. "section": one shared set for all of them, from the
    type's sections first, ranked by the sub-questions' summed scores and filled from the rest of the paper.
    "similarity": each sub-question's own top k, as a normal question gets."""
    paper, k, spec = reader.paper, reader.k, TEMPLATES[kind]
    questions = [q.format(x=x) for _, q in spec["aspects"]]
    ok = {i for i, p in enumerate(paper.paragraphs) if english(p)}
    if spec["passages"] == "similarity":
        n = len(paper.paragraphs)
        return [[i for i in reader.retriever.top_k(paper.paper_id, q, n) if i in ok][:k] for q in questions]
    summed = defaultdict(float)
    for q in questions:
        for i, v in fused(reader.retriever, paper.paper_id, q).items():
            summed[i] += v
    # "about" always shows the abstract (at most two paragraphs)
    chosen = [i for i, s in enumerate(paper.sections) if s == "Abstract" and i in ok][:2] if kind == "about" else []
    pool = [i for i, s in enumerate(paper.sections)
            if i in ok and i not in chosen and any(w in (s or "").lower() for w in SECTIONS[kind])]
    chosen += _top(summed, k - len(chosen), pool)
    chosen += [i for i in _top(summed, len(summed)) if i in ok and i not in chosen][:k - len(chosen)]
    return [chosen] * len(questions)


CITE_YEAR_RE = re.compile(r"\bet al\.?,?\s*(?:19|20)\d\d[a-z]?\b|\(\s*(?:19|20)\d\d[a-z]?\s*\)")


def numbers_ok(answer, quotes):
    """Every number in the answer appears in its quotes (True when the answer has none). Citation years are ignored."""
    answer = CITE_YEAR_RE.sub(" ", answer)
    text = " ".join(q for _, q in quotes).replace(",", "")
    return all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", text)
               for n in {m.replace(",", "") for m in NUM_RE.findall(answer)})


def answer_one(reader, question, context, p_to_idx, reuse=False, lp=None, low=None):
    """One sub-question through the judge, the generator and the reader's rules, plus the number check.
    KEEP IN STEP with Reader._ask: the same rule sequence, copied so the normal pipeline stays untouched.
    reuse: the GGUF judge keeps the passages read for the previous sub-question (same passages, new question).
    lp: a judge score already computed for this prompt. low: a lower cutoff, shown as "low"."""
    paper = reader.paper
    prompt = format_prompt(reader.tok, context, question)
    if lp is None:
        lp = reader.gguf.abstain_score_reuse(prompt) if reuse else reader.abstain_score(prompt)
    score = -lp
    confidence = reader.confidence(score) or ("low" if low is not None and score >= low else None)
    debug = {"question": question, "reason": None, "abstain_score": lp, "verifier_score": score, "outcome": None,
             "confidence": None, "raw_output": None, "repaired": None}

    def withheld(reason, outcome):
        debug["reason"], debug["outcome"] = reason, outcome
        return {"status": "abstain", "debug": debug}

    if confidence is None and reader.skip_below_low:
        return withheld("verifier", "not generated")
    raw = reader.generate(prompt)
    debug["raw_output"] = raw
    parsed = parse(raw)
    if parsed is None:
        return withheld("format", "format")
    quotes, answer = parsed
    if answer == ABSTAIN:
        return withheld("generator abstained", "abstain")
    if not format_ok(quotes, answer, False, p_to_idx):
        quotes, edits = repair_quotes(quotes, answer)
        if not edits or not format_ok(quotes, answer, False, p_to_idx):
            return withheld("format", "format")
        debug["repaired"] = edits
    check = quote_check(quotes, paper, p_to_idx)
    if check != "ok":
        return withheld(check, check)
    if not numbers_ok(answer, quotes):                               # overview only: a number its quotes don't show
        return withheld("number not in its quotes", "numbers")
    debug["outcome"] = "answer"
    debug["confidence"] = confidence
    if confidence is None:
        return withheld("verifier", "answer")
    shown = [reader._quote(p, q, paper.sections[p_to_idx[p]], norm_ws(paper.paragraphs[p_to_idx[p]]))
             for p, q in quotes]
    cited = dict.fromkeys(BIBREF_RE.findall(" ".join([answer] + [q for _, q in quotes])))
    refs = [{"placeholder": ph, "cited_as": reader.display[ph][0], "reference": reader.display[ph][1]}
            for ph in cited if ph in reader.display]
    return {"status": "answer", "confidence": confidence, "answer": reader._restore(answer), "raw_answer": answer,
            "quotes": shown, "references": refs, "debug": debug}


# ---- abbreviation fallback: "What is RAG?" withheld -> the paper's own spelled-out form, checked by rule
ABBREV_FALLBACK = True                                               # False: never try it
ABBREV_PATTERNS = CONCEPT_PATTERNS + [rf"^what does {_A}(?P<x>.+?) stand for\??$",
                                      rf"^what is {_A}(?P<x>.+?) short for\??$"]
SKIP_WORDS = {"of", "the", "and", "for", "in", "on", "to", "a", "an", "with", "by"}


def abbrev_term(question):
    """X when the question asks what an abbreviation X is (2+ capitals, one token), else None."""
    q = " ".join(question.strip().split())
    for p in ABBREV_PATTERNS:
        m = re.match(p, q, re.I)
        if m:
            x = m.group("x").strip(" ?.")
            ok = re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,11}", x) and sum(c.isupper() for c in x) >= 2
            return x if ok else None
    return None


def _expansion_before(text, letters):
    """The words ending `text` whose initials spell `letters` ("AC Optimal Power Flow" for A,C,O,P,F), else None."""
    words = list(re.finditer(r"[A-Za-z][A-Za-z0-9']*", text))
    i, j, start = len(letters) - 1, len(words) - 1, None
    if not words or text[words[-1].end():].strip():
        return None
    while i >= 0 and j >= 0:
        w = words[j].group()
        caps = [c.lower() for c in w] if w.isupper() and len(w) >= 2 else None
        if caps and caps == letters[i - len(caps) + 1:i + 1]:
            i -= len(caps)
        elif w[0].lower() == letters[i]:
            i -= 1
        elif w.lower() not in SKIP_WORDS:
            return None
        start = words[j].start()
        j -= 1
    return text[start:].strip() if i < 0 else None


def abbreviation(reader, question):
    """A shown answer spelling out the abbreviation as the paper does, with its defining sentences as quotes;
    None if the question isn't one, the switch is off, or the paper never spells it out."""
    x = abbrev_term(question) if ABBREV_FALLBACK else None
    if not x:
        return None
    paper, letters = reader.paper, [c.lower() for c in x if c.isupper()]
    found = None
    for i, p in enumerate(paper.paragraphs):
        text = norm_ws(p)
        for m in re.finditer(r"\(\s*" + re.escape(x) + r"\s*[,;)]", text):
            sent_start = max(text.rfind(". ", 0, m.start()) + 2, 0) if text.rfind(". ", 0, m.start()) >= 0 else 0
            exp = _expansion_before(text[sent_start:m.start()], letters)
            if exp:
                found = (i, exp)
                break
        if found:
            break
    described = paper_description(paper, x)
    if found:
        i, exp = found
        quotes = [(i, next(s for s in _sentences(norm_ws(paper.paragraphs[i])) if exp in s))]
        if described and described[2] != quotes[0][1]:
            quotes.append((described[0], described[2]))
    elif described:                                                  # never spelled out: the paper's description
        exp, quotes = described[1], [(described[0], described[2])]
    else:
        return None
    return _defined(reader, question, exp, quotes, {"abbreviation": x, "reason": "spelled out in the paper"})


_sentences = lambda t: re.split(r"(?<=[.!?])\s+", t)
DESCRIBING_VERBS = set("""is are refers controls attenuates addresses involves denotes means describes consists measures
computes maps scales rescales normalises normalizes represents combines converts encodes assigns estimates captures
quantifies transforms defines determines compresses aims seeks extends generalises generalizes""".split())


def paper_description(paper, x):
    """(paragraph, clause, sentence) for the first sentence or "while / whereas / ;" clause, in reading order, where X
    (or "<word> X") is the subject of a describing verb ("Residual rebalancing controls ..."). Else None."""
    head = re.compile(rf"^(?:the )?(?:[\w\-]+ )?{re.escape(x)}(?:\s*\([^)]*\))?(?:\s+BIBREF\d+)?\s+(?P<v>[a-z]+)\b"
                      rf"(?!\s+(?:used|applied)\b)", re.I)          # "X is applied ..." says nothing about X
    for j, p in enumerate(paper.paragraphs):
        if paper.sections[j] == "Figures and tables" or not english(p):
            continue
        for s in _sentences(norm_ws(p)):
            for clause in re.split(r",? (?:while|whereas) |; ", s):
                m = head.match(clause)
                if m and m.group("v").lower() in DESCRIBING_VERBS:
                    return j, clause.rstrip(".") + ".", s
    return None


def description(reader, question):
    """A "defined in text" answer from the paper's own description of the term of a definition-style question."""
    x = concept_term(question) if ABBREV_FALLBACK else None
    d = paper_description(reader.paper, x) if x else None
    if not d:
        return None
    return _defined(reader, question, d[1], [(d[0], d[2])], {"term": x, "reason": "described in the paper"})


def _defined(reader, question, answer, quotes, debug, confidence="defined"):
    paper = reader.paper
    shown = [reader._quote(n, q, paper.sections[j], norm_ws(paper.paragraphs[j])) for n, (j, q) in enumerate(quotes, 1)]
    cited = dict.fromkeys(BIBREF_RE.findall(" ".join([answer] + [q for _, q in quotes])))
    refs = [{"placeholder": ph, "cited_as": reader.display[ph][0], "reference": reader.display[ph][1]}
            for ph in cited if ph in reader.display]
    return {"status": "answer", "confidence": confidence, "answer": reader._restore(answer), "quotes": shown,
            "references": refs, "debug": {"question": question, **debug}}


DEFINITION_LOW = 0.8      # definition questions are shown from this judge score (40-item definition set, 2026-11-17)


def definition_answer(reader, question, result):
    """For a "What is X?" answer the judge withheld with a score >= DEFINITION_LOW: the answer, shown as low."""
    d, x = result["debug"], concept_term(question)
    if (result["status"] == "answer" or d.get("reason") != "verifier" or not x or re.search(r"\d", x)
            or d.get("verifier_score", -99) < DEFINITION_LOW):
        return None
    ranked = reader.retriever.top_k(reader.paper.paper_id, question, reader.k)     # Reader._ask's passages
    context, p_to_idx = build_context(reader.paper, ranked, reader.k)
    r = answer_one(reader, question, context, p_to_idx, lp=d["abstain_score"], low=DEFINITION_LOW)
    return r if r["status"] == "answer" else None


# Origin questions get only the citation the paper attaches to X, since the model invents sources.
ORIGIN_PATTERNS = [rf"^where (?:does|did|do) {_A}(?P<x>.+?) (?:come from|originate(?: from)?)\??$",
                   rf"^what (?:is|was|are) the origins? of {_A}(?P<x>.+?)\??$",
                   rf"^who (?:first )?(?:proposed|introduced|invented|developed) {_A}(?P<x>.+?)\??$",
                   rf"^where (?:was|were|is) {_A}(?P<x>.+?) (?:first )?(?:proposed|introduced)\??$"]
_HEAD_NOUN = (r"(?:model|models|system|systems|framework|method|algorithm|architecture|dataset|corpus|metric|toolkit"
              r"|network|networks|work|tagger|parser|embeddings|score)")
_PHRASE_END = (r"(?=\s*(?:[,.;:()—–]|$|\s(?:and|which|that|for|to|as|in|on|with|model|models|framework|method|approach"
               r"|architecture|system|dataset|corpus|metric)\b))")
_CLAIM = set("""show showed shown use used find found report reported combine combined propose proposed demonstrate
demonstrated suggest suggested observe observed achieve achieved apply applied employ employed train trained introduce
introduced present presented argue argued note noted""".split())     # "BIBREF3 show ...": the citation starts a claim
_WE, _INTRO = r"\b(?:we|this (?:paper|work|thesis))", r"(?:propose|introduce|present)[sd]?"
origin_term = lambda question: concept_term(question, ORIGIN_PATTERNS, digits=True)


def cited_origin(paper, x):
    """(paragraph, "BIBREFn" or "this paper", sentence) where the paper gives X's source, in reading order, else None."""
    xl = x.lower()
    forms = {xl, xl + "s"} | ({xl[:-3] + e for e in ("e", "es", "ed")} if xl.endswith("ing") and len(xl) > 5 else set())
    X = r"(?<![\w\-])(?:" + "|".join(map(re.escape, sorted(forms, key=len, reverse=True))) + r")(?![\w\-])"
    ours = [rf"{_WE}\s+{_INTRO}\s+(?!to\b)(?:(?:a|an|the)\s+)?(?:(?:new|novel)\s+)?(?:[\w\-]+\s+and\s+)?{X}{_PHRASE_END}",
            rf"{_WE}\b[^.]{{0,15}}\b(?:propose|introduce|present|define)[sd]?\s+(?!to\b)(?:(?:a|an|the)\s+)?"
            rf"(?:(?!(?:to|for|of|on|in|with|by|from)\b)[\w\-]+\s+){{1,4}}\(\s*{X}\s*\)"]     # "we introduce a ... (GTD)"
    cited = [rf"(?<![\w\-])[\w\-]+ {X}(?:\s*\([^)]{{1,60}}\))?,?\s+(BIBREF\d+)",                 # "sigmoid gating [3]"
             rf"{X}(?:\s+{_HEAD_NOUN})?(?:\s*\([^)]{{1,60}}\))?,?\s+(BIBREF\d+)",                  # "the X model [3]"
             rf"(BIBREF\d+),?\s+(?:subsequently\s+|first\s+)?{_INTRO}\s+(?:(?:a|an|the)\s+)?(?:[\w\-]+\s+)?{X}",
             rf"{X},?\s+(?:first\s+|originally\s+)?(?:proposed|introduced|developed|presented)\s+(?:by|in)\b[^.]{{0,60}}?(BIBREF\d+)"]
    for j, p in enumerate(paper.paragraphs):
        if paper.sections[j] == "Figures and tables" or not english(p):
            continue
        for s in _sentences(norm_ws(p)):
            if any(re.search(o, s, re.I) for o in ours):
                return j, "this paper", s
            for n, pt in enumerate(cited):
                for m in re.finditer(pt, s, re.I):
                    w = re.match(r"(?:BIBREF\d+[,\s]*)*([a-z]+)", s[m.end():].lstrip(", "))
                    if n < 2 and w and (w.group(1) in _CLAIM or w.group(1).endswith("ed")):
                        continue
                    return j, m.group(1), s
    return None


def origin(reader, question):
    """A "seen in text" answer with the paper's own source for an origin question's term, withheld if it gives none."""
    x = origin_term(question)
    found = cited_origin(reader.paper, x)
    if not found:
        return {"status": "abstain", "debug": {"question": question, "term": x, "reason": "origin: no source in the paper"}}
    j, ph, s = found
    name = x[:1].upper() + x[1:]
    said = f"{name} is introduced in this paper." if ph == "this paper" else f"{name} comes from {ph}."
    return _defined(reader, question, said, [(j, s)], {"term": x, "reason": "origin rule"}, "seen")


def answer(reader, question):
    """The app's entry point: overview for a general question, the citation rule for an origin question, else
    Reader.ask, with the fallbacks if that is withheld."""
    kind = detect(question)
    if kind:
        return run(reader, question, kind)
    if origin_term(question):
        return origin(reader, question)
    result = reader.ask(question)
    if result["status"] == "answer" and result.get("confidence") != "low":
        return result
    defined = abbreviation(reader, question) or description(reader, question)   # beats a low or withheld answer
    if defined:
        return defined
    if result["status"] == "answer":
        return result
    defined = definition_answer(reader, question, result)
    if defined:
        return defined
    x = concept_for(reader, question)
    if not x:
        return result
    concept = run(reader, question, "concept", x)
    if concept["found"]:
        return concept
    return {**result, "debug": {**result["debug"], "concept_fallback": concept["debug"]}}


def run(reader, question, kind, x=None):
    """The overview for a general question of type `kind` (x: the concept term). A repeat on the same paper comes
    from the reader's per-paper answer memory (Reader.index clears it)."""
    key = ("overview", question)
    if key in reader._answers:
        hit = reader._answers[key]
        return {**hit, "debug": {**hit["debug"], "cached": True}}
    paper, spec = reader.paper, TEMPLATES[kind]
    reuse = reader.gguf is not None and spec["passages"] == "section"     # one passage set: read it once
    if reader.gguf is not None:
        reader.gguf.forget_prefix()
    aspects, missing, seen, sections = [], [], {}, []
    for (label, sub), ranked in zip(spec["aspects"], passage_sets(reader, kind, x)):
        sub = sub.format(x=x)
        context, p_to_idx = build_context(paper, ranked, reader.k)
        sections += [paper.sections[i] for i in p_to_idx.values()]
        r = answer_one(reader, sub, context, p_to_idx, reuse)
        if r["status"] != "answer":
            missing.append(label)
            aspects.append({"label": label, "status": "withheld", "debug": r["debug"]})
            continue
        same = norm_ws(r["raw_answer"]).lower()
        if same in seen:                                             # the same answer as an earlier aspect: once
            aspects.append({"label": label, "status": "same", "same_as": seen[same], "debug": r["debug"]})
            continue
        seen[same] = label
        aspects.append({"label": label, **r})
    found = sum(a["status"] in ("answer", "same") for a in aspects)
    names = [s.split(":::")[0].strip() or "Unnamed section" for s in sections]
    result = {"status": "overview", "kind": kind, "heading": spec["heading"] or x, "found": found,
              "total": len(spec["aspects"]), "aspects": aspects, "missing": missing,
              "sections": list(dict.fromkeys(names)), "n_passages": reader.k,
              "debug": {"overview": kind, "question": question,
                        "aspects": [{"label": a["label"], **a["debug"]} for a in aspects]}}
    reader._answers[key] = result
    return result
