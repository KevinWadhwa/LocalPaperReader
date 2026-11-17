"""Overview mode: a general question ("What are the results?") answered as a few fixed, specific sub-questions.

The trained judge and generator only know specific questions, so a general one is detected by rule, replaced by
its type's fixed sub-questions, and each sub-question goes through the same judge, generator and quote rules as a
normal question. The found aspects are shown together, with the missing ones named, so a partial overview never
reads as complete. Reader.ask is not used or changed: normal questions keep their pipeline exactly.

Measured on 50 dev papers, fp32 (Nov 2026): at least one aspect found for 94% (results), 96% (methodology),
82% (limitations) and 98% (about) of papers, against 62 / 32 / 28 / 98% for the general question asked directly.
Section-chosen passages won for results, limitations and about; per-question similarity won for methodology.
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
                    rf"^where does {_A}(?P<x>.+?) come from\??$",
                    rf"^what (?:is|are) the (?:origin|definition|meaning)s? of {_A}(?P<x>.+?)\??$"]
CONCEPT_BAD = set("""of in for on with by to from their they them this these those that used using between and or do does
did is are was were be been what which who how why when where whose its it his her our we you your than as at into about
over under""".split())
GENERIC = set("""baseline baselines dataset datasets data model models method methods task tasks metric metrics result
results approach approaches corpus corpora feature features experiment experiments annotator annotators crowdworkers
system systems benchmark benchmarks setting settings language languages""".split())   # want a list, not a definition
CONCEPT_MIN_SHARE = 0.10                                             # X in at least this share of paragraphs


def concept_term(question):
    """The term X of a definition-style question, else None."""
    q = " ".join(question.strip().split())
    for p in CONCEPT_PATTERNS:
        m = re.match(p, q, re.I)
        if m:
            x = m.group("x").strip(" ?.")
            words = x.lower().split()
            if 1 <= len(words) <= 4 and not set(words) & CONCEPT_BAD and x.lower() not in GENERIC:
                return x
            return None
    return None


def concept_for(reader, question):
    """The term when the concept fallback applies to this question on the indexed paper, else None."""
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


def numbers_ok(answer, quotes):
    """Every number in the answer appears in its quotes (True when the answer has none)."""
    text = " ".join(q for _, q in quotes).replace(",", "")
    return all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", text)
               for n in {m.replace(",", "") for m in NUM_RE.findall(answer)})


def answer_one(reader, question, context, p_to_idx, reuse=False):
    """One sub-question through the judge, the generator and the reader's rules, plus the number check.
    KEEP IN STEP with Reader._ask: the same rule sequence, copied so the normal pipeline stays untouched.
    reuse: the GGUF judge keeps the passages read for the previous sub-question (same passages, new question)."""
    paper = reader.paper
    prompt = format_prompt(reader.tok, context, question)
    lp = reader.gguf.abstain_score_reuse(prompt) if reuse else reader.abstain_score(prompt)
    score = -lp
    confidence = reader.confidence(score)
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


def answer(reader, question):
    """The app's entry point: overview for a general question, else Reader.ask, with the concept fallback if that
    is withheld."""
    kind = detect(question)
    if kind:
        return run(reader, question, kind)
    result = reader.ask(question)
    if result["status"] == "answer":
        return result
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
