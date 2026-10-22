import re
import string
from collections import Counter

from qasper.context import contains_answer

ABSTAIN = "I couldn't find the answer in the retrieved sections."
MAX_QUOTES, MIN_WORDS, MAX_WORDS = 3, 5, 50
FORMAT_PENALTY, FABRICATION_PENALTY, UNGROUNDED_PENALTY = -0.5, -1.0, -0.5
MISCITE_PENALTY = -0.5

SAFE = r"(?:(?!</?(?:quote|answer)\b).)*"
QUOTE = rf'<quote p="P?(\d+)">({SAFE})</quote>'
ANSWER = rf"<answer>({SAFE})</answer>"
FULL_RE = re.compile(rf"\s*(?:{QUOTE}\s*)*{ANSWER}\s*", re.DOTALL)
QUOTE_RE = re.compile(QUOTE, re.DOTALL)
ANSWER_RE = re.compile(ANSWER, re.DOTALL)


def norm_ws(s):
    return " ".join(s.split())


def parse(output):
    if not FULL_RE.fullmatch(output):
        return None
    quotes = [(int(p), norm_ws(q)) for p, q in QUOTE_RE.findall(output)]
    answer = norm_ws(ANSWER_RE.search(output).group(1))
    return quotes, answer


def format_ok(quotes, answer, abstained, p_to_idx):
    if abstained:
        return not quotes
    if not answer or not 1 <= len(quotes) <= MAX_QUOTES:
        return False
    texts = [q for _, q in quotes]
    if len(set(texts)) < len(texts):
        return False
    return all(p in p_to_idx and MIN_WORDS <= len(q.split()) <= MAX_WORDS for p, q in quotes)


def normalise_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalise_answer(pred).split(), normalise_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_score(pred, question):
    golds = [a.text for a in question.answers if a.kind not in ("unanswerable", "empty")]
    if question.kind == "yes_no":
        return float(any(normalise_answer(pred) == normalise_answer(g) for g in golds))
    return max(token_f1(pred, g) for g in golds)

def quote_check(quotes, paper, p_to_idx):
    shown = [norm_ws(paper.paragraphs[i]) for i in p_to_idx.values()]
    worst = "ok"
    for p, q in quotes:
        if q in norm_ws(paper.paragraphs[p_to_idx[p]]):
            continue
        if not any(q in s for s in shown):
            return "fabricated"
        worst = "miscited"
    return worst

def reward(output, question, paper, p_to_idx, target, gold_ps, a=0.1, bonus=0.4):
    parsed = parse(output)
    if parsed is None:
        return FORMAT_PENALTY, "format"
    quotes, answer = parsed
    abstained = answer == ABSTAIN
    if not format_ok(quotes, answer, abstained, p_to_idx):
        return FORMAT_PENALTY, "format"
    check = quote_check(quotes, paper, p_to_idx)
    if check == "fabricated":
        return FABRICATION_PENALTY, "fabricated"
    if check == "miscited":
        return MISCITE_PENALTY, "miscited"
    if not abstained and question.kind == "extractive" and \
            not contains_answer(" ".join(q for _, q in quotes), answer):
        return UNGROUNDED_PENALTY, "ungrounded"
    if target == "abstain":
        return (1.0, "abstain") if abstained else (0.0, "answer")
    if abstained:
        return a, "abstain"
    cited_gold = any(p in gold_ps for p, _ in quotes)
    return answer_score(answer, question) * (1 + bonus * cited_gold), "answer"


if __name__ == "__main__":
    import json
    import statistics
    from transformers import AutoTokenizer
    from qasper.data import Paper, Answer, Question, load_split, table_only
    from qasper.context import build_context, target_label

    # ---------- 1. hand-made cases ----------
    abstract = "We compare against a CRF baseline on the CoNLL-2003 English dataset."
    para = "We train all models on the CoNLL-2003 English dataset for named entity recognition."
    paper = Paper("p", "T", abstract, [abstract, para], ["Abstract", "Experiments"])
    q_ext = Question("q1", "p", "Which dataset?",
                     [Answer("extractive", "CoNLL-2003", [para], [1], False, True)], "answerable", "extractive")
    q_yn = Question("q2", "p", "Do they use CoNLL?",
                    [Answer("yes_no", "Yes", [para], [1], False, True)], "answerable", "yes_no")
    p_to_idx, gold_ps = {1: 0, 2: 1}, [2]

    good_q = '<quote p="2">We train all models on the CoNLL-2003 English dataset</quote>'
    abstain = f"<answer>{ABSTAIN}</answer>"
    cases = [
        ("correct, cites gold",        q_ext, "answer",  good_q + "<answer>CoNLL-2003</answer>", 1.4, "answer"),
        ("partly correct (F1 2/3)",    q_ext, "answer",  good_q + "<answer>the CoNLL-2003 dataset</answer>", 2 / 3 * 1.4, "answer"),
        ("correct, cites non-gold",    q_ext, "answer",  '<quote p="1">a CRF baseline on the CoNLL-2003 English dataset</quote><answer>CoNLL-2003</answer>', 1.0, "answer"),
        ("newline inside quote",       q_ext, "answer",  '<quote p="2">We train all\nmodels on the CoNLL-2003 English dataset</quote><answer>CoNLL-2003</answer>', 1.4, "answer"),
        ("fabricated quote",           q_ext, "answer",  '<quote p="2">We train all models on the CoNLL-2012 English dataset</quote><answer>CoNLL-2003</answer>', -1.0, "fabricated"),
        ("real quote cited to wrong p",q_ext, "answer",  '<quote p="1">We train all models on the CoNLL-2003 English dataset</quote><answer>CoNLL-2003</answer>', -1.0, "fabricated"),
        ("ungrounded extractive",      q_ext, "answer",  '<quote p="2">English dataset for named entity recognition.</quote><answer>CoNLL-2003</answer>', -0.5, "ungrounded"),
        ("answer with no quote",       q_ext, "answer",  "<answer>CoNLL-2003</answer>", -0.5, "format"),
        ("text outside tags",          q_ext, "answer",  "Sure! " + good_q + "<answer>CoNLL-2003</answer>", -0.5, "format"),
        ("answer before quote",        q_ext, "answer",  "<answer>CoNLL-2003</answer>" + good_q, -0.5, "format"),
        ("tag-crossing trick",         q_ext, "answer",  '<quote p="2">We train all models on</quote> junk <quote p="2">the CoNLL-2003 English dataset here</quote><answer>CoNLL-2003</answer>', -0.5, "format"),
        ("quote too short",            q_ext, "answer",  '<quote p="2">CoNLL-2003 English dataset</quote><answer>CoNLL-2003</answer>', -0.5, "format"),
        ("duplicate quotes",           q_ext, "answer",  good_q + good_q + "<answer>CoNLL-2003</answer>", -0.5, "format"),
        ("p out of range",             q_ext, "answer",  '<quote p="5">We train all models on the CoNLL-2003 English dataset</quote><answer>CoNLL-2003</answer>', -0.5, "format"),
        ("abstain on answerable",      q_ext, "answer",  abstain, 0.1, "abstain"),
        ("abstain with a quote",       q_ext, "answer",  good_q + abstain, -0.5, "format"),
        ("near-miss abstain string",   q_ext, "answer",  "<answer>I couldn't find the answer.</answer>", -0.5, "format"),
        ("abstain target, abstains",   q_ext, "abstain", abstain, 1.0, "abstain"),
        ("abstain target, answers",    q_ext, "abstain", good_q + "<answer>CoNLL-2003</answer>", 0.0, "answer"),
        ("yes/no exact",               q_yn,  "answer",  good_q + "<answer>Yes</answer>", 1.4, "answer"),
        ("yes/no not exact",           q_yn,  "answer",  good_q + "<answer>Yes, they do</answer>", 0.0, "answer"),
    ]
    n_fail = 0
    for name, q, target, out, exp_score, exp_outcome in cases:
        score, outcome = reward(out, q, paper, p_to_idx, target, gold_ps if target == "answer" else [])
        ok = abs(score - exp_score) < 1e-6 and outcome == exp_outcome
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:30s} got ({score:+.3f}, {outcome}) expected ({exp_score:+.3f}, {exp_outcome})")
    print(f"\nhand-made cases: {len(cases) - n_fail}/{len(cases)} passed\n")

    # ---------- 2. real data (train) ----------
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    papers, questions = load_split("data/qasper/qasper-train-v0.3.json")
    ranking = json.load(open("data/qasper/retrieval_train.json"))["ranking"]
    labels, reasons, outcomes, lengths, example = Counter(), Counter(), Counter(), [], None
    for q in questions:
        paper = papers[q.paper_id]
        context, p_to_idx = build_context(paper, ranking[q.question_id])
        target, gold_ps = target_label(q, paper, p_to_idx)
        labels[target] += 1
        if target is None:
            continue
        lengths.append(len(tok(context)["input_ids"]))
        if target == "abstain":
            reasons["unanswerable" if q.label == "unanswerable" else
                    "table-only" if table_only(q) else "retrieval miss"] += 1
            outcomes["abstain target: " + str(reward(abstain, q, paper, p_to_idx, target, gold_ps))] += 1
        else:
            example = example or (context, q, target, gold_ps)
            words = paper.paragraphs[p_to_idx[gold_ps[0]]].split()[:MAX_WORDS]
            if len(words) < MIN_WORDS:
                outcomes["answer target: gold paragraph under 5 words"] += 1
                continue
            gold_text = next(a.text for a in q.answers if a.kind not in ("unanswerable", "empty"))
            out = f'<quote p="{gold_ps[0]}">{" ".join(words)}</quote><answer>{gold_text}</answer>'
            outcomes["answer target: " + reward(out, q, paper, p_to_idx, target, gold_ps)[1]] += 1

    n_used = labels["answer"] + labels["abstain"]
    print("labels:", dict(labels))
    print(f"natural abstention rate: {labels['abstain'] / n_used:.1%} of {n_used}")
    print("abstain reasons:", dict(reasons))
    q10 = statistics.quantiles(lengths, n=10)
    print(f"context tokens: median {statistics.median(lengths):.0f}, 90th {q10[-1]:.0f}, max {max(lengths)}")
    print("real-data reward outcomes:")
    for k, v in sorted(outcomes.items()):
        print(f"  {v:5d}  {k}")
    context, q, target, gold_ps = example
    print(f"\n--- example: {q.question}  target={target} gold_ps={gold_ps}\n{context[:2000]}")