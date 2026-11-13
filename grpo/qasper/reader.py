"""Answer one question about one paper end to end: retrieve, prompt, abstain score, generate, rules, verifier.

The two model calls are `abstain_score` and `generate`; a GGUF backend replaces only those two.
"""
import re
from types import SimpleNamespace

# torch / peft / qasper.evaluate / sentence_transformers are imported lazily (inside the HF-path code only), so the
# GGUF laptop build with the ONNX embedder (embedder_path) never needs torch and the PyInstaller bundle can exclude
# it. transformers (tokenizers) is needed by both backends.
from transformers import AutoModelForCausalLM, AutoTokenizer

from qasper.bm25 import PaperIndex, tokenize
from qasper.context import build_context
from qasper.dense import DenseRetriever
from qasper.hybrid import HybridRetriever
from qasper.pdf_parse_v3 import restore
from qasper.prompt import format_prompt
from qasper.reward import ABSTAIN, MAX_QUOTES, MAX_WORDS, MIN_WORDS, format_ok, norm_ws, parse, quote_check

EMBEDDER = "BAAI/bge-small-en-v1.5"         # DenseRetriever's default
MAX_NEW_TOKENS = 384
BIBREF_RE = re.compile(r"\bBIBREF\d+\b")
ABSTAIN_COMPLETION = f"<answer>{ABSTAIN}</answer><|im_end|>"


def repair_quotes(quotes, answer):
    """Bring quotes that break only the count/length rules back inside them, without changing the answer.

    On 4B dev, 50 of the 57 format failures are long or extra quotes (a quote over 50 words, more than 3 quotes)
    around an answer as good as the shown ones. Here: drop quotes under MIN_WORDS, cut a quote over MAX_WORDS to the
    MAX_WORDS-word window sharing the most words with the answer (earliest on ties), drop repeats, keep the first
    MAX_QUOTES. Every kept quote is a contiguous piece of one the model wrote, so format_ok and quote_check still
    decide afterwards. The answer and the verifier score are untouched, so the tiers apply unchanged.
    Returns (quotes, edits); edits is empty when nothing changed."""
    need = set(tokenize(answer))
    out, edits = [], []
    for n, (p, q) in enumerate(quotes, 1):
        words = q.split()
        if len(words) < MIN_WORDS:
            edits.append(f"dropped quote {n} ({len(words)} words)")
            continue
        if len(words) > MAX_WORDS:
            start = max(range(len(words) - MAX_WORDS + 1),
                        key=lambda s: (len(need & set(tokenize(" ".join(words[s:s + MAX_WORDS])))), -s))
            q = " ".join(words[start:start + MAX_WORDS])
            edits.append(f"cut quote {n} from {len(words)} to {MAX_WORDS} words")
        if q in (x for _, x in out):
            edits.append(f"dropped quote {n} (repeat)")
            continue
        out.append((p, q))
    if len(out) > MAX_QUOTES:
        edits.append(f"dropped {len(out) - MAX_QUOTES} quote(s) beyond {MAX_QUOTES}")
        out = out[:MAX_QUOTES]
    return out, edits


def teacher_forced_logp(llm, p_ids, comp_ids):
    """log P(comp_ids | p_ids) from a llama-cpp-python Llama built with logits_all=False.

    Llama.eval only keeps per-position logits with logits_all=True, and then holds an n_ctx x vocab float32 buffer
    for the object's whole life (2.32 GB at n_ctx 4096) to score 16 tokens. This decodes the same n_batch chunks at
    the same positions as Llama.eval, but flags logits only on the rows that predict a completion token (plus each
    chunk's last token, as set_batch always does) and reads them with get_logits_ith. The forward pass is the same;
    only the number of output rows differs. Uses llama-cpp-python 0.3.36's private _ctx/_batch: re-check it, and
    re-run the old-vs-new equivalence gate, after any llama-cpp-python upgrade."""
    import numpy as np
    ids = list(p_ids) + list(comp_ids)
    first = len(p_ids) - 1                      # the row at position first + j predicts comp_ids[j]
    llm.reset()
    llm._ctx.kv_cache_seq_rm(-1, 0, -1)         # what Llama.eval does after reset()
    n_vocab, total = llm.n_vocab(), 0.0
    for start in range(0, len(ids), llm.n_batch):          # the same chunks as Llama.eval
        chunk = ids[start:start + llm.n_batch]
        llm._batch.set_batch(batch=chunk, n_past=start, logits_all=False)
        rows = [j for j in range(len(chunk)) if first <= start + j <= len(ids) - 2]
        for j in rows:
            llm._batch.batch.logits[j] = True               # logits only on the rows we score
        llm._ctx.decode(llm._batch)
        for j in rows:
            ptr = llm._ctx.get_logits_ith(j)
            if not ptr:
                raise RuntimeError(f"llama.cpp returned no logits for batch row {j}")
            row = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).astype(np.float64)
            row = row - row.max()
            total += row[comp_ids[start + j - first]] - np.log(np.exp(row).sum())
    llm.input_ids[:len(ids)] = ids              # leave the Llama's state as Llama.eval would
    llm.n_tokens = len(ids)
    return float(total)


class GGUFBackend:
    """The two model calls (abstain_score, generate) through llama-cpp-python, for the laptop build.

    GGUF can't toggle a LoRA on/off, so the two roles are two files: `verifier_gguf` is the frozen SFT
    (adapter off) that produces the abstain score, `generator_gguf` is the SFT+GRPO merge (adapter on)
    that writes answers. Scoring and generation reproduce evaluate.py/gguf_verifier_gate exactly:
    tokenise the prompt with add_bos=True, special=True (create_completion's default special=False would
    mangle the chat-template tokens), teacher-force the abstain completion and sum log_softmax over its
    tokens (teacher_forced_logp: only the completion rows get logits, so no n_ctx x vocab buffer),
    generate greedily (temperature 0) stopping at <|im_end|>.

    OS-portable: the only platform-specific piece is which llama-cpp-python wheel is installed (Metal on
    Mac, AVX2/CUDA on Windows/Linux). This class, and everything downstream of it, is identical.
    """
    def __init__(self, verifier_gguf, generator_gguf, tok, n_ctx=8192, verifier_n_ctx=4096,
                 n_gpu_layers=-1, n_threads=8):
        #different ctx for generator and verifier, verifier has fit method to enable truncations, otherwise too memory intensive for 0.5% of cases.
        from llama_cpp import Llama
        self.tok = tok
        self.comp_ids = tok(ABSTAIN_COMPLETION, add_special_tokens=False)["input_ids"]
        self.n_ctx, self.verifier_n_ctx = n_ctx, verifier_n_ctx
        common = dict(n_gpu_layers=n_gpu_layers, n_threads=n_threads, verbose=False)
        self.verifier = Llama(model_path=verifier_gguf, logits_all=False,
                              n_ctx=verifier_n_ctx, **common)   # teacher-forced logprobs via teacher_forced_logp
        self.generator = Llama(model_path=generator_gguf, logits_all=False, n_ctx=n_ctx, **common)
        # tokeniser-drift guard: the GGUF must tokenise the abstain completion to the SAME ids as HF
        gg = self.verifier.tokenize(ABSTAIN_COMPLETION.encode(), add_bos=False, special=True)
        assert gg == self.comp_ids, f"verifier GGUF completion ids differ from HF:\n {gg}\n {self.comp_ids}"

    # head: the system block defines the output contract the verifier was calibrated on, so it must
    # survive. tail: the prompt ends "Question: ...<|im_end|><|im_start|>assistant<think></think>",
    # and the scored completion is conditioned on exactly that, so it must survive too.
    HEAD_KEEP, TAIL_KEEP = 384, 768

    def _fit(self, p_ids):
        """Truncate p_ids from the MIDDLE so p_ids + comp_ids fits verifier_n_ctx.

        Without this, an over-length prompt makes llama_decode return 1 ("failed to find a memory
        slot for batch of size 512") 
        Truncating the MIDDLE (context paragraphs), not the left, is deliberate: left-truncation
        would eat the system instructions, and right-truncation would eat the question, so this biases TOWARD abstaining.
        """
        budget = self.verifier_n_ctx - len(self.comp_ids)
        if len(p_ids) <= budget:
            return p_ids
        keep_tail = min(self.TAIL_KEEP, budget - self.HEAD_KEEP)
        head, tail = p_ids[:budget - keep_tail], p_ids[len(p_ids) - keep_tail:]
        print(f"  [verifier] prompt {len(p_ids)} tok > budget {budget}: dropped "
              f"{len(p_ids) - budget} mid-context tokens", flush=True)
        return head + tail

    def abstain_score(self, prompt):
        """logP(<answer>ABSTAIN</answer><|im_end|> | prompt) through the verifier GGUF. Returns the same
        (negative) log-prob the HF path returns, so the reader's s = -lp and the tiers are unchanged."""
        p_ids = self.verifier.tokenize(prompt.encode(), add_bos=True, special=True)
        return teacher_forced_logp(self.verifier, self._fit(p_ids), self.comp_ids)

    def generate(self, prompt):
        """Greedy generation through the generator GGUF (temperature 0 == evaluate.generate's do_sample=False),
        stopping at <|im_end|>. Prompt tokenised with special=True so the chat-template tokens are real."""
        p_ids = self.generator.tokenize(prompt.encode(), add_bos=True, special=True)
        out = self.generator.create_completion(p_ids, max_tokens=MAX_NEW_TOKENS, temperature=0.0,
                                               stop=["<|im_end|>"])
        return out["choices"][0]["text"]


class _Dense(DenseRetriever):
    """DenseRetriever with an already-loaded embedder; the encoding is DenseRetriever's own."""
    def __init__(self, papers, model):
        self.model = model
        self.emb = {pid: model.encode(p.paragraphs, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
                    for pid, p in papers.items()}


class _Hybrid(HybridRetriever):
    """HybridRetriever (same top_k) that doesn't reload the embedder for every paper."""
    def __init__(self, papers, embedder, w=0.3, c=5):
        self.papers, self.w, self.c = papers, w, c
        self.bm25 = {pid: PaperIndex(p) for pid, p in papers.items()}
        self.dense = _Dense(papers, embedder)


def pick_device(device="auto"):
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_off_cuda(sft_path, adapter_path, device, dtype):
    """evaluate.load_model for a laptop (Apple Silicon MPS, or CPU): same tokenizer, padding, dtype and adapter."""
    import torch
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(sft_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16)
    model = PeftModel.from_pretrained(model.to(device), adapter_path)
    model.eval()
    return tok, model


class Reader:
    def __init__(self, sft_path, adapter_path, tiers, k=16, device="auto", dtype="fp32",
                 backend="hf", verifier_gguf=None, generator_gguf=None, n_gpu_layers=-1,
                 n_ctx=8192, verifier_n_ctx=4096, tokenizer_path=None, skip_below_low=True, embedder_path=None):
        # tiers: {"high": s_hi, "medium": s_med, "low": s_low} on the verifier score s = -logP(abstain).
        # An answer is shown with a confidence label when s >= low, abstained when s < low.
        # skip_below_low: the score comes from the prompt alone, so below `low` the answer is abstained whatever
        # the generator writes; skip generating it. Decisions are identical either way. False only for checks
        # that compare every raw output (readerchecks/).
        # backend "hf": the torch model (cutpoints from configs/reader_4b.json, dtype as calibrated, fp32).
        # backend "gguf": quantised GGUFs, re-calibrated on Q4).
        device = pick_device(device)
        self.backend = backend
        if backend == "gguf":
            # the tokenizer is still the HF, the two model calls go through GGUFBackend. The embedder (retrieval) runs on CPU.
            self.tok = AutoTokenizer.from_pretrained(tokenizer_path or sft_path)
            self.tok.padding_side = "left"
            self.gguf = GGUFBackend(verifier_gguf, generator_gguf, self.tok,
                                    n_ctx=n_ctx, verifier_n_ctx=verifier_n_ctx,
                                    n_gpu_layers=n_gpu_layers)
            self.model = None
        elif device == "cuda":                                           # exactly as calibrated (incl. TF32)
            from qasper.evaluate import load_model
            self.tok, self.model = load_model(sft_path, adapter_path, dtype)
            self.gguf = None
        else:
            self.tok, self.model = load_off_cuda(sft_path, adapter_path, device, dtype)
            self.gguf = None
        self.device = device
        if backend == "gguf" and embedder_path:                         # bge-small through onnxruntime, no torch
            from qasper.onnx_embed import OnnxEmbedder
            self.embedder = OnnxEmbedder(embedder_path)
        else:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(EMBEDDER, device=device)
        self.tiers, self.k = tiers, k
        self.skip_below_low = skip_below_low
        self.paper = self.retriever = None
        self.display = {}

    def confidence(self, score):
        """HIGH / MEDIUM / LOW for a shown answer, or None if the score is below the abstain cutpoint."""
        if score >= self.tiers["high"]:
            return "high"
        if score >= self.tiers["medium"]:
            return "medium"
        if score >= self.tiers["low"]:
            return "low"
        return None

    def index(self, paper, display_map=None):
        self.paper, self.display = paper, display_map or {}
        self.retriever = _Hybrid({paper.paper_id: paper}, self.embedder)

    def abstain_score(self, prompt):
        """log P(<answer>ABSTAIN</answer> | prompt) from the frozen SFT (adapter off), as the verifier was calibrated."""
        if self.gguf is not None:
            return self.gguf.abstain_score(prompt)
        from qasper.evaluate import abstain_logps
        n = len(self.tok(prompt)["input_ids"])
        with self.model.disable_adapter():
            return abstain_logps(self.model, self.tok, [SimpleNamespace(prompt=prompt, n_tokens=n)], batch_size=1)[0]

    def generate(self, prompt):
        """The GRPO generator (adapter on), greedy."""
        if self.gguf is not None:
            return self.gguf.generate(prompt)
        from qasper.evaluate import generate
        return generate(self.model, self.tok, [prompt], MAX_NEW_TOKENS)[0]

    def _restore(self, text):
        return restore(text, self.display)

    def _quote(self, p, quote, section, paragraph):
        """Split the paragraph around the quote in placeholder text (the quote check guarantees it's there), then
        restore each piece, so the restored quote always lines up with its paragraph for highlighting."""
        i = paragraph.find(quote)
        before, after = paragraph[:i], paragraph[i + len(quote):]
        r = self._restore
        return {"p": p, "section": section, "quote": r(quote), "paragraph": r(before) + r(quote) + r(after),
                "before": r(before), "after": r(after)}

    def ask(self, question):
        paper = self.paper
        ranked = self.retriever.top_k(paper.paper_id, question, self.k)
        context, p_to_idx = build_context(paper, ranked, self.k)
        prompt = format_prompt(self.tok, context, question)
        lp = self.abstain_score(prompt)
        score = -lp                                                      # confidence tiers on this score
        confidence = self.confidence(score)
        debug = {"reason": None, "abstain_score": lp, "verifier_score": score, "tiers": self.tiers, "outcome": None,
                 "confidence": None, "raw_output": None, "repaired": None, "context": context, "p_to_idx": p_to_idx,
                 "n_prompt_tokens": len(self.tok(prompt)["input_ids"])}

        def abstain(reason, outcome):
            debug["reason"], debug["outcome"] = reason, outcome
            return {"status": "abstain", "answer": ABSTAIN, "quotes": [], "references": [], "debug": debug}

        if confidence is None and self.skip_below_low:                  # abstained whatever it would write
            return abstain("verifier", "not generated")
        raw = self.generate(prompt)
        debug["raw_output"] = raw
        parsed = parse(raw)
        if parsed is None:
            return abstain("format", "format")
        quotes, answer = parsed
        if answer == ABSTAIN:
            return abstain("generator abstained", "abstain")
        if not format_ok(quotes, answer, False, p_to_idx):
            quotes, edits = repair_quotes(quotes, answer)
            if not edits or not format_ok(quotes, answer, False, p_to_idx):
                return abstain("format", "format")
            debug["repaired"] = edits
        check = quote_check(quotes, paper, p_to_idx)
        if check != "ok":
            return abstain(check, check)                                 # "fabricated" or "miscited"
        debug["outcome"] = "answer"                                      # passes the rules: the tier decides
        debug["confidence"] = confidence
        if confidence is None:                                           # below the abstain cutpoint
            debug["reason"] = "verifier"
            return {"status": "abstain", "answer": ABSTAIN, "quotes": [], "references": [], "debug": debug}

        shown = [self._quote(p, q, paper.sections[p_to_idx[p]], norm_ws(paper.paragraphs[p_to_idx[p]]))
                 for p, q in quotes]
        cited = dict.fromkeys(BIBREF_RE.findall(" ".join([answer] + [q for _, q in quotes])))
        refs = [{"placeholder": ph, "cited_as": self.display[ph][0], "reference": self.display[ph][1]}
                for ph in cited if ph in self.display]
        return {"status": "answer", "confidence": confidence, "answer": self._restore(answer),
                "quotes": shown, "references": refs, "debug": debug}
