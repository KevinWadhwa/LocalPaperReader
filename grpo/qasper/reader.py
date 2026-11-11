"""Answer one question about one paper end to end: retrieve, prompt, abstain score, generate, rules, verifier.

The two model calls are `abstain_score` and `generate`; a GGUF backend replaces only those two.
"""
import re
from types import SimpleNamespace

# torch / peft / qasper.evaluate are imported lazily (inside the HF-path functions only), so the
# GGUF laptop build never imports torch and the PyInstaller bundle can exclude it. transformers
# (tokenizer) and sentence_transformers (retrieval embedder) are needed by both backends.
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from qasper.bm25 import PaperIndex
from qasper.context import build_context
from qasper.dense import DenseRetriever
from qasper.hybrid import HybridRetriever
from qasper.pdf_parse import restore
from qasper.prompt import format_prompt
from qasper.reward import ABSTAIN, format_ok, norm_ws, parse, quote_check

EMBEDDER = "BAAI/bge-small-en-v1.5"         # DenseRetriever's default
MAX_NEW_TOKENS = 384
BIBREF_RE = re.compile(r"\bBIBREF\d+\b")
ABSTAIN_COMPLETION = f"<answer>{ABSTAIN}</answer><|im_end|>"


class GGUFBackend:
    """The two model calls (abstain_score, generate) through llama-cpp-python, for the laptop build.

    GGUF can't toggle a LoRA on/off, so the two roles are two files: `verifier_gguf` is the frozen SFT
    (adapter off) that produces the abstain score, `generator_gguf` is the SFT+GRPO merge (adapter on)
    that writes answers. Scoring and generation reproduce evaluate.py/gguf_verifier_gate exactly:
    tokenise the prompt with add_bos=True, special=True (create_completion's default special=False would
    mangle the chat-template tokens), teacher-force the abstain completion with logits_all=True and sum
    log_softmax over its tokens, generate greedily (temperature 0) stopping at <|im_end|>.

    OS-portable: the only platform-specific piece is which llama-cpp-python wheel is installed (Metal on
    Mac, AVX2/CUDA on Windows/Linux). This class, and everything downstream of it, is identical.
    """
    def __init__(self, verifier_gguf, generator_gguf, tok, n_ctx=8192, verifier_n_ctx=4096,
                 n_gpu_layers=-1, n_threads=8):
        # The two models get DIFFERENT context sizes on purpose; do NOT collapse them into one.
        #   generator (logits_all=False) -> 8192. No per-position logits buffer, so context is
        #     nearly free, and the generator gate that PASSED on the Mac ran at 8192: its longest
        #     dev prompt was 4953 tokens and scored f1 1.000 (== fp32), so nothing was truncated.
        #     4096 would silently truncate ~1% of dev prompts and more of real uploaded papers.
        #   verifier (logits_all=True) -> 4096, and do NOT raise it casually. llama-cpp-python
        #     0.3.36 only fills llm.scores when logits_all=True, and it then allocates
        #     (n_ctx, 151936) float32 ONCE IN Llama.__init__ and holds it for the object's whole
        #     life. That is SUSTAINED RSS, not a peak, and it is sized to the ceiling, not to the
        #     current prompt: 2.32GB at 4096, 2.90GB at 5120, 3.48GB at 6144, 4.64GB at 8192
        #     (8192 OOM-killed the gate). So n_ctx here is a permanent memory bill.
        #     Why not logits_all=False (which would cost only n_batch rows == 0.29GB)? Because in
        #     0.3.36 Llama.eval's logits_all=False branch is literally `pass` — the logit copy is
        #     commented out with "logits are only needed for logprobs which requires logits_all".
        #     self.scores would stay all zeros and the abstain score would be silently wrong, not
        #     an error. Verified by reading the installed source; re-check before changing this.
        #   4096 does NOT fit all of dev: prompt+completion+BOS exceeds it on 5/904 rows (0.55%),
        #     the longest 4970 tokens, and that is why a full-dev reader run at 4096 died at row
        #     ~191 with "failed to find a memory slot for batch of size 512" (llama_decode 1).
        #     The fix is abstain_score's _fit guard, NOT a bigger buffer: a user-uploaded paper can
        #     out-length any ceiling we pick, so truncation is needed regardless, and once it
        #     exists a bigger n_ctx only buys not-truncating those 5 rows for ~0.6-1.2GB resident.
        #     Raise it only if truncation is shown to cost real accuracy. See _fit.
        import numpy as np
        from llama_cpp import Llama
        self.np = np
        self.tok = tok
        self.comp_ids = tok(ABSTAIN_COMPLETION, add_special_tokens=False)["input_ids"]
        self.n_ctx, self.verifier_n_ctx = n_ctx, verifier_n_ctx
        common = dict(n_gpu_layers=n_gpu_layers, n_threads=n_threads, verbose=False)
        self.verifier = Llama(model_path=verifier_gguf, logits_all=True,
                              n_ctx=verifier_n_ctx, **common)   # teacher-forced logprobs
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
        slot for batch of size 512") and the whole run dies — that is exactly how the full-dev
        reader check crashed at row ~191 when the verifier was at n_ctx 4096. Every dev prompt fits
        at 6144, so this should never fire on dev; it exists because a user-uploaded paper can be
        longer than anything in dev and the app must degrade instead of raising.

        Truncating the MIDDLE (context paragraphs), not the left, is deliberate: left-truncation
        would eat the system instructions, and right-truncation would eat the question. Both would
        change the distribution the abstain score is calibrated against far more than dropping some
        retrieved context does. The score from a truncated prompt is still a real score, but it saw
        less evidence, so it biases TOWARD abstaining (the safe direction for a grounded reader).
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
        np = self.np
        p_ids = self.verifier.tokenize(prompt.encode(), add_bos=True, special=True)
        p_ids = self._fit(p_ids)
        self.verifier.reset()
        self.verifier.eval(p_ids + self.comp_ids)
        total = 0.0
        for j, tok_id in enumerate(self.comp_ids):
            row = np.array(self.verifier.scores[len(p_ids) + j - 1], dtype=np.float64)
            row = row - row.max()
            total += row[tok_id] - np.log(np.exp(row).sum())
        return float(total)

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
                 n_ctx=8192, verifier_n_ctx=4096):
        # tiers: {"high": s_hi, "medium": s_med, "low": s_low} on the verifier score s = -logP(abstain), picked on dev.
        # An answer is shown with a confidence label when s >= low, abstained when s < low.
        # backend "hf": the torch model (cutpoints from configs/reader_4b.json, dtype as calibrated, fp32).
        # backend "gguf": the two quantised GGUFs (cutpoints from configs/reader_4b_gguf.json, RE-calibrated on Q4).
        device = pick_device(device)
        self.backend = backend
        if backend == "gguf":
            # the tokenizer is still the HF one (prompts/chat template built identically); the two model
            # calls go through GGUFBackend. The embedder (retrieval) runs on CPU/MPS as usual.
            self.tok = AutoTokenizer.from_pretrained(sft_path)
            self.tok.padding_side = "left"
            self.gguf = GGUFBackend(verifier_gguf, generator_gguf, self.tok,
                                    n_ctx=n_ctx, verifier_n_ctx=verifier_n_ctx,
                                    n_gpu_layers=n_gpu_layers)
            self.model = None
        elif device == "cuda":                                           # exactly as calibrated (incl. TF32)
            self.tok, self.model = load_model(sft_path, adapter_path, dtype)
            self.gguf = None
        else:
            self.tok, self.model = load_off_cuda(sft_path, adapter_path, device, dtype)
            self.gguf = None
        self.device = device
        self.embedder = SentenceTransformer(EMBEDDER, device=device)
        self.tiers, self.k = tiers, k
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
        n = len(self.tok(prompt)["input_ids"])
        with self.model.disable_adapter():
            return abstain_logps(self.model, self.tok, [SimpleNamespace(prompt=prompt, n_tokens=n)], batch_size=1)[0]

    def generate(self, prompt):
        """The GRPO generator (adapter on), greedy."""
        if self.gguf is not None:
            return self.gguf.generate(prompt)
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
        raw = self.generate(prompt)
        score = -lp                                                      # confidence tiers on this score
        debug = {"reason": None, "abstain_score": lp, "verifier_score": score, "tiers": self.tiers, "outcome": None,
                 "confidence": None, "raw_output": raw, "context": context, "p_to_idx": p_to_idx,
                 "n_prompt_tokens": len(self.tok(prompt)["input_ids"])}

        def abstain(reason, outcome):
            debug["reason"], debug["outcome"] = reason, outcome
            return {"status": "abstain", "answer": ABSTAIN, "quotes": [], "references": [], "debug": debug}

        parsed = parse(raw)
        if parsed is None:
            return abstain("format", "format")
        quotes, answer = parsed
        if answer == ABSTAIN:
            return abstain("generator abstained", "abstain")
        if not format_ok(quotes, answer, False, p_to_idx):
            return abstain("format", "format")
        check = quote_check(quotes, paper, p_to_idx)
        if check != "ok":
            return abstain(check, check)                                 # "fabricated" or "miscited"
        debug["outcome"] = "answer"                                      # passes the rules: the tier decides
        confidence = self.confidence(score)
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
