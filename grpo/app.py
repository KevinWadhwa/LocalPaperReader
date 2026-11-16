"""LocalPaperReader: upload a PDF, ask questions about it. Each question is answered on its own (the chat history
is shown, not sent to the model). Models load once, at startup.

    PYTHONPATH=. python app.py                                   # configs/reader_4b_gguf.json, opens the browser
    PYTHONPATH=. python app.py --config configs/reader_4b.json --no-browser    # fp32 on an HPC GPU node
"""
import argparse
import html
import inspect
import json
import os
import sys

import gradio as gr

from qasper import overview
from qasper.pdf_parse_v3 import parse_pdf
from qasper.reader import Reader
from qasper.reward import ABSTAIN

CAPTION = "Figures and tables"
APP_DIR = os.path.dirname(os.path.abspath(__file__))        # the app's own folder, also inside a bundle
PATH_KEYS = ("verifier_gguf", "generator_gguf", "generator_lora", "tokenizer_path", "embedder_path")
MODEL_KEYS = ("verifier_gguf", "generator_gguf", "generator_lora")      # too big to ship: fetched on first launch


def data_dir():
    """Per-user folder for the downloaded model files."""
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "LocalPaperReader")


def resolve_paths(cfg):
    """The config's file paths made absolute: relative to the app's folder, not to where it was started from
    (a double-clicked app starts elsewhere). A model file not found there is looked for in the per-user data
    folder, where the first-launch download puts it."""
    cfg = dict(cfg)
    for k in PATH_KEYS:
        if cfg.get(k):
            p = os.path.join(APP_DIR, cfg[k])               # an absolute path stays as it is
            if k in MODEL_KEYS and not os.path.exists(p):
                p = os.path.join(data_dir(), "models", os.path.basename(cfg[k]))
            cfg[k] = p
    return cfg


def load_config(path):
    """The config with its file paths resolved. A `path` not found from the current folder is taken relative to
    the app's folder."""
    if not os.path.exists(path):
        path = os.path.join(APP_DIR, path)
    return resolve_paths(json.load(open(path)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reader_4b_gguf.json")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab (headless / HPC)")
    return ap.parse_args()


def highlight(q):
    """The quote's paragraph, HTML-escaped, with the quote in <mark>."""
    return html.escape(q["before"]) + "<mark>" + html.escape(q["quote"]) + "</mark>" + html.escape(q["after"])


CONFIDENCE_BADGE = {
    "high": "🟢 **High confidence**: most likely correct.",
    "medium": "🟡 **Medium confidence**: plausible; check it against the quoted passage.",
    "low": "🟠 **Low confidence**: a lead, not an answer; verify against the quoted passage.",
}


def render_quotes(quotes, first=1):
    """Each quote with its section, and its paragraph behind a drop-down."""
    parts = []
    for n, q in enumerate(quotes, first):
        parts += [f"> {html.escape(q['quote'])}", f"> — *{html.escape(q['section'] or 'Unnamed section')}*", "",
                  f"<details><summary>Show paragraph {n}</summary>{highlight(q)}</details>", ""]
    return parts


def render_references(refs):
    if not refs:
        return []
    return ["**References cited**"] + [f"- {html.escape(r['reference'] or r['cited_as'])}" for r in refs]


def render(result):
    if result["status"] == "overview":
        return render_overview(result)
    if result["status"] != "answer":
        return html.escape(ABSTAIN)
    parts = []
    badge = CONFIDENCE_BADGE.get(result.get("confidence"))
    if badge:
        parts += [badge, ""]
    parts += [f"**{html.escape(result['answer'])}**", ""]
    parts += render_quotes(result["quotes"])
    parts += render_references(result["references"])
    return "\n".join(parts)


def render_overview(result):
    """A general question's aspects: how many were found, each with its own confidence and quotes, the missing
    ones by name, and where it looked, so a partial overview never reads as a complete summary."""
    if not result["found"]:
        return html.escape(ABSTAIN)
    parts = [f"### {html.escape(result['heading'])}: {result['found']} of {result['total']} aspects found", ""]
    n, refs = 1, {}
    for a in result["aspects"]:
        if a["status"] == "same":
            parts += [f"**{html.escape(a['label'])}**: same as {html.escape(a['same_as'])} above.", ""]
        elif a["status"] == "answer":
            parts += [f"#### {html.escape(a['label'])}", CONFIDENCE_BADGE.get(a["confidence"], ""), "",
                      f"**{html.escape(a['answer'])}**", ""]
            parts += render_quotes(a["quotes"], n)
            n += len(a["quotes"])
            refs.update({r["placeholder"]: r for r in a["references"]})
    if result["missing"]:
        parts += [f"*Not found with enough confidence: {html.escape(', '.join(result['missing']))}.*", ""]
    parts += [f"<small>From {result['n_passages']} passages in: {html.escape(' · '.join(result['sections']))}</small>", ""]
    parts += render_references(list(refs.values()))
    return "\n".join(parts)


def build_demo(cfg):
    """The reader and the interface for a config from load_config. The caller launches it."""
    # a gguf config keeps sft_path / adapter_path under "provenance" (not loaded), so .get here
    reader = Reader(cfg.get("sft_path"), cfg.get("adapter_path"), cfg["tiers"], cfg["k"],
                    cfg.get("device", "auto"), cfg.get("dtype", "fp32"),
                    backend=cfg.get("backend", "hf"),
                    verifier_gguf=cfg.get("verifier_gguf"), generator_gguf=cfg.get("generator_gguf"),
                    n_gpu_layers=cfg.get("n_gpu_layers", -1),
                    n_ctx=cfg.get("n_ctx", 8192),
                    verifier_n_ctx=cfg.get("verifier_n_ctx", 4096),
                    tokenizer_path=cfg.get("tokenizer_path"),
                    embedder_path=cfg.get("embedder_path"),
                    copy_ahead=cfg.get("copy_ahead", False),
                    generator_lora=cfg.get("generator_lora"),
                    shared_cache=cfg.get("shared_cache", False))
    if reader.backend == "gguf":
        print(f"backend gguf, n_gpu_layers {cfg.get('n_gpu_layers', -1)}, embedder on {reader.device}")
    else:
        print(f"backend {reader.backend} on {reader.device}")
    placeholder_note = (" · confidence cutpoints are placeholders: every rule-passing answer is shown as low"
                        if cfg.get("tiers_placeholder") else "")

    def upload(path, progress=gr.Progress()):
        if not path:
            return None, "No file.", []
        progress(0.1, desc="Parsing the PDF")
        try:
            paper, display = parse_pdf(path, os.path.basename(path).removesuffix(".pdf"))
        except Exception as e:                                           #bad PDF or scanned
            return None, f"Couldn't parse the PDF: {e}", []
        progress(0.7, desc="Indexing")
        reader.index(paper, display)
        n_cap = paper.sections.count(CAPTION)
        n_sec = len(set(s for s in paper.sections if s not in ("Abstract", CAPTION)))
        n_ref = sum(ph.startswith("BIBREF") for ph in display)
        status = (f"**{html.escape(paper.title or paper.paper_id)}**: parsed {len(paper.paragraphs) - n_cap} paragraphs, "
                  f"{n_sec} sections, {n_cap} captions, {n_ref} cited references{placeholder_note}")
        return (paper, display), status, []

    def ask(question, history, state):
        question = (question or "").strip()
        if not question:
            return history, "", gr.update()
        history = history + [{"role": "user", "content": question}]
        if state is None:
            return history + [{"role": "assistant", "content": "Upload a paper first."}], "", gr.update()
        paper, display = state
        if reader.paper is not paper:                                    # another session indexed a different paper
            reader.index(paper, display)
        try:
            kind = overview.detect(question)                             # a general question: answered in parts
            result = overview.run(reader, question, kind) if kind else reader.ask(question)
        except Exception as e:                                           # one message for every failure
            result = {"status": "abstain", "debug": {"reason": f"error: {type(e).__name__}: {e}"}}
        debug = json.loads(json.dumps(result["debug"]))                  # gr.JSON needs str keys (p_to_idx)
        return history + [{"role": "assistant", "content": render(result)}], "", debug

    chat_kwargs = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot).parameters else {}
    with gr.Blocks(title="LocalPaperReader") as demo:
        gr.Markdown("## LocalPaperReader\nAnswers by quoting the paper, or says it couldn't find the answer.")
        state = gr.State(None)
        pdf = gr.File(label="Paper (PDF)", file_types=[".pdf"], type="filepath")
        status = gr.Markdown("Upload a PDF to start.")
        chat = gr.Chatbot(label="Questions", height=520, **chat_kwargs)
        question = gr.Textbox(label="Question", placeholder="e.g. Which datasets do they use?")
        show_debug = gr.Checkbox(label="debug", value=False)
        debug = gr.JSON(label="debug", visible=False)

        pdf.upload(upload, pdf, [state, status, chat])
        question.submit(ask, [question, chat, state], [chat, question, debug])
        show_debug.change(lambda on: gr.update(visible=on), show_debug, debug)

    return demo


def main():
    args = parse_args()
    build_demo(load_config(args.config)).queue().launch(server_name="127.0.0.1", server_port=args.port,
                                                        show_error=True, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
