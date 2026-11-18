"""LocalPaperReader: upload a PDF, ask questions about it. Each question is answered on its own (the chat history
is shown, not sent to the model). Models load once, at startup.

    PYTHONPATH=. python app.py                                   # configs/reader_4b_gguf.json, opens the browser
    PYTHONPATH=. python app.py --config configs/reader_4b.json --no-browser    # fp32 on an HPC GPU node
"""
import argparse
import inspect
import json
import os
import sys

import gradio as gr

import paperstore
from paperfetch import download
from qasper import overview
from qasper.pdf_parse_v3 import parse_pdf
from readergui import styling
from readergui.rendering import (ASSESSING, DOWNLOADING, INDEXING, NO_PAPER, PARSING, WRITING, header, looking_in,
                                 oval, render, stage, start)
from readergui.stages import StagedReader, ask_stages, overview_stages

CAPTION = "Figures and tables"
APP_DIR = os.path.dirname(os.path.abspath(__file__))        # the app's own folder, also inside a bundle
PATH_KEYS = ("verifier_gguf", "generator_gguf", "generator_lora", "tokenizer_path", "embedder_path")
MODEL_KEYS = ("verifier_gguf", "generator_gguf", "generator_lora")      # too big to ship: fetched on first launch


def data_dir():
    """Per-user folder for the downloaded model files and the kept papers."""
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
    ap.add_argument("--debug", action="store_true", help="show the debug panel under the question box")
    return ap.parse_args()


def build_demo(cfg, debug=False):
    """The reader and the interface for a config from load_config. The caller launches it.
    debug: also show the debug panel (app.py --debug)."""
    # a gguf config keeps sft_path / adapter_path under "provenance" (not loaded), so .get here
    reader = StagedReader(cfg.get("sft_path"), cfg.get("adapter_path"), cfg["tiers"], cfg["k"],
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

    papers = os.path.join(data_dir(), "papers")

    def open_paper(path):
        """Parse, keep and index the PDF at `path`, with the stages in the top bar."""
        yield None, header(stage=PARSING), gr.update(value=[], placeholder=start())
        try:
            paper, display = parse_pdf(path, os.path.basename(path).removesuffix(".pdf"))
        except Exception as e:                                           #bad PDF or scanned
            yield None, header(summary=f"Couldn't parse the PDF: {e}"), gr.update()
            return
        try:
            paperstore.keep(path, paper.title, papers)
        except OSError as e:                                             # not kept: it still opens
            print(f"paper not kept: {e}")
        yield None, header(stage=INDEXING), gr.update()
        with reader.busy:                                                # not under a question still in the model
            reader.index(paper, display)
        n_cap = paper.sections.count(CAPTION)
        n_sec = len(set(s for s in paper.sections if s not in ("Abstract", CAPTION)))
        n_ref = sum(ph.startswith("BIBREF") for ph in display)
        summary = f"{n_sec} sections · {len(paper.paragraphs) - n_cap} paragraphs · {n_ref} references{placeholder_note}"
        yield (paper, display), header(paper.title or paper.paper_id, summary), gr.update(value=[], placeholder=start(book=False))

    def upload(path):
        if not path:
            yield None, header(summary="No file."), gr.update()
            return
        yield from open_paper(path)

    def from_link(link):
        if not (link or "").strip():
            yield gr.update(), gr.update(), gr.update()
            return
        yield None, header(stage=DOWNLOADING), gr.update(value=[], placeholder=start())
        try:
            path = download(link)
        except RuntimeError as e:
            yield None, header(summary=f"Couldn't download the paper: {e}"), gr.update()
            return
        yield from open_paper(path)

    def ask(question, history, state):
        question = (question or "").strip()
        if not question:
            yield history, "", gr.update()
            return
        history = history + [{"role": "user", "content": question}]

        def reply(content):                                              # the chat with this as the answer so far
            return history + [{"role": "assistant", "content": content}], "", gr.update()

        if state is None:
            yield reply(oval(NO_PAPER))
            return
        yield history, "", gr.update()                                   # the question shows at once
        paper, display = state
        if reader.paper is not paper:                                    # another session indexed a different paper
            with reader.busy:
                reader.index(paper, display)
        try:
            kind = overview.detect(question)                             # a general question: answered in parts
            if not kind and overview.origin_term(question):              # "Where does X come from?": no model
                result, steps = overview.origin(reader, question), None
            else:
                steps = overview_stages(reader, question, kind) if kind else ask_stages(reader, question)
            x = None
            while steps:
                for name, value in steps:                                # the same stages for both
                    if name == "assess":
                        yield reply(stage(ASSESSING, looking_in(value["sections"], value["n_passages"])))
                    elif name == "write":
                        yield reply(stage(WRITING))
                    elif x is None or value["found"]:                    # a fallback that finds nothing isn't shown
                        result = value
                steps = None
                weak = result["status"] != "answer" or result.get("confidence") == "low"
                if not kind and x is None and weak:                  # the paper's own definition beats a weak answer
                    defined = overview.abbreviation(reader, question) or overview.description(reader, question)
                    if not defined and result["status"] != "answer":
                        with reader.busy:                            # definition_answer runs the model
                            defined = overview.definition_answer(reader, question, result)
                    if defined:
                        result = defined
                    elif result["status"] != "answer":               # still withheld: concept fallback
                        x = overview.concept_for(reader, question)
                        steps = x and overview_stages(reader, question, "concept", x)
        except Exception as e:                                           # one message for every failure
            result = {"status": "abstain", "debug": {"reason": f"error: {type(e).__name__}: {e}"}}
        debug = json.loads(json.dumps(result["debug"]))                  # gr.JSON needs str keys (p_to_idx)
        yield history + [{"role": "assistant", "content": render(result)}], "", debug

    chat_kwargs = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot).parameters else {}
    with gr.Blocks(title="LocalPaperReader", fill_width=True, fill_height=True) as demo:
        state = gr.State(None)
        card_open = gr.State(False)
        with gr.Row(elem_id="lpr-bar"):                                  # the ids are styled in readergui/styling.py
            open_pdf = gr.Button("Open PDF", elem_id="lpr-open", scale=0)
            head = gr.HTML(header(), elem_id="lpr-head")
        with gr.Column(visible=False, elem_id="lpr-card") as card:       # Open PDF's card
            link = gr.Textbox(show_label=False, container=False, placeholder="Paste link", max_lines=1, elem_id="lpr-link")
            pdf = gr.UploadButton("From device", file_types=[".pdf"], type="filepath", elem_id="lpr-device", scale=0)
        chat = gr.Chatbot(elem_id="lpr-chat", show_label=False, container=False, scale=1, layout="bubble",
                          placeholder=start(), buttons=[], latex_delimiters=[], **chat_kwargs)
        question = gr.Textbox(elem_id="lpr-ask", show_label=False, container=False, placeholder="Ask a question!",
                              submit_btn=True, max_lines=4)
        debug_json = gr.JSON(label="debug", visible=False)
        if debug:
            show_debug = gr.Checkbox(label="debug", value=False)
            show_debug.change(lambda on: gr.update(visible=on), show_debug, debug_json)

        close = lambda: (False, gr.update(visible=False))
        open_pdf.click(lambda o: (not o, gr.update(visible=not o)), card_open, [card_open, card])
        pdf.upload(close, None, [card_open, card])
        link.submit(close, None, [card_open, card])
        pdf.upload(upload, pdf, [state, head, chat], show_progress="hidden")         # the stages are the progress
        link.submit(from_link, link, [state, head, chat], show_progress="hidden")
        link.submit(lambda: "", None, link)
        question.submit(ask, [question, chat, state], [chat, question, debug_json], show_progress="hidden")

    return demo


def main():
    args = parse_args()
    demo = build_demo(load_config(args.config), debug=args.debug)
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, show_error=True,
                        inbrowser=not args.no_browser, **styling.LAUNCH)


if __name__ == "__main__":
    main()
