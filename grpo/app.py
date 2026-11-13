"""LocalPaperReader: upload a PDF, ask questions about it. Each question is answered on its own (the chat history
is shown, not sent to the model). Models load once, at startup.

    PYTHONPATH=. python app.py --config configs/reader_1.7b.json
"""
import argparse
import html
import inspect
import json
import os

import gradio as gr

from qasper.pdf_parse_v3 import parse_pdf
from qasper.reader import Reader
from qasper.reward import ABSTAIN

CAPTION = "Figures and tables"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reader_1.7b.json")
    ap.add_argument("--port", type=int, default=7860)
    return ap.parse_args()


def highlight(q):
    """The quote's paragraph, HTML-escaped, with the quote in <mark>."""
    return html.escape(q["before"]) + "<mark>" + html.escape(q["quote"]) + "</mark>" + html.escape(q["after"])


CONFIDENCE_BADGE = {
    "high": "🟢 **High confidence**: most likely correct.",
    "medium": "🟡 **Medium confidence**: plausible; check it against the quoted passage.",
    "low": "🟠 **Low confidence**: a lead, not an answer; verify against the quoted passage.",
}


def render(result):
    if result["status"] != "answer":
        return html.escape(ABSTAIN)
    parts = []
    badge = CONFIDENCE_BADGE.get(result.get("confidence"))
    if badge:
        parts += [badge, ""]
    parts += [f"**{html.escape(result['answer'])}**", ""]
    for n, q in enumerate(result["quotes"], 1):
        parts += [f"> {html.escape(q['quote'])}", f"> — *{html.escape(q['section'] or 'Unnamed section')}*", "",
                  f"<details><summary>Show paragraph {n}</summary>{highlight(q)}</details>", ""]
    if result["references"]:
        parts.append("**References cited**")
        parts += [f"- {html.escape(r['reference'] or r['cited_as'])}" for r in result["references"]]
    return "\n".join(parts)


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    reader = Reader(cfg["sft_path"], cfg["adapter_path"], cfg["tiers"], cfg["k"],
                    cfg.get("device", "auto"), cfg["dtype"],
                    backend=cfg.get("backend", "hf"),
                    verifier_gguf=cfg.get("verifier_gguf"), generator_gguf=cfg.get("generator_gguf"),
                    n_gpu_layers=cfg.get("n_gpu_layers", -1),
                    n_ctx=cfg.get("n_ctx", 8192),
                    verifier_n_ctx=cfg.get("verifier_n_ctx", 4096),
                    tokenizer_path=cfg.get("tokenizer_path"),
                    embedder_path=cfg.get("embedder_path"))
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
            result = reader.ask(question)
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

    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
