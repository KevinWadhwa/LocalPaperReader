"""Stage reports for the interface: what the reader is doing with a question, while it does it.

qasper/reader.py is not changed. StagedReader adds two small things around it, and ask_stages runs the unchanged
Reader.ask in a thread, so the interface can show each stage while a model call is still running.
"""
import queue
import threading

from qasper.context import build_context
from qasper.reader import BIBREF_RE, Reader


class StagedReader(Reader):
    """Reader that says when the generator starts, and gives each quote the references it cites."""
    on_write = None                                 # called as generation starts; set by ask_stages
    busy = threading.Lock()                         # one question in the model at a time

    def generate(self, prompt):
        if self.on_write is not None:
            self.on_write()
        return super().generate(prompt)

    def _quote(self, p, quote, section, paragraph):
        q = super()._quote(p, quote, section, paragraph)
        q["references"] = [ph for ph in dict.fromkeys(BIBREF_RE.findall(quote)) if ph in self.display]   # placeholders
        return q


def ask_stages(reader, question):
    """Reader.ask as a generator: yields ("assess", {"sections", "n_passages"}) as the judge starts reading,
    ("write", None) as the generator starts, then ("done", result). A repeat question, which Reader.ask answers
    from memory, yields "done" alone."""
    if question in reader._answers:
        yield "done", reader.ask(question)
        return
    paper = reader.paper
    ranked = reader.retriever.top_k(paper.paper_id, question, reader.k)     # the passages Reader._ask will show
    _, p_to_idx = build_context(paper, ranked, reader.k)
    sections = [(paper.sections[i] or "").split(":::")[0].strip() for i in p_to_idx.values()]   # top level, reading order
    yield "assess", {"sections": [s for s in dict.fromkeys(sections) if s], "n_passages": len(p_to_idx)}
    events = queue.Queue()

    def work():
        with reader.busy:
            reader.on_write = lambda: events.put(("write", None))
            try:
                events.put(("done", reader.ask(question)))
            except Exception as e:                  # raised again below, in the caller's thread
                events.put(("error", e))
            finally:
                reader.on_write = None

    threading.Thread(target=work, daemon=True).start()
    while True:
        name, value = events.get()
        if name == "error":
            raise value
        yield name, value
        if name == "done":
            return
