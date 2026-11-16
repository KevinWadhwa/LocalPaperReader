"""What the interface shows, as HTML: the stage messages with their animated icons."""
import base64
import html
import itertools
import os

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icons")     # grpo/icons
ICONS = {f[:-4]: open(os.path.join(ICON_DIR, f), "rb").read() for f in os.listdir(ICON_DIR) if f.endswith(".svg")}

# stage -> (message, icon file name without .svg; "-dark" is added for dark mode)
PARSING = ("Parsing the PDF…", "01-flip-through")
INDEXING = ("Reading and indexing the paper…", "01-flip-through")
ASSESSING = ("Retrieved the most relevant passages. Assessing whether they answer your question…", "02-03-alternate")
WRITING = ("Reasoning out the answer…", "04-teach-it")

_icon_uses = itertools.count()


def icon(name):
    """The animated icon as two embedded images, light and dark: the styles show the one for the theme. The
    counter makes each use a new image, so its loop starts from the beginning."""
    tag = f"<!--{next(_icon_uses)}-->".encode()
    return "".join(f'<img class="lpr-icon {cls}" alt="" src="data:image/svg+xml;base64,'
                   f'{base64.b64encode(ICONS[name + suffix] + tag).decode()}">'
                   for cls, suffix in (("lpr-light", ""), ("lpr-dark", "-dark")))


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
