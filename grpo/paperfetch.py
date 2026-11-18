"""A paper from a link: arXiv, OpenReview, ACL Anthology, bioRxiv/medRxiv, a direct PDF link, or a page that names
its PDF in a citation_pdf_url tag (most journals and open-access sites)."""
import html
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from modelfetch import _context
from paperstore import clean

LIMIT = 50 << 20
AGENT = {"User-Agent": "LocalPaperReader/1.0"}
ARXIV_ID = r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
PDF_TAG = (rb'<meta[^>]*name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)',
           rb'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']citation_pdf_url')


def pdf_url(link):
    """The PDF address for a pasted link, and a file name for it."""
    link = link.strip()
    if re.fullmatch(ARXIV_ID, link):                                     # a bare arXiv id
        link = "https://arxiv.org/abs/" + link
    if not re.match(r"https?://", link):                                 # pasted without https://
        link = "https://" + link
    u = urllib.parse.urlparse(link)
    host = u.netloc.lower().removeprefix("www.")
    if host.endswith("arxiv.org") and (m := re.search(r"/(?:abs|pdf)/" + ARXIV_ID, u.path)):
        return f"https://arxiv.org/pdf/{m[1]}", m[1]
    if host == "openreview.net" and (i := urllib.parse.parse_qs(u.query).get("id")):
        return f"https://openreview.net/pdf?id={i[0]}", i[0]
    if host == "aclanthology.org" and not u.path.endswith(".pdf"):
        return f"https://aclanthology.org/{u.path.strip('/')}.pdf", u.path.strip("/")
    if host in ("biorxiv.org", "medrxiv.org") and "/content/" in u.path and not u.path.endswith(".pdf"):
        p = u.path.rstrip("/").removesuffix(".full")
        return f"https://www.{host}{p}.full.pdf", p.rsplit("/", 1)[-1]
    return link, urllib.parse.unquote(os.path.basename(u.path)).removesuffix(".pdf") or host


def download(link):
    """Download the paper at `link` to a temporary file and return its path. Raises RuntimeError with a message
    for the user."""
    url, name = pdf_url(link)
    for _ in range(2):                                                   # a page, then the PDF it names
        host = urllib.parse.urlparse(url).netloc
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=30,
                                        context=_context()) as r:
                data, url = r.read(LIMIT + 1), r.url
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{host} answered “{e.code} {e.reason}”.") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RuntimeError(f"couldn't reach {host}.") from e
        if len(data) > LIMIT:
            raise RuntimeError("the file is over 50 MB.")
        if data.startswith(b"%PDF-"):
            path = os.path.join(tempfile.mkdtemp(), clean(name) + ".pdf")
            with open(path, "wb") as f:
                f.write(data)
            return path
        tag = next((m for p in PDF_TAG if (m := re.search(p, data, re.I))), None)
        if not tag:
            break
        url = urllib.parse.urljoin(url, html.unescape(tag[1].decode()))
    raise RuntimeError("that link didn't lead to a PDF (it may need a login).")
