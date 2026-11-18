"""The app's own copy of every paper opened. The folder is the list: a paper's name is its file name, and the
file's date is when it was last opened."""
import os
import re
import shutil

from modelfetch import sha256

GENERIC = {"main", "paper", "download", "document", "file", "full", "fulltext", "pdf", "article", "manuscript",
           "untitled", "final", "draft", "camera", "ready", "arxiv", "acl", "long", "short", "copy", "version"}
INCOMING = ".incoming"                              # downloads wait here until kept


def clean(name):
    """A title or link made into a file name: no characters macOS or Windows refuse, at most 120 characters."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " - ", name)
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return name[:120].rstrip(" .-") or "paper"


def junk(name):
    """A name that says nothing: an id, a number or generic words (2310.01234v2, main, paper(3), download)."""
    return len([w for w in re.findall(r"[A-Za-z]{3,}", name) if w.lower() not in GENERIC]) < 2


def keep(path, title, folder, move=False):
    """Copy the PDF at `path` into `folder` unless the same paper is already there. Returns the kept copy's path.
    A junk file name is replaced by the title. The kept copy becomes the newest. `move`: a download, moved in,
    or deleted when the paper is already kept."""
    os.makedirs(folder, exist_ok=True)
    size, digest = os.path.getsize(path), None
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if f.endswith(".pdf") and os.path.getsize(p) == size and sha256(p) == (digest := digest or sha256(path)):
            os.utime(p)
            if move:
                os.remove(path)
            return p
    stem = os.path.basename(path).removesuffix(".pdf")
    stem = clean(title if title and junk(stem) else stem)
    p, n = os.path.join(folder, stem + ".pdf"), 2
    while os.path.exists(p):
        p, n = os.path.join(folder, f"{stem} ({n}).pdf"), n + 1
    (shutil.move if move else shutil.copyfile)(path, p)
    return p


def listing(folder):
    """The kept papers, newest opened first, as [{"file", "name"}]."""
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".pdf") and not f.startswith(".")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return [{"file": f, "name": f.removesuffix(".pdf")} for f in files]


def _own(folder, file):
    """The path of `file` if it is one of the folder's papers, else None."""
    p = os.path.join(folder, os.path.basename(file or ""))
    return p if p.endswith(".pdf") and os.path.isfile(p) else None


def rename(folder, file, name):
    """Rename a kept paper. A name already taken gets " (2)". Returns the new listing."""
    if p := _own(folder, file):
        stem, n = clean(name), 2
        q = os.path.join(folder, stem + ".pdf")
        while os.path.exists(q) and not os.path.samefile(q, p):
            q, n = os.path.join(folder, f"{stem} ({n}).pdf"), n + 1
        os.rename(p, q)
    return listing(folder)


def delete(folder, file):
    """Delete a kept paper. Returns the new listing."""
    if p := _own(folder, file):
        os.remove(p)
    return listing(folder)
