"""First-launch download of the model files too big for the installer, listed under "downloads" in the config:
address, size and sha256 for each file, which may come in parts. A download resumes where it stopped and is checked
against its sha256 before it is moved into place.
"""
import hashlib
import os
import shutil
import ssl
import urllib.error
import urllib.request

CHUNK = 1 << 20
SPARE = 64 << 20                            # free space to leave on the disk


def missing(downloads, paths):
    """The download entries whose file is at none of `paths` (the model paths the app is about to load)."""
    have = {os.path.basename(p) for p in paths if p and os.path.exists(p)}
    return [d for d in downloads if d["file"] not in have]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def _context():
    """TLS settings. A bundled app has no system certificate folder, so use certifi's list when it is there."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(entry, folder, on_progress=None):
    """Download one entry into `folder` and return its path; on_progress(bytes_done, bytes_total) is called about
    once per MB. Raises RuntimeError with a message for the user: too little disk space, a network failure (the
    partial file is kept, so the next call resumes it) or a wrong checksum (the partial file is deleted)."""
    os.makedirs(folder, exist_ok=True)
    final, part = os.path.join(folder, entry["file"]), os.path.join(folder, entry["file"] + ".part")
    total = entry["bytes"]
    done = os.path.getsize(part) if os.path.exists(part) else 0
    if done > total:                        # not a partial copy of this file: start again
        os.remove(part)
        done = 0
    free = shutil.disk_usage(folder).free
    if free < total - done + SPARE:
        raise RuntimeError(f"Not enough disk space for {entry['file']}: {(total - done) / 1e9:.1f} GB needed, "
                           f"{free / 1e9:.1f} GB free.")
    end = 0                                 # where the current part ends in the joined file
    with open(part, "ab") as f:
        for p in entry["parts"]:
            start, end = end, end + p["bytes"]
            if done >= end:                 # this part is already on disk
                continue
            if not p["url"]:
                raise RuntimeError(f"This build has no download address for {entry['file']}.")
            headers = {"Range": f"bytes={done - start}-"} if done > start else {}
            try:
                with urllib.request.urlopen(urllib.request.Request(p["url"], headers=headers), timeout=60,
                                            context=_context()) as r:
                    if done > start and r.status != 206:    # the server sent the whole part: restart this part
                        f.flush()
                        f.truncate(start)
                        done = start
                    while done < end and (b := r.read(min(CHUNK, end - done))):
                        f.write(b)
                        done += len(b)
                        if on_progress:
                            on_progress(done, total)
            except (urllib.error.URLError, OSError, ValueError) as e:
                raise RuntimeError(f"The download was interrupted ({e}). It will resume next time.") from e
            if done != end:
                raise RuntimeError("The download was interrupted. It will resume next time.")
    if sha256(part) != entry["sha256"]:
        os.remove(part)
        raise RuntimeError(f"{entry['file']} did not download correctly (checksum mismatch). Please try again.")
    os.replace(part, final)
    return final


def fetch_all(entries, folder, on_progress=None):
    """fetch() each entry; on_progress(bytes_done, bytes_total) counts across all of them."""
    total, before = sum(d["bytes"] for d in entries), 0
    for d in entries:
        fetch(d, folder, on_progress and (lambda done, _, before=before: on_progress(before + done, total)))
        before += d["bytes"]
