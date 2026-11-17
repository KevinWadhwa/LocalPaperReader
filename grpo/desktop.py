"""The double-click app's entry point. Opens a native window (pywebview, on the system's own web engine) with a
loading page, downloads the model on first launch (modelfetch), then shows the interface app.py serves on a local
port. Closing the window quits.

    PYTHONPATH=. python desktop.py            # from a checkout
    PYTHONPATH=. python desktop.py --browser  # no window, in the browser
"""
import base64
import json
import os
import re
import sys
import threading
import time
import traceback

import webview

CONFIG = "configs/reader_4b_gguf.json"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  /* the look of the app's start screen (readergui/styling.py) */
  body { font: 16px -apple-system, "Segoe UI", system-ui, sans-serif; color: #22252B; background: #FFFDF8; margin: 0;
         height: 100vh; display: flex; align-items: center; justify-content: center; }
  main { width: min(560px, 86vw); text-align: center; }
  img { height: 96px; }
  img.dark { display: none; }
  h1 { font-size: 20px; font-weight: 600; margin: 14px 0 10px; }
  #msg { white-space: pre-wrap; line-height: 1.5; color: #5A5E66; }
  #bar { display: none; height: 8px; background: #E9E5DB; border-radius: 4px; margin: 18px 0 8px; overflow: hidden; }
  #fill { height: 100%; width: 0; background: #1E3176; }
  #count { display: none; font-size: 13px; color: #5A5E66; }
  button { display: none; margin-top: 18px; font: inherit; font-weight: 500; padding: 10px 24px; border: 0;
           border-radius: 22px; background: #2F6B4A; color: #fff; cursor: pointer; }
  @media (prefers-color-scheme: dark) {
    body { color: #ECE9E2; background: #000; }
    img.light { display: none; }
    img.dark { display: inline; }
    #msg, #count { color: #B4B6BB; }
    #bar { background: #26282C; }
    #fill { background: #3952AC; }
    button { background: #3A7F59; }
  }
</style></head><body><main>
  <img class="light" alt="" src="__BOOK__"><img class="dark" alt="" src="__BOOK_DARK__">
  <h1>LocalPaperReader</h1>
  <div id="msg">Starting…</div>
  <div id="bar"><div id="fill"></div></div><div id="count"></div>
  <button id="go" onclick="this.style.display='none'; pywebview.api.go()"></button>
</main><script>
  const el = id => document.getElementById(id);
  function show(msg, button) {                      // a message, and a button if `button` is not empty
    el('msg').textContent = msg;
    el('go').textContent = button; el('go').style.display = button ? 'inline-block' : 'none';
    el('bar').style.display = el('count').style.display = 'none';
  }
  function progress(done, total) {
    el('bar').style.display = el('count').style.display = 'block';
    el('fill').style.width = (100 * done / total) + '%';
    el('count').textContent = (done / 1e9).toFixed(2) + ' of ' + (total / 1e9).toFixed(2) + ' GB';
  }
</script></body></html>"""


def page():
    """The loading page with the app's book, still, in its light and dark drawings."""
    out = PAGE
    for mark, name in (("__BOOK__", "01-flip-through.svg"), ("__BOOK_DARK__", "01-flip-through-dark.svg")):
        svg = re.sub(rb"animation:[^;}]*;?", b"", open(os.path.join(APP_DIR, "icons", name), "rb").read())
        out = out.replace(mark, "data:image/svg+xml;base64," + base64.b64encode(svg).decode())
    return out


class Api:
    """What the loading page can call. pywebview exposes public methods only, hence the underscore."""
    def __init__(self):
        self._go = threading.Event()

    def go(self):                                   # the page's button
        self._go.set()


def startup(window, api):
    """Runs beside the window: fetch what is missing, load the model, then switch the window to the interface."""
    def say(message, button=""):
        window.evaluate_js(f"show({json.dumps(message)}, {json.dumps(button)})")

    last = [0.0]

    def progress(done, total):                      # at most five page updates a second
        if done == total or time.time() - last[0] > 0.2:
            last[0] = time.time()
            window.evaluate_js(f"progress({done}, {total})")

    try:
        import app                                  # loads gradio (seconds): only once the window is up
        import modelfetch
        os.makedirs(app.data_dir(), exist_ok=True)
        log = open(os.path.join(app.data_dir(), "app.log"), "a", buffering=1)
        sys.stdout = sys.stderr = log               # a windowed app has no terminal (on Windows, no stdout at all)
        cfg = app.load_config(CONFIG)
        models = [cfg.get(k) for k in app.MODEL_KEYS]
        need = modelfetch.missing(cfg.get("downloads", []), models)
        gb = sum(d["bytes"] for d in need) / 1e9
        message, button = f"LocalPaperReader needs to download its language model ({gb:.1f} GB, one time).", "Download"
        while need:
            say(message, button)
            api._go.clear()
            api._go.wait()
            say("Downloading the language model…")
            try:
                modelfetch.fetch_all(need, os.path.join(app.data_dir(), "models"), progress)
            except RuntimeError as e:               # modelfetch's messages are written for the user
                message, button = str(e), "Try again"
            need = modelfetch.missing(cfg.get("downloads", []), models)
        say("Loading the model…")
        demo = app.build_demo(cfg)
        _, url, _ = demo.queue().launch(server_name="127.0.0.1", show_error=True, inbrowser=False,
                                        prevent_thread_lock=True, **app.styling.LAUNCH)    # no port: a free one
        window.load_url(url)
    except Exception:
        traceback.print_exc()
        say("LocalPaperReader could not start.\n\n" + traceback.format_exc()[-1200:])


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")          # offline: no version check, no telemetry
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "--browser" in sys.argv:                     # fallback if the window fails, and how a bundle is tested
        sys.argv.remove("--browser")                # headless; the models must already be on disk
        import app
        if sys.stdout is None:                      # Windows, windowed build
            os.makedirs(app.data_dir(), exist_ok=True)
            sys.stdout = sys.stderr = open(os.path.join(app.data_dir(), "app.log"), "a", buffering=1)
        return app.main()
    api = Api()
    window = webview.create_window("LocalPaperReader", html=page(), js_api=api, width=1100, height=900,
                                   min_size=(720, 600))
    webview.start(startup, (window, api))           # returns when the window is closed
    os._exit(0)                                     # stop the server and model threads with it


if __name__ == "__main__":
    main()
