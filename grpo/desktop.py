"""LocalPaperReader in its own window: the entry point of the bundled, double-click app.

Opens the window at once with a loading page, downloads the model files on first launch (modelfetch), then serves
the interface app.py builds on a local port and shows it in the window. Closing the window quits. The window uses
the web engine the system already has (pywebview), so none is shipped.

    PYTHONPATH=. python desktop.py        # from a checkout: no download if the models are in runs/gguf
"""
import json
import os
import sys
import threading
import time
import traceback

import webview

CONFIG = "configs/reader_4b_gguf.json"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { font: 16px -apple-system, "Segoe UI", sans-serif; color: #22252B; background: #FFFDF8; margin: 0;
         height: 100vh; display: flex; align-items: center; justify-content: center; }
  main { width: min(560px, 86vw); text-align: center; }
  h1 { font-size: 22px; margin: 0 0 18px; }
  #msg { white-space: pre-wrap; line-height: 1.5; }
  #bar { display: none; height: 8px; background: #E9E5DB; border-radius: 4px; margin: 18px 0 8px; overflow: hidden; }
  #fill { height: 100%; width: 0; background: #E0603A; }
  #count { display: none; font-size: 14px; opacity: .7; }
  button { display: none; margin-top: 18px; font: inherit; padding: 8px 22px; border: 0; border-radius: 6px;
           background: #E0603A; color: #fff; cursor: pointer; }
</style></head><body><main>
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
                                        prevent_thread_lock=True)      # no port given: gradio picks a free one
        window.load_url(url)
    except Exception:
        traceback.print_exc()
        say("LocalPaperReader could not start.\n\n" + traceback.format_exc()[-1200:])


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")          # offline: no version check, no telemetry
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    api = Api()
    window = webview.create_window("LocalPaperReader", html=PAGE, js_api=api, width=1100, height=900,
                                   min_size=(720, 600))
    webview.start(startup, (window, api))           # returns when the window is closed
    os._exit(0)                                     # stop the server and model threads with it


if __name__ == "__main__":
    main()
