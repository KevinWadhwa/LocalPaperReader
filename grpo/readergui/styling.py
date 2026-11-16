"""The interface's look, in one place: change a value here and restart the app.

LAUNCH goes to every gradio launch() call (app.py and desktop.py).
"""
ICON_PX = 44                                        # height of a stage icon on screen

# the chat gives its images a fixed width, hence !important; body.dark is gradio's dark mode
CSS = f"""
.lpr-icon {{ display: block !important; height: {ICON_PX}px !important; width: auto !important;
            max-width: none !important; margin: 6px 0 0 !important; border-radius: 0 !important; }}
.lpr-icon.lpr-dark, .dark .lpr-icon.lpr-light {{ display: none !important; }}
.dark .lpr-icon.lpr-dark {{ display: block !important; }}
.lpr-detail {{ font-size: 0.85em; opacity: 0.75; }}
@media (prefers-reduced-motion: reduce) {{ .lpr-icon {{ display: none !important; }} }}
"""
LAUNCH = {"css": CSS}
