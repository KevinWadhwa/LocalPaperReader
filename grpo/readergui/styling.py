"""The interface's look, in one place: change a value in the first section and restart the app.

LAUNCH goes to every gradio launch() call (app.py and desktop.py).
"""
import base64
import os

import gradio as gr

# ============================================================ what you can adjust
# ---- colours, as (light mode, dark mode)
PAGE = ("#FFFDF8", "#000000")                       # window background
TEXT = ("#22252B", "#ECE9E2")
MUTED = ("#5A5E66", "#B4B6BB")                      # confidence line, stage messages, section names
LINE = ("#E9E5DB", "#26282C")                       # the line under the top bar
BUTTON = ("#22252B", "#ECE9E2")                     # Open PDF: outline and text
QUESTION_FILL = ("#2F6B4A", "#2F6B4A")              # your question's bubble
QUESTION_TEXT = ("#FFFFFF", "#FFFFFF")
ANSWER_FILL = ("#1E3176", "#243A8A")                # the answer's hexagon
ANSWER_TEXT = ("#FFFFFF", "#FFFFFF")
VIEW_FILL = ("#CDE9FB", "rgba(135,206,250,.2)")     # the "View source" rectangle
VIEW_TEXT = ("#2A6044", "#8FCBA6")
SOURCE_FILL = ("#E2ECF7", "#222E3F")                # the opened source's bubble
SOURCE_LINE = ("#C9D8EA", "#36465C")                # the line above its references
SOURCE_REFS = ("#55606E", "#A9B4C4")
HIGHLIGHT = ("rgba(247,205,59,.5)", "rgba(247,205,59,.28)")     # the quote inside its paragraph
OVAL_FILL = ("#ECE8DD", "#1E2023")                  # "couldn't find the answer"
OVAL_TEXT = ("#3F434A", "#D5D2CB")
BOX_FILL = ("#FFFFFF", "#0E0F11")                   # the question box
BOX_LINE = ("#1E3176", "#3952AC")
PLACEHOLDER = ("#6B6F76", "#A6A8AD")                # "Ask a question!"
SEND_FILL = ("#2F6B4A", "#3A7F59")                  # the send button
SEND_ARROW = ("#FFFFFF", "#FFFFFF")
DOT = {"high": "#2F8F5B", "medium": "#E3A912", "low": "#E0603A"}    # confidence dots, both modes; "defined" and "seen" = SOURCE_FILL

# ---- fonts (Manrope and Nunito Sans ship in fonts/; the others are the system's)
UI_FONT = "-apple-system, 'Segoe UI', system-ui, sans-serif"
QUESTION_FONT = "'Avenir Next', 'Nunito Sans', 'Segoe UI', system-ui, sans-serif"
ANSWER_FONT = "Manrope, -apple-system, 'Segoe UI', system-ui, sans-serif"
SOURCE_FONT = "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif"

# ---- sizes, in px
COLUMN = 760                                        # width of the conversation and the question box
ICON = 44                                           # a stage icon
TOP_ICON = 60                                       # the book in the top bar
START_ICON = 96                                     # the book on the start screen
OPEN_BUTTON = 112                                   # width of Open PDF (also centres the top-bar book)
TEXT_SIZE = 16
ANSWER_SIZE = 16
CONFIDENCE_SIZE = 15
DOT_SIZE = 11
SOURCE_SIZE = 15.5
SMALL_SIZE = 13                                     # the "Looking in" line, references, the loaded line
VIEW_INDENT = 20                                    # how far "View source" sits in from the hexagon's tip
HEX_CAP = 28                                        # width of each pointed end of the hexagon
HEX_CAP_SHAPE = "M28 0 H24 Q15 0 12 12 L4 38 Q0 50 4 62 L12 88 Q15 100 24 100 H28 Z"   # left end, in a 28 x 100 box

# ============================================================ built from the above
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
PAIRS = {"page": PAGE, "text": TEXT, "muted": MUTED, "line": LINE, "button": BUTTON, "question": QUESTION_FILL,
         "question-text": QUESTION_TEXT, "answer": ANSWER_FILL, "answer-text": ANSWER_TEXT, "view": VIEW_FILL,
         "view-text": VIEW_TEXT, "source": SOURCE_FILL, "source-line": SOURCE_LINE, "source-refs": SOURCE_REFS,
         "highlight": HIGHLIGHT, "oval": OVAL_FILL, "oval-text": OVAL_TEXT, "box": BOX_FILL, "box-line": BOX_LINE,
         "placeholder": PLACEHOLDER, "send": SEND_FILL, "send-arrow": SEND_ARROW}


def _b64(data):
    return base64.b64encode(data).decode()


def _font_face(family, weights, file, ranges):
    data = _b64(open(os.path.join(FONT_DIR, file), "rb").read())
    return (f'@font-face {{ font-family: "{family}"; font-style: normal; font-weight: {weights}; '
            f'src: url(data:font/woff2;base64,{data}) format("woff2"); unicode-range: {ranges}; }}\n')


LATIN = ("U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, "
         "U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD")
GREEK = "U+0370-0377, U+037A-037F, U+0384-038A, U+038C, U+038E-03A1, U+03A3-03FF"
FONTS = (_font_face("Manrope", "200 800", "manrope-latin.woff2", LATIN)
         + _font_face("Manrope", "200 800", "manrope-greek.woff2", GREEK)
         + _font_face("Nunito Sans", "200 1000", "nunitosans-latin.woff2", LATIN))
CAP = _b64(f'<svg xmlns="http://www.w3.org/2000/svg" width="{HEX_CAP}" height="100" viewBox="0 0 28 100" '
           f'preserveAspectRatio="none"><path d="{HEX_CAP_SHAPE}"/></svg>'.encode())
MASK = f'url("data:image/svg+xml;base64,{CAP}") center / 100% 100% no-repeat'
VARS = ["".join(f"--lpr-{k}: {v[i]}; " for k, v in PAIRS.items()) for i in (0, 1)]

# gradio prefixes every rule, with a longer prefix when it mentions .dark: a rule that mentions .dark has one
# selector, and is more specific than the rules it must override. The colours switch through the variables, so
# only the first rules and the icon rules mention .dark. The chat styles its images, hence !important on .lpr-icon.
CSS = FONTS + f"""
:root {{ {VARS[0]} }}
.dark {{ {VARS[1]} }}

/* icons: a light and a dark image side by side, one shown */
.lpr-icon {{ display: block !important; height: {ICON}px !important; width: auto !important; max-width: none !important;
            max-height: none !important; margin: 8px 0 0 !important; border-radius: 0 !important; }}
.lpr-icon.lpr-dark {{ display: none !important; }}
.dark .lpr-icon.lpr-light {{ display: none !important; }}
.dark .lpr-icon.lpr-dark {{ display: block !important; }}
.lpr-icon.lpr-top {{ height: {TOP_ICON}px !important; margin: 0 !important; }}
.lpr-icon.lpr-start {{ height: {START_ICON}px !important; margin: 0 auto !important; }}
.lpr-icon.lpr-inline {{ margin: 0 !important; }}

/* top bar */
#lpr-bar {{ align-items: center; flex-wrap: nowrap; gap: 16px; padding: 10px 24px; border-bottom: 1px solid var(--lpr-line); }}
#lpr-open {{ flex: 0 0 {OPEN_BUTTON}px; width: {OPEN_BUTTON}px; min-width: {OPEN_BUTTON}px; min-height: 44px; padding: 0 12px;
            border: 1px solid var(--lpr-button); border-radius: 10px; background: transparent; box-shadow: none;
            color: var(--lpr-button); font-size: 15px; font-weight: 500; }}
#lpr-head {{ flex: 1 1 0; min-width: 0; }}
.lpr-head {{ display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 16px;
            min-height: {TOP_ICON}px; color: var(--lpr-text); }}
.lpr-head-text {{ min-width: 0; padding-right: {(OPEN_BUTTON + 16) // 2}px; }}
.lpr-title {{ display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; font-weight: 600;
             line-height: 1.3; }}
.lpr-sub {{ font-size: {SMALL_SIZE}px; color: var(--lpr-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.lpr-head-stage {{ display: flex; align-items: center; gap: 14px; color: var(--lpr-muted); }}
.lpr-top-slot {{ transform: translateX(-{(OPEN_BUTTON + 16) // 2}px); }}

/* Open PDF's card: the link box looks like the question box, From device like Open PDF */
#lpr-card {{ position: absolute; top: {TOP_ICON + 20}px; left: 24px; z-index: 20; flex: none; width: 300px; min-width: 0;
            gap: 10px; padding: 14px; border: 1px solid var(--lpr-line); border-radius: 10px; background: var(--lpr-page); }}
#lpr-link input, #lpr-link textarea {{ min-height: 44px; padding: 10px 20px !important; border: 1.5px solid var(--lpr-box-line) !important;
            border-radius: 28px !important; background: var(--lpr-box) !important; box-shadow: none !important;
            color: var(--lpr-text); font-family: {QUESTION_FONT}; font-size: {TEXT_SIZE}px; }}
#lpr-link input::placeholder, #lpr-link textarea::placeholder {{ color: var(--lpr-placeholder); opacity: 1; }}
#lpr-device {{ flex: none; align-self: flex-start; width: auto; min-width: 0; min-height: 32px; padding: 0 12px;
              border: 1px solid var(--lpr-button); border-radius: 10px; background: transparent; box-shadow: none;
              color: var(--lpr-button); font-size: {SMALL_SIZE}px; font-weight: 500; }}

/* Previously opened: the button looks like Open PDF, its list like Open PDF's card */
#lpr-bar {{ position: relative; }}
#lpr-recent {{ position: absolute; right: 24px; top: 50%; transform: translateY(-50%); flex: none; width: auto; min-width: 0;
              min-height: 44px; padding: 0 12px; border: 1px solid var(--lpr-button); border-radius: 10px;
              background: transparent; box-shadow: none; color: var(--lpr-button); font-size: 15px; font-weight: 500; }}
#lpr-shelf {{ position: absolute; top: {TOP_ICON + 20}px; right: 24px; z-index: 20; width: 340px; max-height: 360px;
             overflow-y: auto; padding: 6px; border: 1px solid var(--lpr-line); border-radius: 10px; background: var(--lpr-page); }}
.lpr-paper {{ display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 8px; color: var(--lpr-text);
             font-size: 14px; cursor: default; }}
.lpr-paper:hover {{ background: var(--lpr-oval); }}
.lpr-name {{ flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: text; outline: none; }}
.lpr-name[contenteditable="true"] {{ white-space: normal; border-radius: 4px; background: var(--lpr-box);
                                     box-shadow: 0 0 0 1.5px var(--lpr-box-line); }}
.lpr-del {{ visibility: hidden; flex: none; padding: 2px 8px; border: 1px solid var(--lpr-line); border-radius: 6px;
           background: transparent; color: var(--lpr-muted); font-size: {SMALL_SIZE}px; cursor: pointer; }}
.lpr-paper:hover .lpr-del {{ visibility: visible; }}
.lpr-del.lpr-sure {{ visibility: visible; border-color: {DOT["low"]}; color: {DOT["low"]}; }}
.lpr-empty {{ padding: 10px; color: var(--lpr-muted); font-size: 14px; }}

/* conversation */
#lpr-chat {{ border: 0 !important; background: transparent !important; box-shadow: none !important; min-height: 300px; }}
#lpr-chat .bubble-wrap {{ background: transparent !important; }}
#lpr-chat .prose.chatbot.md {{ opacity: 1 !important; }}
#lpr-chat .message-wrap {{ width: 100%; max-width: {COLUMN}px; margin-left: auto; margin-right: auto; }}
#lpr-chat .user {{ padding: 10px 16px !important; border: 0 !important; border-radius: 18px 18px 4px 18px !important;
                  background: var(--lpr-question) !important; box-shadow: none !important; font-family: {QUESTION_FONT}; }}
#lpr-chat .user * {{ color: var(--lpr-question-text) !important; font-family: {QUESTION_FONT}; }}
#lpr-chat .bot {{ width: 100%; padding: 0 !important; border: 0 !important; background: transparent !important;
                 box-shadow: none !important; }}
.lpr-start-box {{ display: flex; flex-direction: column; align-items: center; gap: 20px; max-width: 420px; margin: 0 auto;
                 text-align: center; color: var(--lpr-muted); }}
.lpr-stage {{ color: var(--lpr-muted); }}
.lpr-detail {{ margin-top: 4px; font-size: {SMALL_SIZE}px; }}

/* an answer */
.lpr-answer {{ display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin: 0 0 22px; }}
.lpr-conf {{ display: flex; align-items: center; gap: 9px; font-size: {CONFIDENCE_SIZE}px; color: var(--lpr-muted); }}
.lpr-dot {{ flex: 0 0 auto; display: inline-block; width: {DOT_SIZE}px; height: {DOT_SIZE}px; border-radius: 50%; }}
.lpr-dot.lpr-high {{ background: {DOT["high"]}; }}
.lpr-dot.lpr-medium {{ background: {DOT["medium"]}; }}
.lpr-dot.lpr-low {{ background: {DOT["low"]}; }}
.lpr-dot.lpr-defined, .lpr-dot.lpr-seen {{ background: var(--lpr-source); }}
.lpr-part {{ font-weight: 600; color: var(--lpr-text); }}
.lpr-hex {{ display: inline-flex; align-items: stretch; max-width: 100%; }}
.lpr-cap {{ flex: 0 0 {HEX_CAP}px; width: {HEX_CAP}px; min-width: {HEX_CAP}px; background: var(--lpr-answer);
           -webkit-mask: {MASK}; mask: {MASK}; }}
.lpr-cap.lpr-cap-r {{ transform: scaleX(-1); }}
.lpr-hex-text {{ padding: 11px 12px; background: var(--lpr-answer); color: var(--lpr-answer-text);
                box-shadow: -1px 0 0 var(--lpr-answer), 1px 0 0 var(--lpr-answer);
                font-family: {ANSWER_FONT}; font-size: {ANSWER_SIZE}px; font-weight: 600; line-height: 1.4; }}
.lpr-src {{ max-width: calc(100% - {VIEW_INDENT}px); margin-left: {VIEW_INDENT}px; }}
.lpr-src summary {{ display: inline-flex; flex-wrap: wrap; align-items: center; gap: 10px; list-style: none; cursor: pointer;
                   font-size: 14px; color: var(--lpr-muted); }}
.lpr-src summary::-webkit-details-marker {{ display: none; }}
.lpr-view {{ display: inline-flex; align-items: center; gap: 9px; min-height: 32px; box-sizing: border-box; padding: 4px 12px;
            border-radius: 3px; background: var(--lpr-view); color: var(--lpr-view-text); font-weight: 600; }}
.lpr-view::before {{ content: ""; width: 6px; height: 6px; border-right: 2px solid currentColor;
                    border-bottom: 2px solid currentColor; transform: rotate(-45deg); transition: transform .15s; }}
.lpr-src[open] .lpr-view::before {{ transform: rotate(45deg); }}
.lpr-src:not([open]) .lpr-para {{ display: none; }}
.lpr-para {{ margin-top: 8px; padding: 14px 18px; border-radius: 4px 18px 18px 18px; background: var(--lpr-source);
            color: var(--lpr-text); font-family: {SOURCE_FONT}; font-size: {SOURCE_SIZE}px; line-height: 1.6; }}
.lpr-para-sec {{ margin-bottom: 6px; color: var(--lpr-source-refs); font-family: {UI_FONT}; font-size: {SMALL_SIZE}px; font-weight: 600; }}
.lpr-para mark {{ padding: 1px 2px; border-radius: 3px; background: var(--lpr-highlight); color: inherit; }}
.lpr-refs {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--lpr-source-line); color: var(--lpr-source-refs);
            font-family: {UI_FONT}; font-size: {SMALL_SIZE}px; }}
.lpr-oval {{ display: inline-block; padding: 10px 26px; border-radius: 999px; background: var(--lpr-oval);
            color: var(--lpr-oval-text); font-family: {ANSWER_FONT}; }}
.lpr-missing {{ font-size: 14px; color: var(--lpr-muted); }}

/* question box */
#lpr-ask {{ width: 100%; max-width: {COLUMN}px; margin: 0 auto 24px; padding: 0 20px; box-sizing: border-box; }}
#lpr-ask .input-container {{ align-items: center; gap: 8px; padding: 5px 5px 5px 20px; border: 1.5px solid var(--lpr-box-line);
                            border-radius: 28px; background: var(--lpr-box); }}
#lpr-ask textarea {{ min-height: 44px; padding: 10px 0 !important; border: 0 !important; background: transparent !important;
                    box-shadow: none !important; color: var(--lpr-text); font-family: {QUESTION_FONT}; font-size: {TEXT_SIZE}px; }}
#lpr-ask textarea::placeholder {{ color: var(--lpr-placeholder); opacity: 1; }}
#lpr-ask .submit-button {{ flex: 0 0 44px; width: 44px; min-width: 44px; height: 44px; border: 0; border-radius: 50%;
                          background: var(--lpr-send) !important; color: var(--lpr-send-arrow) !important; }}

@media (prefers-reduced-motion: reduce) {{
  .lpr-icon.lpr-anim.lpr-light {{ height: 0 !important; margin: 0 !important; visibility: hidden; }}
  .lpr-icon.lpr-anim.lpr-dark {{ height: 0 !important; margin: 0 !important; visibility: hidden; }}
}}
"""

_FONT = [f.strip().strip("'") for f in UI_FONT.split(",")]
THEME = gr.themes.Base(font=_FONT, font_mono=["ui-monospace", "Menlo", "Consolas", "monospace"]).set(
    body_background_fill=PAGE[0], body_background_fill_dark=PAGE[1],
    background_fill_primary=PAGE[0], background_fill_primary_dark=PAGE[1],
    background_fill_secondary=PAGE[0], background_fill_secondary_dark=PAGE[1],
    body_text_color=TEXT[0], body_text_color_dark=TEXT[1],
    body_text_color_subdued=MUTED[0], body_text_color_subdued_dark=MUTED[1],
    border_color_primary=LINE[0], border_color_primary_dark=LINE[1],
    block_background_fill=PAGE[0], block_background_fill_dark=PAGE[1],
    block_border_width="0px", block_shadow="none",
    input_background_fill=BOX_FILL[0], input_background_fill_dark=BOX_FILL[1],
    input_placeholder_color=PLACEHOLDER[0], input_placeholder_color_dark=PLACEHOLDER[1],
    chatbot_text_size=f"{TEXT_SIZE}px")
LAUNCH = {"theme": THEME, "css": CSS, "footer_links": []}
