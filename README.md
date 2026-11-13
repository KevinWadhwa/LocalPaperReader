# LocalPaperReader

## Asking questions about a PDF

`app.py` answers questions about one paper by quoting it verbatim, or abstains with
"I couldn't find the answer in the retrieved sections." Each question is answered on its own; the chat history is
shown but not sent to the model. Run everything from `grpo/` with the `localpaperreader` env active.

### 1. Start GROBID (PDF parser)

```
cd /scratch_hive/kw1525/grobid-0.8.1
nohup ./gradlew run > /tmp/grobid_$USER.log 2>&1 &
until [ "$(curl -s localhost:8070/api/isalive)" = "true" ]; do sleep 5; done; echo "GROBID up"
```

It takes about a minute; the Gradle progress bar sitting at ~75% "EXECUTING" is normal. The app expects GROBID on
the same machine (`grobid_url` in the config).

### 2. Start the app on a GPU

```
cd /scratch_hive/kw1525/LocalPaperReader/grpo
nvidia-smi --query-gpu=index,memory.free --format=csv     # pick a GPU with ~40000 MiB free
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1            # the index you picked
PYTHONPATH=. python app.py --config configs/reader_4b.json
```

(`configs/reader_1.7b.json` runs the smaller pair for the laptop path below; it is not tier-calibrated.)

The models load once, at startup. The app listens on `127.0.0.1:7860` only, so it is reachable from that machine
and nowhere else.

### 3. Use it

From a second terminal on the same machine, without a browser:

```
NO_PROXY=127.0.0.1,localhost python - <<'EOF'
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7860", verbose=False)
print(c.predict(handle_file("data/pdf_test/paper.pdf"), api_name="/upload")[0])
chat, _, debug = c.predict("which datasets did they experiment with?", [], api_name="/ask")
m = chat[-1]["content"]
print(m if isinstance(m, str) else "\n".join(b.get("text", "") for b in m))
print(debug["reason"], debug["abstain_score"], debug["raw_output"])
EOF
```

In a browser: forward the port from the machine the browser runs on, e.g.
`ssh -N -L 7860:127.0.0.1:7860 <user>@<GPU node>`, then open http://localhost:7860. Upload a PDF, wait for the
status line ("parsed N paragraphs, …"), and ask. A shown answer carries a confidence label (high / medium / low).
Tick **debug** to see the confidence, the abstain score, the tier cutpoints, why an answer was abstained, the raw
model output and the retrieved context.

### 4. Configs and confidence tiers

| field | meaning |
|---|---|
| `sft_path` | merged SFT model; with the adapter off it gives the verifier's abstain score |
| `adapter_path` | GRPO LoRA adapter; with it on, the generator |
| `tiers` | confidence cutpoints on `s = -log P(abstain)`: `{high, medium, low}`. A rule-passing answer is shown labelled **high** (`s >= high`), **medium** (`>= medium`) or **low** (`>= low`); below `low` it is abstained |
| `k` | paragraphs retrieved per question (16, as in training) |
| `dtype` | `fp32`: must match the dtype the abstain scores were calibrated in |
| `device` | optional, default `auto`: CUDA, else Apple Silicon (`mps`), else `cpu` |
| `grobid_url` | GROBID server |

The default model is the 4B pair: `--config configs/reader_4b.json`. Its cutpoints
(`{high: 3.477, medium: 2.598, low: 1.575}`) were picked on QASPER dev and validated on test — on test the shown
tiers are HIGH ≈ 90% correct, MEDIUM ≈ 60%, LOW ≈ 44% (F1 ≥ 0.6), with ≈ 29% of rule-passing answers abstained.

**`configs/reader_1.7b.json` has placeholder cutpoints** (`tiers_placeholder: true`, `low: -1e9`): with them every
rule-passing answer is shown as **low**, nothing is abstained. The 1.7B pair has not been tier-calibrated; use the
4B config for real behaviour, or recalibrate with `qasper/verifier.py` on 1.7B dev features.

### What happens to a question

1. Retrieve 16 paragraphs (BM25 + bge-small, fused), build the prompt.
2. Abstain score: `s = -log P(abstain answer | prompt)` from the frozen SFT (adapter off).
3. Generate with the GRPO adapter on (greedy, up to 384 new tokens).
4. Abstain if the output doesn't parse (`format`), the model abstained (`generator abstained`), the quotes break the
   format rules (`format`), a quote isn't in the paper (`fabricated`) or is cited to the wrong paragraph
   (`miscited`), or the score is below the `low` cutpoint (`verifier`). All abstentions show the same message; the
   debug panel shows which reason.
5. Otherwise show the answer with a confidence label — **high**, **medium** or **low** by where `s` falls among the
   tier cutpoints — followed by each quote with its section, the paragraph with the quote highlighted, and the
   references cited. The label is honest about reliability: a **low** answer is a lead to verify against its quote,
   not a confident answer.

PDFs are parsed into QASPER-style text: resolved citations become `BIBREFn`, table and figure references `TABREFn` /
`FIGREFn`, displayed equations `DISPLAYFORMn`. The model and the quote check only ever see that text; placeholders
are turned back into the paper's own text for display.

### Checks and known limits

- `parsescripts/check_parse.py`: GROBID parse vs QASPER's JSON for 3 dev papers. Captions, paragraph counts and
  citation linking agree; whole-paragraph matching fails (52–70% vs an 80% target) because GROBID often merges
  paragraphs that QASPER keeps separate, although 84–97% of QASPER's paragraph text is found inside GROBID's. GROBID
  also drops non-Latin script and run-in sub-headings ("Datasets."), and leaves some citations unlinked; those stay
  as raw text.
- `readerchecks/check_reader.py` (`sbatch slurm/check_reader.sh`): the reader on 20 QASPER dev questions reproduces
  the stored `evaluate.py` run: identical prompts and raw outputs, same rule outcomes, decisions equal to the τ rule.
  Its abstain scores differ from the stored verifier features by up to 4.1e-3 (tolerance 5e-3), because the features
  were scored in batches of 4 and the reader scores one prompt at a time, and TF32 matrix multiplications round
  differently at different batch shapes (rebuilding the original batches gives 0.00). TF32 stays on, as in
  calibration. The verifier threshold itself is estimated from about 800 dev answers, so its own uncertainty is far
  larger than this numerical noise. Batch-shape effects change fewer than 1% of decisions, only those within about
  5e-3 of τ.

### Verifier and generator investigation (concluded Nov 2026)

A series of one-off experiments asked whether the shown-answer risk could be lowered further. They are **concluded
and their scripts have been removed** (`selfconsistency_pilot.py`, `selfconsistency_features.py`,
`citation_margin_pilot.py`, `combiner_experiment.py`, `family_probe.py`, and their slurm wrappers); their output
`.npz`/`.jsonl` under `runs/` are kept as the evidence. Findings, in short:

- **The residual risk is confident, quote-supported *wrong* answers on retrievable questions**, not abstention
  errors. Retrieval recall@16 is ~90%, and ~72% of shown-wrong answers are on genuinely answerable questions where
  the generator read the right (or a plausible wrong) paragraph and still answered wrong.
- **No verifier feature separates these.** Tried: the frozen-SFT abstain score (baseline), a prompt-probe, an
  answer-end hidden-state probe, self-consistency (agreement across k sampled answers), and cross-encoder
  question↔citation relevance. None beats the baseline under the win rule (risk@30 + twin guard); self-consistency
  and the answer probe help at low coverage but worsen the twin guard. All four measure "looks right," and the
  failures look right.
- **Combiner changes don't help:** gradient-boosting over the features is worse than the linear combiner; including
  the 0.2–0.6 F1 "middle" band gives only a tiny low-coverage gain. The hard floor (wrong answers no signal catches)
  is ~9%.
- **The ceiling is the generator, not the verifier, and not the model family.** SFT/GRPO raise *willingness to
  answer* and cut format errors but leave answer correctness flat (`f1|ans` ≈ 0.52). Few-shot base `f1|ans` is
  ~0.50–0.58 across Qwen3-4B/8B, Qwen2.5-7B, Gemma-2-9B and Mistral-7B — a QASPER + quote-then-answer-format ceiling,
  not something a bigger or different ~7–9B model fixes.
- **`verifier.py` keeps an optional `--agreement` path** (a 4th stacked challenger) from this work. It is off by
  default and the default run reproduces the original three-challenger result exactly; it needs a
  `*_agreement.npz`, which only the removed `selfconsistency_features.py` produced.

The untried lever is a grounding reward in GRPO (gold-cited-paragraph reward at train time); a diagnostic showed it
could address at most ~48% of confident-wrong answers (the "cited a non-gold paragraph" half), so it is a partial
fix at best.

## Running the web page on a laptop (no HPC)

Everything runs on the laptop: GROBID, the models and the page. Works on Apple Silicon (the model runs on the Mac's
GPU through MPS) and on Windows or Linux without an NVIDIA GPU (the model runs on the CPU; integrated graphics
aren't used). Needs about 16 GB of RAM (the 1.7B model in fp32 takes ~7 GB), ~10 GB of disk, Python 3.12 and
Docker Desktop. Expect each answer to take much longer than on the A100.

### 1. Copy the files from the HPC (once)

In VS Code, connected to the HPC: in the Explorer, right-click each item below, choose **Download…**, and save it
into one folder on the laptop (e.g. `LocalPaperReader-grpo/`), keeping the same paths:

| from `/scratch_hive/kw1525/LocalPaperReader/grpo/` | size |
|---|---|
| `qasper/` | <1 MB |
| `app.py` | |
| `configs/` | |
| `runs/sft_v2_1.7b/merged/` | 6.5 GB |
| `runs/grpo_v2_1.7b/final/` | 67 MB |
| `data/pdf_test/paper.pdf` (optional, to test) | 0.7 MB |

The `runs/...` folders must end up at `runs/sft_v2_1.7b/merged` and `runs/grpo_v2_1.7b/final` inside your folder,
because the config refers to them by those paths.

### 2. Python environment

Mac (Terminal), inside your folder:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1 transformers==5.19.0 peft==0.21.2 sentence-transformers==6.1.0 rank-bm25==0.2.2 gradio==6.31.0 numpy pdfplumber==0.11.10
```

Windows (PowerShell), inside your folder:

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.14.1 transformers==5.19.0 peft==0.21.2 sentence-transformers==6.1.0 rank-bm25==0.2.2 gradio==6.31.0 numpy
```

These are the versions on the HPC. The retrieval embedder (`BAAI/bge-small-en-v1.5`, ~130 MB) downloads by itself
on the first run, so don't set `HF_HUB_OFFLINE` on the laptop.

### 3. Start GROBID

With Docker Desktop running:

```
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.1
```

Leave it running. It's ready when http://localhost:8070/api/isalive shows `true` in a browser (about a minute).
This is GROBID's lightweight image, the closest to the HPC install. If it won't start on Apple Silicon, add
`--platform linux/amd64` after `run` (it then runs emulated, more slowly).

### 4. Start the app

Mac, in a second terminal, inside your folder:

```
source .venv/bin/activate
PYTHONPATH=. python app.py --config configs/reader_1.7b.json
```

Windows (PowerShell), inside your folder:

```
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python app.py --config configs\reader_1.7b.json
```

It prints `models on mps` (Mac) or `models on cpu` (Windows), then `Running on local URL: http://127.0.0.1:7860`.
Open http://localhost:7860 in your browser, upload a PDF and ask. Stop it with Ctrl+C.

### Caveats

- Not yet run on a Mac or a Windows laptop. The CPU path was tested on the HPC with a tiny model; MPS is untested.
- The tier cutpoints were calibrated on CUDA. MPS and CPU compute slightly differently (the same kind of effect as
  the batch-shape noise above), so a laptop can land an answer in a neighbouring tier (or on the other side of the
  abstain cutpoint) when its score is very close to a cutpoint. The size of that difference hasn't been measured off
  CUDA.
- The laptop path uses `configs/reader_1.7b.json` (the smaller pair, for RAM), whose cutpoints are placeholders
  (every rule-passing answer shows as **low**). The calibrated tiers are the 4B ones in `configs/reader_4b.json`;
  the 1.7B pair has not been tier-calibrated.
