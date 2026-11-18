<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="grpo/icons/01-flip-through-dark.svg">
    <img src="grpo/icons/01-flip-through.svg" alt="LocalPaperReader logo: an open book" width="200">
  </picture>
</p>

<h1 align="center">LocalPaperReader</h1>

<p align="center">Ask questions about a research paper and get answers quoted straight from it, or an honest "I couldn't find that".<br>
Runs entirely on your Mac. No internet, no account, nothing leaves your computer.</p>

---

## Why I built this

When I asked the usual online chatbots about a paper, they often sounded sure of themselves while being wrong. They
filled gaps with things they half-remembered from training, or made up details that weren't in the paper at all. If
you're relying on them to understand a paper, that's quietly misleading.

LocalPaperReader works the other way round. Every answer has to quote the paper word for word, and those quotes are
checked against the actual text before you see them. If the paper doesn't support an answer, it says so instead of
guessing. Each answer also comes with a confidence label (high, medium or low) and the paragraph it came from, so
you can check it yourself in a second.

## Getting started

1. Download LocalPaperReader.dmg from the [latest release](https://github.com/ada-kw1525/LocalPaperReader/releases/latest).
2. Open it and drag LocalPaperReader into Applications.
3. The first time you open it, macOS will say it "could not verify LocalPaperReader is free of malware". That's
   only because the app isn't signed with a paid Apple developer account yet. Click Done, then go to System
   Settings → Privacy & Security, scroll to the bottom and click Open Anyway next to LocalPaperReader. You only need
   to do this once. Or, if you prefer Terminal, run this and then open the app as normal:
   ```
   xattr -dr com.apple.quarantine /Applications/LocalPaperReader.app
   ```
4. On first launch it downloads the model (about 2.6 GB, once). After that it works fully offline.
5. Drop in a PDF or its link, wait a moment while it reads the paper, and ask away. Try specific questions ("Which datasets did
   they use?") or general ones ("What are the main results?", "What are the limitations?").

You'll need an Apple Silicon Mac (M1 or newer) and about 3 GB of free disk space. A Windows version is coming soon.

Something not working, or got an idea? [Open an issue](https://github.com/ada-kw1525/LocalPaperReader/issues) or
email me at [your-email-here]. I'd genuinely love to hear how it goes.

## What's going on inside

I didn't have much hands-on experience with LLMs before this, so I built it as a way to properly learn how they
work, from retrieval right through to training and squeezing a model onto a laptop. Here's how it fits together:

- **Reading the PDF.** The paper is split into paragraphs and sections with `pdfplumber`, keeping citations,
  tables and figures linked.
- **Finding the right passages (RAG).** For each question, retrieval-augmented generation picks the 16 most
  relevant paragraphs using BM25 keyword search combined with dense embeddings (`bge-small-en-v1.5`, run through
  ONNX). Only those paragraphs go to the model.
- **Fine-tuning.** The base model is Qwen3-4B, fine-tuned with supervised fine-tuning (SFT) using LoRA adapters on
  [QASPER](https://allenai.org/data/qasper), a dataset of questions about NLP papers. It learned to quote first,
  answer second, and abstain when the evidence isn't there.
- **Reinforcement learning.** It was then refined with GRPO (Group Relative Policy Optimisation), rewarded for
  correct, well-quoted answers and penalised for fabricated or mis-cited quotes.
- **A second opinion.** A separate verifier pass scores how likely the model thinks the paper doesn't contain the
  answer. That score sets the high, medium and low labels, which were calibrated on held-out data. On the test set,
  high answers were right about 90% of the time.
- **Quote checking.** Every quote must appear verbatim in the paper and point to the right paragraph, otherwise
  the answer is withheld.
- **Quantisation.** The model is compressed to 4-bit GGUF (`Q4_K_M`) and runs with llama.cpp on the Mac's GPU
  (Metal). The fine-tuned behaviour is a small LoRA file switched on and off at runtime, so one model does both
  jobs.

## Runs on your laptop

All of this is done with a model of just 4 billion parameters, small enough to run on an ordinary laptop instead
of a data centre. Everything happens on your machine, completely offline once the model is downloaded. It
typically uses about 3 GB of RAM and is quick to answer, and a general "overview" question takes a little longer
because it's answered as a few smaller questions.

A couple of honest limits: it was trained on NLP papers, so it's at its best on those (though other fields work
too). Like any model it can still be confidently wrong, which is exactly why every answer shows its quotes. A low
label means "here's a lead, check it", not "here's the answer".

## License

[MIT](LICENSE)
