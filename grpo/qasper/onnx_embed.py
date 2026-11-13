"""The retrieval embedder (BAAI/bge-small-en-v1.5) through onnxruntime, so the GGUF laptop build needs no torch.

model.onnx is exported from the cached HF weights (runs/embedder_onnx, with the tokenizer files beside it). On all
1005 dev questions it gives the same top-16 paragraph sets as torch on CPU (embedding cosine >= 0.9999997).
"""
import numpy as np


class OnnxEmbedder:
    """bge-small through onnxruntime: SentenceTransformer.encode for this model (truncate at 512 tokens, CLS
    vector, L2-normalise) with no torch. Returns float32 numpy."""
    def __init__(self, path, threads=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(f"{path}/model.onnx", opts, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(path)

    def encode(self, texts, batch_size=64, **_):          # extra SentenceTransformer kwargs are accepted and ignored
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="np")
            h = self.sess.run(None, {k: enc[k].astype(np.int64)
                                     for k in ("input_ids", "attention_mask", "token_type_ids")})[0][:, 0]
            out[idx] = h / np.linalg.norm(h, axis=1, keepdims=True)
        return out
