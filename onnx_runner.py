"""Inference for the exported task-1 model: image -> "Person present / Person count".

Needs only numpy, Pillow, `tokenizers` and a backend (no torch / transformers):
    --backend ort   ONNX Runtime on the .onnx files (CUDA EP if available), reference / x86 check
    --backend trt   TensorRT engines built from the .onnx files (see README.md), e.g. on Jetson Orin

Pipeline (mirrors common.lora.generate with greedy decoding):
    1. resize the frame to the export size (bicubic), scale to [0,1], normalise with mean/std 0.5
    2. vision_encoder -> image_embeds [N, 2048] + deepstack_embeds [3, N, 2048]
    3. prompt embeddings = embed_tokens[input_ids], image-token rows replaced by image_embeds;
       deepstack_embeds scattered to the image-token rows (zeros elsewhere)
    4. decoder prefill (empty KV cache), then one token per step until an EOS token

Usage (from the repo root):
    .venv/bin/python task1_people_count/deploy/onnx_runner.py data/frames/incabin_0.5s/frame_000001.png
    python3 onnx_runner.py --backend trt frame.png          # inside the package folder, e.g. on the Orin
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
# inside the exported package the runner sits next to config.json; in the repo the package is a subfolder
DEFAULT_DIR = HERE if (HERE / "config.json").exists() else HERE / "qwen3vl2b_people_v2"


def load_config(model_dir):
    return json.load(open(Path(model_dir) / "config.json"))


def mrope_positions(input_ids, image_token_id, grid_h, grid_w, merge):
    """3-D (t, h, w) M-RoPE positions for one prompt with image blocks; same as
    Qwen3VLModel.get_rope_index. Returns positions [3, L] and the decode-time offset (rope delta)."""
    ids = np.asarray(input_ids)
    chunks, cur, i, n = [], 0, 0, len(ids)
    while i < n:
        is_img = ids[i] == image_token_id
        j = i
        while j < n and (ids[j] == image_token_id) == is_img:
            j += 1
        if is_img:
            h, w = grid_h // merge, grid_w // merge
            assert j - i == h * w, f"image block has {j - i} tokens, expected {h * w}"
            t = np.full(h * w, cur)
            chunks.append(np.stack([t, np.repeat(np.arange(h), w) + cur, np.tile(np.arange(w), h) + cur]))
            cur += max(h, w)
        else:
            r = np.arange(j - i) + cur
            chunks.append(np.stack([r, r, r]))
            cur += j - i
        i = j
    pos = np.concatenate(chunks, axis=1).astype(np.int64)
    return pos, int(pos.max()) + 1 - n


class OrtBackend:
    def __init__(self, onnx_dir, cfg):
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):  # CUDA/cuDNN from the nvidia-* pip wheels (x86 venv)
            ort.preload_dlls()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.vision = ort.InferenceSession(str(onnx_dir / cfg["files"]["vision_encoder"]), providers=providers)
        self.decoder = ort.InferenceSession(str(onnx_dir / cfg["files"]["decoder"]), providers=providers)
        self.dtype = np.float16 if self.vision.get_inputs()[0].type == "tensor(float16)" else np.float32

    def run_vision(self, image):
        return self.vision.run(None, {"image": image})

    def run_decoder(self, feeds):
        return self.decoder.run(None, feeds)


class TrtBackend:
    """TensorRT engines (<name>_vision.engine, <name>_decoder.engine) with numpy in/out over cuda-python.
    Simple and readable rather than fast: a production loop would keep the KV cache on the device."""

    def __init__(self, onnx_dir, cfg):
        import tensorrt as trt
        from cuda import cudart
        self.trt, self.cudart = trt, cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engines, self.contexts = {}, {}
        for name in ("vision_encoder", "decoder"):
            engine_file = Path(cfg["files"][name]).with_suffix(".engine")
            eng = runtime.deserialize_cuda_engine((onnx_dir / engine_file).read_bytes())
            self.engines[name], self.contexts[name] = eng, eng.create_execution_context()
        eng = self.engines["vision_encoder"]
        self.dtype = trt.nptype(eng.get_tensor_dtype("image"))
        _, self.stream = cudart.cudaStreamCreate()

    def _run(self, name, feeds):
        trt, cudart = self.trt, self.cudart
        eng, ctx = self.engines[name], self.contexts[name]
        bufs = []
        for k, v in feeds.items():
            v = np.ascontiguousarray(v.astype(trt.nptype(eng.get_tensor_dtype(k)), copy=False))
            ctx.set_input_shape(k, v.shape)
            _, ptr = cudart.cudaMalloc(max(v.nbytes, 1))
            cudart.cudaMemcpyAsync(ptr, v.ctypes.data, v.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            ctx.set_tensor_address(k, ptr)
            bufs.append(ptr)
        outs = []
        for i in range(eng.num_io_tensors):
            k = eng.get_tensor_name(i)
            if eng.get_tensor_mode(k) != trt.TensorIOMode.OUTPUT:
                continue
            arr = np.empty(tuple(ctx.get_tensor_shape(k)), dtype=trt.nptype(eng.get_tensor_dtype(k)))
            _, ptr = cudart.cudaMalloc(max(arr.nbytes, 1))
            ctx.set_tensor_address(k, ptr)
            bufs.append(ptr)
            outs.append((arr, ptr))
        ctx.execute_async_v3(self.stream)
        for arr, ptr in outs:
            cudart.cudaMemcpyAsync(arr.ctypes.data, ptr, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        for ptr in bufs:
            cudart.cudaFree(ptr)
        return [arr for arr, _ in outs]

    def run_vision(self, image):
        return self._run("vision_encoder", {"image": image})

    def run_decoder(self, feeds):
        return self._run("decoder", feeds)


class PeopleCounter:
    def __init__(self, onnx_dir=DEFAULT_DIR, backend="ort"):
        onnx_dir = Path(onnx_dir)
        self.meta = m = load_config(onnx_dir)
        self.backend = (OrtBackend if backend == "ort" else TrtBackend)(onnx_dir, m)
        self.dtype = self.backend.dtype
        self.tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        self.embed = np.load(onnx_dir / "embed_tokens.npy", mmap_mode="r")
        self.input_ids = np.array(m["input_ids"], dtype=np.int64)
        self.image_mask = self.input_ids == m["image_token_id"]
        self.positions, self.rope_delta = mrope_positions(self.input_ids, m["image_token_id"], m["grid_h"],
                                                          m["grid_w"], m["spatial_merge_size"])
        self.prompt_embeds = np.asarray(self.embed[self.input_ids], dtype=self.dtype)
        self.eos = set(m["eos_token_ids"])

    def preprocess(self, image):
        m = self.meta
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        img = img.convert("RGB").resize((m["image_width"], m["image_height"]), Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.array(m["image_mean"], np.float32)) / np.array(m["image_std"], np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(self.dtype)

    def _empty_cache(self):
        m = self.meta
        shape = (1, m["num_kv_heads"], 0, m["head_dim"])
        return {f"past_{kind}_{i}": np.zeros(shape, self.dtype) for i in range(m["num_layers"]) for kind in ("key", "value")}

    def generate(self, image, max_new_tokens=512):
        m = self.meta
        image_embeds, deepstack = self.backend.run_vision(self.preprocess(image))

        embeds = self.prompt_embeds.copy()
        embeds[self.image_mask] = image_embeds
        ds = np.zeros((m["num_deepstack"], 1, len(self.input_ids), m["hidden_size"]), self.dtype)
        ds[:, 0, self.image_mask] = deepstack
        feeds = {"inputs_embeds": embeds[None], "position_ids": self.positions[:, None, :],
                 "deepstack_embeds": ds, **self._empty_cache()}
        ds_step = np.zeros((m["num_deepstack"], 1, 1, m["hidden_size"]), self.dtype)

        tokens, seq_len = [], len(self.input_ids)
        for _ in range(max_new_tokens):
            logits, *present = self.backend.run_decoder(feeds)
            tok = int(np.argmax(logits[0]))
            if tok in self.eos:
                break
            tokens.append(tok)
            pos = np.full((3, 1, 1), seq_len + self.rope_delta, dtype=np.int64)
            feeds = {"inputs_embeds": np.asarray(self.embed[[tok]], dtype=self.dtype)[None], "position_ids": pos,
                     "deepstack_embeds": ds_step}
            feeds.update({f"past_{kind}_{i}": present[2 * i + j]
                          for i in range(m["num_layers"]) for j, kind in enumerate(("key", "value"))})
            seq_len += 1
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def parse_output(text):
    """Same rules as task1_people_count.prompts.parse_output (kept here so the runner is standalone)."""
    import re
    present = re.search(r"person present\s*:\s*(yes|no)", text, re.I)
    count = re.search(r"person count\s*:\s*(\d+)", text, re.I)
    return {"person_present": present.group(1).lower() == "yes" if present else None,
            "person_count": int(count.group(1)) if count else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--backend", default="ort", choices=["ort", "trt"])
    args = ap.parse_args()
    counter = PeopleCounter(args.model_dir, args.backend)
    for path in args.images:
        t0 = time.time()
        raw = counter.generate(path)
        print(json.dumps({"image": path, "raw": raw, **parse_output(raw), "seconds": round(time.time() - t0, 3)}))


if __name__ == "__main__":
    sys.exit(main())
