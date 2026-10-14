"""Build TensorRT engines from the exported ONNX files. Run this ON the target device (Jetson Orin):
an engine is tied to the GPU architecture and TensorRT version that built it.

The ONNX graphs are already fp16 with the precision-critical ops (RMSNorm, attention softmax,
rotary angles) cast to fp32 inside the graph, so the network is built STRONGLY_TYPED: TensorRT keeps
every tensor in the dtype the ONNX graph gives it instead of choosing per layer.

Usage:
    python3 build_trt.py                  # inside the package folder: writes <name>_vision.engine, <name>_decoder.engine
    python3 build_trt.py --max-seq 2048   # longest prompt + answer the decoder accepts
Equivalent trtexec commands are in README.md.
"""
import argparse
import json
import time
from pathlib import Path

import tensorrt as trt

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE if (HERE / "config.json").exists() else HERE / "qwen3vl2b_people_v2"


def build(onnx_path, engine_path, shapes=None, workspace_gb=8):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):  # resolves the external .data file next to it
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if shapes:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in shapes.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")
    engine_path.write_bytes(engine)
    print(f"saved -> {engine_path}  ({engine.nbytes / 2**20:.0f} MiB, {time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--max-seq", type=int, default=2048, help="max prompt + generated tokens")
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--only", choices=["vision_encoder", "decoder"], default=None)
    args = ap.parse_args()
    d = Path(args.model_dir)
    m = json.load(open(d / "config.json"))
    vision, decoder = d / m["files"]["vision_encoder"], d / m["files"]["decoder"]

    if args.only in (None, "vision_encoder"):
        build(vision, vision.with_suffix(".engine"), workspace_gb=args.workspace_gb)

    if args.only in (None, "decoder"):
        hidden, kv, hd, n_ds, ms = m["hidden_size"], m["num_kv_heads"], m["head_dim"], m["num_deepstack"], args.max_seq
        prompt = len(m["input_ids"])
        # one profile: seq 1..max (prefill uses the prompt length, decode uses 1); tuned for decode
        shapes = {"inputs_embeds": ((1, 1, hidden), (1, 1, hidden), (1, ms, hidden)),
                  "position_ids": ((3, 1, 1), (3, 1, 1), (3, 1, ms)),
                  "deepstack_embeds": ((n_ds, 1, 1, hidden), (n_ds, 1, 1, hidden), (n_ds, 1, ms, hidden))}
        for i in range(m["num_layers"]):
            for kind in ("key", "value"):
                shapes[f"past_{kind}_{i}"] = ((1, kv, 0, hd), (1, kv, prompt + 16, hd), (1, kv, ms - 1, hd))
        build(decoder, decoder.with_suffix(".engine"), shapes, workspace_gb=args.workspace_gb)


if __name__ == "__main__":
    main()
