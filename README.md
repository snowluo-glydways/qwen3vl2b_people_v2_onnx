# qwen3vl2b_people_v2: in-cabin people count model (ONNX, for TensorRT / Jetson Orin)

Task 1 v2 fine-tuned model: **Qwen3-VL-2B-Instruct + LoRA `lora_Qwen3-VL-2B_v2`**.
The LoRA is merged into the weights, so no adapter is needed at deployment time.

- **Input**: one RGB frame from the in-cabin fisheye camera (1280x720, resized internally to 1280x704).
- **Output**: two lines of text, `Person present: Yes/No` and `Person count: N`.

Precision fp16, ONNX opset 17. Exported 2026-10-13.

## Getting the model weights

This git repo holds the code, the two ONNX graphs, the config, the tokenizer and the eval results.
The three large weight files (4.5 GB in total) are too big for git and are stored on Google Drive:

- **Google Drive**: `My Drive/people_count/qwen3vl2b_people_v2/` of snow.luo@glydways.com
  (https://drive.google.com/drive/folders/1twJejRDU5PuaPKuzkv7PXLFWZweU9Aph, ask the owner for access).
  The Drive folder is the complete package (code + weights); this repo is a copy of its small files.

Download these three files into the repo root, next to the `.onnx` files, **without renaming them**:

| File | Size | MD5 |
|---|---|---|
| `qwen3vl2b_people_v2_vision.onnx.data` | 0.8 GB | `c63867cce35a682a82e4522df09b1536` |
| `qwen3vl2b_people_v2_decoder.onnx.data` | 3.4 GB | `b95181f9aaf6444a3a3a5e63c5b47364` |
| `embed_tokens.npy` | 0.6 GB | `02ccafa9639868f2ef69843565e404c1` |

Download in the browser, or on a headless machine with [rclone](https://rclone.org/drive/) (after `rclone config` for a Drive remote named `gdrive`):

```bash
rclone copy gdrive:people_count/qwen3vl2b_people_v2 . \
  --include "*.onnx.data" --include "embed_tokens.npy" --progress
md5sum *.onnx.data embed_tokens.npy   # compare with the table above
```

## Quick start

All commands run in the repo root (the folder that contains `config.json`), after the weights are downloaded.

**x86 + NVIDIA GPU, or CPU (ONNX Runtime):** for checking the model, not for deployment.

```bash
pip install numpy pillow tokenizers onnxruntime-gpu   # or `onnxruntime` for CPU only (slow)
python3 onnx_runner.py --backend ort frame1.png frame2.png
```

**Jetson Orin (TensorRT):** the deployment target. See [Running on Jetson Orin](#running-on-jetson-orin) for requirements.

```bash
pip3 install numpy pillow tokenizers cuda-python   # tensorrt ships with JetPack 6
python3 build_trt.py                                # once per device: writes *_vision.engine and *_decoder.engine
python3 onnx_runner.py --backend trt frame.png
```

Each image prints one JSON line:

```json
{"image": "frame.png", "raw": "Person present: Yes\nPerson count: 2", "person_present": true, "person_count": 2, "seconds": 0.93}
```

| Option | Default | Meaning |
|---|---|---|
| `images` | (required) | One or more image paths (any format Pillow reads; RGB in-cabin frame, 1280x720) |
| `--backend` | `ort` | `ort` = ONNX Runtime on the .onnx files (CUDA if available, else CPU); `trt` = TensorRT engines from `build_trt.py` |
| `--model-dir` | folder of the script | Folder with `config.json`, the .onnx/.data files, `embed_tokens.npy` and `tokenizer.json` |

`build_trt.py` options: `--max-seq` (default 2048, longest prompt + answer), `--workspace-gb` (default 8),
`--only vision_encoder|decoder` (build one engine only).

## Folder contents

| File | Size | Where | Description |
|---|---|---|---|
| `qwen3vl2b_people_v2_vision.onnx` | 0.2 MB | git | Vision encoder (graph) |
| `qwen3vl2b_people_v2_vision.onnx.data` | 0.8 GB | Drive | Vision encoder weights (ONNX external data; must stay in the same folder as the .onnx, do not rename) |
| `qwen3vl2b_people_v2_decoder.onnx` | 0.8 MB | git | Language-model decoder + lm_head with KV cache (graph) |
| `qwen3vl2b_people_v2_decoder.onnx.data` | 3.4 GB | Drive | Decoder weights (same rules as above) |
| `embed_tokens.npy` | 0.6 GB | Drive | Token embedding table [151936, 2048] fp16, looked up on the host |
| `config.json` | 11 KB | git | Model file names, tokenised prompt (1072 ids, 880 of them image tokens), input size, mean/std, EOS ids, model dimensions |
| `tokenizer.json` | 11 MB | git | Decodes generated token ids to text |
| `onnx_runner.py` | | git | Standalone inference script. Needs only numpy, Pillow and tokenizers. Backend: ONNX Runtime (`ort`) or TensorRT (`trt`) |
| `build_trt.py` | | git | Builds TensorRT engines from the .onnx files. **Must be run on the Orin** |
| `eval/` | | git | Test-set comparison with the original model (see below) |
| `*.engine` | ~4.1 GB | built locally | TensorRT engines written by `build_trt.py`; not shared, they only work on the device that built them |

### Inputs and outputs of the two ONNX models

| Model | Inputs | Outputs |
|---|---|---|
| vision | `image` [1,3,704,1280] fp16 | `image_embeds` [880,2048], `deepstack_embeds` [3,880,2048] |
| decoder | `inputs_embeds` [B,L,2048] fp16<br>`position_ids` [3,B,L] int64 (M-RoPE t/h/w rows)<br>`deepstack_embeds` [3,B,L,2048] fp16<br>`past_key_i` / `past_value_i` [B,8,P,128] fp16 (i = 0..27) | `logits` [B,151936] fp32 (last position only)<br>`present_key_i` / `present_value_i` [B,8,P+L,128] |

L and P are dynamic in the decoder. Prefill uses L=1072, P=0; each decode step uses L=1.

A VLM cannot be a single .onnx file: it generates text one token at a time, and its weights exceed the
2 GB protobuf limit. It is therefore split into a vision model and a decoder model, which is the usual
way to deploy multimodal LLMs with TensorRT.

## Inference steps

`onnx_runner.py` implements the steps below. A C++ port should follow them exactly:

1. Resize the RGB frame to 1280x704 (bicubic), compute `x = (x/255 - 0.5) / 0.5`, convert to CHW fp16.
2. Run vision to get `image_embeds` and `deepstack_embeds`.
3. Build the decoder inputs:
   - `inputs_embeds = embed_tokens[config.input_ids]`, with the 880 rows where `input_ids == image_token_id` replaced, in order, by `image_embeds`;
   - `deepstack_embeds` goes into the same 880 rows of an all-zero [3,1,1072,2048] tensor.
4. Prefill: `position_ids` from `mrope_positions()` (constant for this prompt), empty KV cache (P=0).
5. Decode: take the argmax of `logits` (greedy) at each step, then feed one token:
   - all 3 rows of `position_ids` = `seq_len + rope_delta`;
   - `deepstack_embeds` = all zeros [3,1,1,2048];
   - the previous step's `present_*` become this step's `past_*`.

   Stop at an EOS token (151645 or 151643).
6. Parse `Person present: Yes/No` and `Person count: N`.

## Running on Jetson Orin

```bash
# clone this repo on the Orin and download the weights (see "Getting the model weights"), about 4.9 GB
pip3 install numpy pillow tokenizers cuda-python   # tensorrt ships with JetPack
python3 build_trt.py                                # writes *_vision.engine and *_decoder.engine, a few minutes
python3 onnx_runner.py --backend trt frame.png
# {"image": "frame.png", "raw": "Person present: Yes\nPerson count: 2", "person_present": true, "person_count": 2, ...}

# several images in one run
python3 onnx_runner.py --backend trt frame1.png frame2.png frame3.png
python3 onnx_runner.py --backend trt frames/*.png      # whole folder
```

- **Several images per run**: the model is loaded once, then the images are processed one after another
  (batch size 1, about 0.9 s each on an A10G), and each image prints its own JSON line. This is much faster than
  starting the script once per image, which reloads the 4 GB model every time. To call it from your own code,
  load once and call per frame:
  ```python
  from onnx_runner import PeopleCounter, parse_output
  counter = PeopleCounter(backend="trt")            # load once
  result = parse_output(counter.generate(frame))   # frame = file path or PIL.Image
  ```
- **Requires JetPack 6.x (TensorRT >= 10).** The engines are built STRONGLY_TYPED, which keeps the fp32 computation of the critical ops (RMSNorm, softmax, RoPE) inside the fp16 graph, and `position_ids` is an int64 input. TensorRT 8.5 (JetPack 5) supports neither.
- **An engine is tied to the GPU architecture and TensorRT version**, so it must be built on the Orin and cannot be copied from another machine.
- **Memory**: about 4.1 GB of engines, plus 0.6 GB of embeddings, the KV cache and activations. Orin NX 16 GB or AGX Orin works; an 8 GB Orin Nano does not fit.
- **The input resolution is fixed at 1280x704.** A camera with a different resolution needs a re-export from the training repo (`export_onnx.py --sample-frame <a frame from the new camera>`).
- **`onnx_runner.py` is not optimised for speed**: its TRT backend copies the KV cache between host and GPU at every step. For production, keep the KV cache on the GPU and capture the decode step in a CUDA graph.

## Test-set comparison with the original model

- **Test set**: the test part of `split_v2.json`, 52 frames, human-labelled ground truth. Frames per GT count: 0 people 2, 1 person 5, 2 people 25, 3 people 10, 4 people 10.
- **Original model**: HF transformers Qwen3-VL-2B (bf16) + PEFT LoRA adapter, greedy decoding, the same way as `finetune/eval_test.py` in the training repo. This re-run matches the `eval_test.json` recorded at training time exactly.
- **Test machine**: x86 A10G, TensorRT 10.3 (the JetPack 6 version), 2026-10-13.

| Backend | count exact | count MAE | present acc | same text as original | same count as original | sec/frame |
|---|---|---|---|---|---|---|
| **Original** PyTorch bf16 + LoRA | 0.942 (49/52) | 0.058 | 0.981 | - | - | 0.93 |
| **ONNX** Runtime CUDA fp16 | **0.962 (50/52)** | 0.038 | **1.000** | 50/52 | 51/52 | 0.81 |
| **TensorRT** fp16 | **0.962 (50/52)** | 0.038 | **1.000** | 50/52 | 51/52 | 0.93 |

Count exact by GT count:

| GT count | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Original | 2/2 | 5/5 | 22/25 | 10/10 | 10/10 |
| ONNX / TensorRT | 2/2 | 5/5 | 23/25 | 10/10 | 10/10 |

Frames with an error or a disagreement are listed below. On the other 48 frames all three backends give identical, correct output.

| Frame | GT | Original | ONNX | TensorRT | Note |
|---|---|---|---|---|---|
| frame_000012 | 2 | 3 | 3 | 3 | Same error in all three (a model error) |
| frame_000014 | 2 | **3** | 2 | 2 | Only the original is wrong |
| frame_000015 | 2 | 1 | 1 | 1 | Same error in all three |
| frame_000027_n395 | 0 | 0 | 0 | 0 | Original says `Person present: Yes`, which contradicts count 0; ONNX/TRT say `No`, which is correct |

**Conclusion**: the exported model loses no accuracy; on this test set it is slightly better, with one more
correct count and one more correct present answer. ONNX Runtime and TensorRT give identical output on all 52 frames.

The few disagreements come from numerical precision, not from the export:
- The original runs in bf16, the export in fp16. fp16 has more mantissa bits, so it is closer to fp32.
- In fp32 (TF32 off), the export wrappers match the HF implementation to 1e-4 relative error on the vision features and 5e-4 on the logits.
- The vision tower is very precision-sensitive: the HF model's own bf16 and fp32 vision features already differ by 20%.

Per-stage check (frame_000011, against the original bf16 model):

| Check | Result |
|---|---|
| input ids | identical |
| M-RoPE positions | identical |
| first prefill token | identical |
| preprocessing | Pillow bicubic vs. HF torchvision bicubic: max difference 0.008 after normalisation |

Note: the test and training sets come from the same video (same cabin, same people, same lighting).
This result only shows that the export does not change the model's behaviour; it says nothing about
generalisation to a new cabin or a new video.

### `eval/` files

| File | Contents |
|---|---|
| `summary.json` | All metrics for the three backends, plus the per-stage numeric check |
| `torch.json` / `ort.json` / `trt.json` | Per-frame raw output text, parsed result, time |
| `compare_test.csv` | One row per frame: GT, count from each backend, agreement flag, raw outputs |

To reproduce (from the training repo root):
1. Run `.venv/bin/python task1_people_count/deploy/build_trt.py`. It writes x86 engines into this folder.
2. Run `.venv/bin/python task1_people_count/deploy/compare_test.py --backends torch ort trt --numeric`.
