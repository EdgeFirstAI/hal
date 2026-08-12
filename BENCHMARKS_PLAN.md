# Article 1 Benchmarks — JPEG Decode → Model Input

**Scope:** JPEG decode and preprocessing only. **v0.4** — supersedes v0.3.

**Home:** this plan and its results land in the `EdgeFirstAI/hal` repository. It expands and cleans up `BENCHMARKS.md`; the harness code lives in a new `benchmarks/` subfolder (see §2).

**In scope:** encoded JPEG bytes → letterboxed, normalized, correctly-typed model input tensor.

**Out of scope:** NPU/delegate comparisons, post-processing/NMS, overlay rendering, camera capture, GPU contention under concurrent display load. Those belong to later articles in the series — in particular the live-camera-with-overlay article, which is where contention effects are properly the subject rather than a distraction.

**Validation anchor:** COCO val2017, detection + segmentation, EFPI-style — mAP reported for every arm.

---

## 1. The organizing question

Not "which library is fastest" but **where does the pixel work actually run**.

A JPEG decode to a model tensor is four jobs: entropy decode, inverse DCT, chroma upsample + YCbCr→RGB, and geometric resize + normalize + dtype. Every implementation splits those four across the CPU, a shader, a fixed-function 2D blitter, or a fixed-function codec — and the split is the whole story.

| Class | Entropy + IDCT | Upsample, CSC, scale, normalize, dtype |
|---|---|---|
| **CPU** | CPU SIMD | CPU SIMD |
| **Hybrid-GL** | CPU SIMD → native NV12/NV16/NV24 | GPU shader, single fused pass |
| **Hybrid-2D** | CPU SIMD → native chroma | Fixed-function 2D blitter (G2D, RGA) |
| **HW+GL** | Fixed-function codec → dma-buf | GPU shader, single fused pass |
| **HW+2D** | Fixed-function codec → dma-buf | Fixed-function 2D blitter |
| **GPU-decode** | GPU (CUDA) | GPU — *x86 / Jetson only* |

The i.MX 95 is the only board in the fleet expected to run all five of the first five, which makes it the article's spine. Every other board is a partial slice testing whether the i.MX 95 conclusion generalizes.

> [!NOTE]
> Hybrid-GL and Hybrid-2D stay separate. A shader is programmable; a blitter is fixed-function. Where the ranking between them inverts across boards, that inversion is the argument for capability-probed backend selection — and is more interesting than either row alone.

**There is no OpenGL JPEG decoder, on any board, and that's a finding.** Entropy decode is inherently serial and doesn't map to a fragment shader; the only true GPU decoders are CUDA. This is the structural reason the *hybrid* class exists at all. State it plainly — readers will ask.

---

## 2. Deliverable and repository layout

```text
hal/
├── BENCHMARKS.md              # expanded + cleaned; published results land here
└── benchmarks/                # NEW — excluded from CI/CD
    ├── README.md              # how to reproduce on your own target
    ├── probe/                 # capability probe (§12 step 1)
    ├── common/                # timing, CSV schema, COCO loader, mAP hookup
    ├── modules/               # descriptive names (no c1_/h3_ prefixes)
    │   ├── turbojpeg/         # libjpeg-turbo reference arm (native C)
    │   ├── hal_cpu/
    │   ├── hal_gl/
    │   ├── hal_g2d/
    │   ├── gstreamer/         # deferred
    │   └── hal_v4l2_gl/
    ├── parity/                # §10 output-parity harness
    └── results/<board>/
```

Requirements for the subfolder:

- **Links against the HAL by path.** Rust modules take path deps into `crates/` and use the public API like any other consumer. That is also a standing check that the public surface is sufficient to build a real pipeline — if a module needs something not `pub`, that's a HAL bug, not a harness workaround.
- **Excluded from the workspace, not just from CI.** Set `exclude = ["benchmarks"]` in the root `[workspace]` so `cargo build --workspace` and `cargo test --workspace` never reach it; `benchmarks/` is its own Cargo workspace. Add the matching path filter to the workflows. These modules pull in OpenCV, GStreamer, TurboJPEG, CUDA and MediaPipe — none of which belong anywhere near the HAL's dependency graph or its build times.
- **Reproducible by an outsider on their own hardware.** Each module is a standalone, buildable sample with its own README, dependency list, and one-command invocation. If a reader with an i.MX 95 or a Pi 5 can't rerun a published number, the number isn't published in good faith.
- **Sample code is part of the deliverable, not scaffolding.** The OpenCV and TurboJPEG modules in particular double as reference implementations readers will copy. Write them to be read.
- **Results are committed.** `results/<board>/` under version control, so numbers carry provenance and can be diffed across HAL releases.

---

## 3. Why this is measured end-to-end

A decode-only benchmark is easy to win and means nothing. Decoding to NV12 is ~24% faster than to RGB — but only because upsample and CSC moved downstream into the shader. Stop the stopwatch at the decoder and you've published an accounting trick.

Microbench each stage for insight; **publish speedups measured across the whole front half**, tensor boundary and accuracy held fixed.

Constant across every arm on a board:

1. **Output tensor contract** — 640×640, letterbox pad 114, centered, RGB, board's required dtype/layout.
2. **Everything downstream** — same model artifact, same inference backend, same EdgeFirst schema decoder, same NMS thresholds. Using our own NPU enablement is correct here: it's the constant, not the subject.
3. **Dataset and metric** — full COCO val2017, mAP50-95 per arm.

Measured, not just latency — offloading moves cost sideways rather than removing it:

- **CPU%** per-core, not aggregate
- **DDR read/write bandwidth** — the GPU path trades CPU cycles for memory traffic; if the win doesn't survive here we haven't proven it. Use the `linux-perf` skill's PMU counters.
- **Peak RSS + DMA-buf footprint**
- **GPU busy %** where the driver exposes it

---

## 4. Fixed constants

Deliberately constrained. Every degree of freedom removed here is a confound we don't have to explain later.

| Constant | Value | Rationale |
|---|---|---|
| Model | YOLOv8n (det), YOLOv8n-seg (seg), 640×640 | Widest export coverage |
| Target shape | Full `imgsz` padding, 640×640, every arm | Matches fixed-shape exported engines |
| Pad value | **114**, no exceptions | Ultralytics convention; any other value is a different problem |
| Pad placement | Centered | |
| Resize | **Bilinear or better** (`INTER_LINEAR`, GPU bilinear, or `INTER_AREA` on downscale) | Nearest is out of scope — it changes accuracy, not just speed |
| NMS | `conf=0.001`, `iou=0.7` for mAP; `conf=0.25` for latency | |
| Dataset | COCO val2017, all 5000 images, **as JPEG on disk** | The decode is under test |
| Iterations | 10 warmup / 100 measured (microbench); full 5000 (E2E) | |
| Governor | `performance`, pinned; temps logged start and end | |

**On Ultralytics `rect`:** the HAL always pads to the full `imgsz` target, and so does Ultralytics for the engines we cover. Per the Ultralytics predict docs, minimum-rectangle padding applies only when every image in the batch shares a shape *and* the backend supports it — PyTorch `.pt`, dynamic ONNX, or Triton — and otherwise images are padded to the full `imgsz` target regardless. `rect=False` is recommended for fixed-input exported models. Since our targets are fixed-shape exports, full-`imgsz` padding is already the effective behaviour; set `rect=False` explicitly on the PyTorch arms so they match, and record the setting. No further caveat needed. (<https://docs.ultralytics.com/modes/predict#fixed-shape-vs-minimum-rectangle-rect>)

> [!IMPORTANT]
> COCO val2017 spans ~1000 distinct resolutions. Log geometry renegotiations per arm — on the V4L2 path this is the difference between ~1 ms of ioctls and ~110 ms of kernel buffer reallocation.

---

## 5. Claims to reconfirm

Every constraint below comes from our own prior documentation or a third-party report. All of it predates this plan and some of it is likely stale. **Nothing here is a settled fact until re-measured in step 1 of the run order.** Do not write any of it into the article, and do not design an arm out of the matrix on its basis, until confirmed.

| # | Claim | Source | If confirmed | If refuted |
|---|---|---|---|---|
| R1 | NV12→planar GL on Vivante GC7000UL hangs the GPU unrecoverably | `BENCHMARKS.md` Known Issue #8 | Ours to fix — open an internal ticket and a public GitHub Issue if the root cause is in the Vivante driver | Correct `BENCHMARKS.md`; the i.MX 8MP Hybrid-GL row is fully in play |
| R2 | Vivante GLES cannot emit packed RGB natively; two-pass shader is 3–4× slower than G2D | `BENCHMARKS.md` Known Issue #10 | Document as a driver capability limit; public GitHub Issue referencing the vendor driver | Correct `BENCHMARKS.md` |
| R3 | macOS GL coverage is YUYV→RGBA only | `BENCHMARKS.md` gap #17 | Close the gap in the HAL before benchmarking macOS | **Likely outdated — correct `BENCHMARKS.md` and run the full macOS matrix.** Treat this as the expected outcome |
| R4 | DMA-buf letterbox on i.MX 95 is slower than PBO (3.4 ms vs 1.4 ms) | `BENCHMARKS.md` Known Issue #6, v1.3-era | Investigate; v3.8 numbers suggest it's already superseded | Remove the stale issue from `BENCHMARKS.md` |
| R5 | The `DstDma` zero-copy V4L2 path never fires on COCO despite 31.2% passing the alignment gate | `V4L2_VS_NEON_BENCHMARK.md` §4, §13 | Ours to fix — pool-stride hypothesis; must be resolved before publishing the hardware class | Update the benchmark doc |
| R6 | NVIDIA EGL cannot import DMA-buf; PBO only | `BENCHMARKS.md` platform notes | Document as vendor limitation; public GitHub Issue | Correct `BENCHMARKS.md` |
| R7 | Jetson YUV EGL import unsupported | `BENCHMARKS.md` platform notes | As R6 | As R6 |
| R8 | `mxc-jpeg` output format tracks the source's encoded colorspace (4:4:4 → V308, not NV12) | NXP community forum threads, third-party | Public GitHub Issue against our own docs describing the behaviour; stratify results per §9 | Simplifies the hardware class considerably |
| R9 | `libcamerasrc` does not advertise `video/x-raw(memory:DMABuf)`; `hailonet` expects CPU-accessible buffers | Hailo community forum, third-party | Public GitHub Issue on the relevant upstream project, linked from the article | Reframe the Pi 5 section |
| R10 | MediaPipe `GpuBuffer` has no dma-buf import path on Linux | Inferred from docs, unverified | Note as a scoping limit for the H5 arm | Raises H5's priority |

**Disposition rule.** Anything inside our control gets fixed, and `BENCHMARKS.md` gets corrected in the same PR. Anything outside our control that survives re-measurement gets a **public GitHub Issue** — not JIRA — on the appropriate upstream project, so the article can link to it and the limitation is publicly on record rather than asserted in a blog post.

---

## 6. Class composition per board

Provisional. Cells marked ⚠️ depend on a §5 claim and may open up once re-measured.

| Board | CPU | Hybrid-GL | Hybrid-2D | HW+GL | HW+2D | GPU-decode |
|---|---|---|---|---|---|---|
| **i.MX 95** (Mali G310) | ✅ | ✅ DMA-buf | ✅ G2D/PXP | ✅ `mxc-jpeg` | ✅ | ❌ |
| **i.MX 8M Plus** (Vivante) | ✅ | ⚠️ R1, R2 | ✅ G2D | ⚠️ verify node | ⚠️ | ❌ |
| **Raspberry Pi 5** (V3D) | ✅ | ✅ DMA-buf | ❌ | ❌ | ❌ | ❌ |
| **Jetson Orin Nano Super** | ✅ | ✅ PBO ⚠️ R7 | ❌ | ⚠️ NVJPG via L4T API | ❌ | ✅ nvJPEG/DALI |
| **x86-64 + NVIDIA** | ✅ | ✅ PBO ⚠️ R6 | ❌ | ❌ | ❌ | ✅ nvJPEG/GPUJPEG |
| **macOS M2 Max** | ✅ | ⚠️ R3 | ❌ | ❌ (private API) | ❌ | ❌ |

---

## 7. Stage A — decoder microbench

### CPU class

| ID | Decoder | Output |
|---|---|---|
| D1 | `edgefirst-codec` NEON/SSE | RGB u8 |
| D2 | `edgefirst-codec` | RGB f32 / i8 (fused dtype) |
| D3 | libjpeg-turbo `tjDecompress2` | RGB u8 |
| D4 | libjpeg-turbo + `TJFLAG_FASTUPSAMPLE\|TJFLAG_FASTDCT` | RGB u8 |
| D5 | OpenCV `cv2.imdecode` | BGR u8 |
| D6 | Pillow(-SIMD) | RGB u8 |
| D7 | `zune-jpeg` / `image` crate | RGB u8 |
| D8 | `torchvision.io.decode_jpeg` (CPU) | RGB tensor |
| D9 | GStreamer `jpegdec` | I420 |

### Hybrid class (decode half — native chroma out)

| ID | Decoder | Output |
|---|---|---|
| D10 | `edgefirst-codec` | NV12 |
| D11 | `edgefirst-codec` | NV16 / NV24 (4:2:2 / 4:4:4 sources) |
| D12 | libjpeg-turbo `tjDecompressToYUV2` | planar YUV |

### Hardware class

| ID | Decoder | Output | Board |
|---|---|---|---|
| D13 | HAL V4L2 M2M (`mxc-jpeg`) | native NV12/NV16/NV24 | i.MX 95 |
| D14 | GStreamer `v4l2jpegdec capture-io-mode=dmabuf` | NV12 | i.MX 95 |
| D15 | `mxc_jpeg_test` (raw ioctl) | native | i.MX 95 — driver floor |
| D16 | ImageIO / `CGImageSource` | BGRA | macOS (opaque — may be CPU) |

### GPU-decode class

| ID | Decoder | Output | Board |
|---|---|---|---|
| D17 | nvJPEG (`decode_jpeg(device='cuda')` / DALI) | RGB on GPU | x86, Jetson |
| D18 | GPUJPEG (CUDA) | RGB on GPU | x86, Jetson — optional |

Report per decoder: ms p50/p95/p99, MP/s, CPU%, hot-loop allocations.

---

## 8. Stage B — preprocessor microbench

Input: whatever Stage A produced. Output: the §4 tensor contract.

| ID | Preprocessor | Class | Boards |
|---|---|---|---|
| P1 | HAL `convert()` GL + DMA-buf | Hybrid-GL / HW+GL | i.MX, Pi 5 |
| P2 | HAL `convert()` GL + PBO | Hybrid-GL | x86, Jetson |
| P3 | HAL `convert()` GL + IOSurface | Hybrid-GL | macOS |
| P4 | HAL `convert()` G2D | Hybrid-2D / HW+2D | i.MX |
| P5 | HAL `convert()` CPU | CPU | all |
| P6 | Ultralytics `LetterBox` + numpy/torch | CPU | all |
| P7 | OpenCV `resize` + `copyMakeBorder` + `blobFromImage` | CPU | all |
| P8 | libyuv NV12→RGB + scale | CPU | all |
| P9 | GStreamer `glupload ! glcolorconvert ! videoscale/videobox` | Hybrid-GL | Linux |
| P10 | GStreamer `imxvideoconvert_g2d` | Hybrid-2D | i.MX |
| P11 | MediaPipe `ImageToTensorCalculator` GPU | Hybrid-GL | ⚠️ R10 |
| P12 | MediaPipe `ImageToTensorCalculator` CPU | CPU | all |
| P13 | `torch.nn.functional.interpolate` + normalize | GPU-decode | x86, Jetson, macOS |
| P14 | DALI `fn.resize` + `fn.crop_mirror_normalize` | GPU-decode | x86, Jetson |

MediaPipe (P11/P12) is the highest-effort arm — bazel build, GLES 3.1+ requirement. Timebox it. It's the closest functional analogue to `convert()` in open source, so worth real effort, but confirm it builds on one embedded board before promising it in the post.

---

## 9. Stage C — composed arms (the published numbers)

Full COCO val2017 sweep with mAP. Everything downstream identical.

| Arm | Class | Decode | Preprocess | Boards |
|---|---|---|---|---|
| **C1** | CPU | `cv2.imdecode` | Ultralytics `LetterBox` | all — *the reference everyone knows* |
| **C2** | CPU | `cv2.imdecode` | OpenCV `blobFromImage` | all |
| **C3** | CPU | libjpeg-turbo RGB | libyuv / OpenCV | all |
| **C4** | CPU | HAL NEON → RGB | HAL CPU | all |
| **H1** | Hybrid-GL | libjpeg-turbo YUV (D12) | libyuv + OpenCV | all — *native chroma without the HAL* |
| **H2** | Hybrid-GL | GStreamer `jpegdec` | `glupload ! glcolorconvert` | Linux |
| **H3** | Hybrid-GL | HAL NEON → NV12 | HAL GL | all with GL |
| **H4** | Hybrid-2D | HAL NEON → NV12 | HAL G2D | i.MX |
| **H5** | Hybrid-GL | OpenCV decode | MediaPipe GPU | ⚠️ R10 |
| **W1** | HW+GL | GStreamer `v4l2jpegdec` dmabuf | `glupload ! glcolorconvert` | i.MX 95 |
| **W2** | HW+GL | HAL V4L2 → NV12 | HAL GL | i.MX 95 |
| **W3** | HW+2D | HAL V4L2 → NV12 | HAL G2D | i.MX 95 |
| **G1** | GPU-decode | nvJPEG | DALI GPU | x86, Jetson |

**W1 is the arm that matters most.** The strongest genuinely independent implementation of the same idea, on the only board with a hardware codec. Tune it properly — `GST_GL_API=gles2`, correct `io-mode` on *both* queues, buffer-pool depth — and if it wins, publish that. A tuned competitor that beats us on one axis makes every other number more believable.

**H1 is the honest control.** libjpeg-turbo has shipped native-chroma decode since 1.5, so decode-to-native-format is not ours. H1 isolates how much of the win is the *idea* versus the *implementation*: dma-buf as the decode target, MCU-aligned strided writes, and the single fused GPU pass. Note that `tjDecompressToYUV2` does an internal copy when dimensions aren't MCU multiples — most of COCO. Fair cost for H1 to carry, but name it.

---

## 10. Source-chroma stratification

Per R8, `mxc-jpeg` may emit a format tracking the *encoded* colorspace of the source. If so, a 4:4:4 JPEG comes out as V308 rather than NV12, GStreamer silently inserts a software `videoconvert`, and the "hardware" arm quietly becomes a CPU arm.

COCO val2017 contains both 4:2:0 and 4:4:4 files. Therefore:

- Run a **source-chroma census** of val2017 headers before anything else (§12 step 2).
- Log the **negotiated capture FourCC per image**, not per run.
- Publish the histogram (`% 4:2:0 → NV12`, `% 4:4:4 → V308`, `% software fallback`).
- Report hardware-class timings **stratified by source subsampling**, not just pooled.

If a slice of the dataset silently falls off the hardware path, that's a finding on its own — and pooling would otherwise make W1/W2 look worse than they are.

---

## 11. Output parity

Before comparing speed, check the arms compute the same thing. They won't.

Against a float64 reference (libjpeg-turbo fancy upsampling → BT.601 full-range → bilinear resize → pad 114), report per arm: max abs channel error, mean abs error, % pixels differing by > 1 LSB, and resulting Δ mAP50-95 (det) / Δ mask mAP (seg).

| Divergence source | Effect |
|---|---|
| Fancy vs fast chroma upsampling | libjpeg-turbo `TJFLAG_FASTUPSAMPLE`; our nearest/integer `texelFetch` replication |
| BT.601 vs BT.709, limited vs full range | JFIF says full-range BT.601; drivers frequently assume otherwise |
| Chroma siting | Specified by neither V4L2 nor DRM — the shader is the only place it's determined |
| Resize kernel | `INTER_LINEAR` vs GPU bilinear vs `INTER_AREA` on downscale |
| Rounding | `round(dh - 0.1)` asymmetry in the Ultralytics letterbox |
| Quantization point | Normalize in f32 then quantize, vs fused in the shader |

Run on a 50-image subset **before** any 5000-image sweep. If it moves mAP, it reframes the article. If it doesn't, the null result is what licenses every speed claim.

---

## 12. Run order

1. **Re-measure every §5 claim.** Nothing else starts until R1–R10 are confirmed or refuted, `BENCHMARKS.md` is corrected, and any out-of-our-control survivors have public GitHub Issues filed.
2. **Source-chroma census of COCO val2017** — one pass over the headers. Determines how much of the dataset the hardware class can take at all.
3. Capability probe on all six boards — `v4l2-ctl --list-formats-ext`, `ls /dev/dma_heap/`, `eglinfo | grep dma_buf_import`, `gst-inspect-1.0 | grep -E "jpegdec|v4l2"`.
4. Stage A decode microbench everywhere. Cheap, produces the headline table.
5. Parity check (§11) on 50 images. Catch divergence before burning full sweeps.
6. Stage C on i.MX 95, all five classes. **The article's spine.**
7. Stage C on Pi 5 and x86 — the two contrarian results.
8. Stage B preprocessor microbench, to explain *why* Stage C came out that way.
9. i.MX 8MP, Jetson, macOS to complete the matrix.
10. MediaPipe (H5) last, timeboxed.

---

## 13. Results schema

`results/<board>/<class>_<arm>.csv`:

```csv
board,class,arm,decode_id,preproc_id,src_subsampling,
model,task,src_format,negotiated_capture_fourcc,dst_format,dtype,layout,
ms_p50,ms_p95,ms_p99,mpix_per_s,
cpu_pct_total,cpu_pct_peak_core,gpu_busy_pct,
ddr_read_mbs,ddr_write_mbs,peak_rss_mb,dmabuf_mb,hot_loop_allocs,
map50_95,map50,mask_map50_95,max_abs_err,mean_abs_err,pct_pixels_gt_1lsb,
n_images,geometry_renegotiations,sw_fallback_frames,
temp_start_c,temp_end_c,versions,notes
```

`src_subsampling` ∈ {`420`, `422`, `444`, `gray`}. `sw_fallback_frames` counts frames where a nominally-hardware arm hit a software path — non-zero and unreported means the arm is mislabelled. Stage C rows must have `map50_95`.

---

## 14. Board-specific must-dos

**i.MX 95** — resolve R5 before publishing. Right now the hardware class is a strided copy-out, and W1 might legitimately achieve a zero-copy import we don't. Run the aligned-subset experiment first.

**i.MX 8M Plus** — re-measure R1 and R2 before assuming any cell is closed. If the Hybrid-2D-beats-Hybrid-GL ranking holds after re-measurement, that inversion versus i.MX 95 is the argument for capability-probed backend selection and deserves its own section. If it doesn't hold, `BENCHMARKS.md` needs correcting and the section goes away.

**Raspberry Pi 5** — A76 decodes fast (~4.2 ms at 720p). Hybrid-GL may lose to pure CPU. Publish it if so.

**x86 / Jetson** — expect zero-copy to lose to plain staging. Most novel result in the article; give it a section, not a footnote.

**macOS** — R3 is the one most likely to be stale. Re-measure the actual format coverage first; the expectation is that the full matrix runs and `BENCHMARKS.md` gets corrected.

---

## 15. Things that will bite

- **`cv2.imdecode` returns BGR.** Silent channel swap → mAP collapse. Check first.
- **OpenCV ships its own libjpeg-turbo.** Version and SIMD level may differ from the system build; record both.
- **GStreamer negotiation lies.** Verify with `GST_DEBUG=GST_CAPS:4` that the dmabuf path actually negotiated rather than silently falling back to system memory. This is the single most likely way to accidentally strawman W1.
- **Thermals.** A 5000-image sweep will throttle the Pi 5 and the FRDM boards. Active cooling, log temps, discard throttled runs.
- **Stale documentation is the default.** Several §5 claims are a year or more old. Treat `BENCHMARKS.md` as a hypothesis under test, not a specification.
