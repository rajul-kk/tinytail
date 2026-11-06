# Scoop check

Date: 2026-11-06. Each source is tagged by what was actually read in this session:
**abstract** (arXiv abstract page), **README** (GitHub landing page), **snippet** (search-result text only).
No full PDFs were read, so every verdict below is provisional and must be re-checked against full texts before any
novelty claim is written down.

| Source | What was read | What it covers | Overlap with this study |
|---|---|---|---|
| Tizpaz-Niari and Sankaranarayanan, arXiv 2404.07170, "Worst-Case Convergence Time of ML Algorithms via Extreme Value Theory" | abstract; search snippet (3,333 DNN controllers, EVT predicted inference convergence accurately in about 75% of cases) | EVT applied to ML training convergence time and DNN inference time on general systems. Abstract does not mention microcontrollers. | Closest on method (EVT for DNN timing). Not a per-inference execution-time study on an MCU, no interference variable, no cache-vs-cacheless contrast. Full text not read, so a Cortex-M experiment hidden in it cannot be excluded. |
| Bartoli et al., arXiv 2505.15622, "Benchmarking Energy and Latency in TinyML" (IJCNN 2025) | abstract | Energy and latency method on STM32N6 (NPU); phases pre/inference/post; 1,000 runs per model; no external power meter; energy-delay product. | Source of the energy/latency methodology to reproduce. Reports averages; abstract mentions no percentile, tail or WCET analysis. Different hardware class (NPU). |
| Breier, Kucerak, Hou, arXiv 2602.16309, "The Weight of a Bit" | abstract | EMFI sensitivity of number formats for ResNet/VGG on an embedded memory chip. | No timing measurement in abstract. Relevant only as fault-sensitivity context (the repo does not do fault injection in this phase). |
| MLPerf Tiny, arXiv 2106.07597 | abstract | Benchmark suite: keyword spotting, visual wake words, image classification, anomaly detection; accuracy, latency, energy. | Defines the reference model and task to reproduce. Reports latency per inference, not tail bounds. |
| mlcommons/tiny repository | README (landing page only; the `benchmark` folder contents and the anomaly-detection model files were not shown) | TFLM-based reference implementations, results tables for v0.5 to v1.1. | Reference code location. Model details taken from the author's prior knowledge (dense autoencoder on ToyADMOS/DCASE 2020 task 2 toy car, 640-dim input) and must be verified against the repo when the benchmark folder is opened. |
| Search: MBPTA / pWCET on DNN inference on microcontrollers (3 queries) | snippets only | Found: Kumar and Jain, pWCET for GoogleNet/CaffeNet from measured inference times (large networks, snippet only); Fedotova, Krause, Siemens, MBPTA with EVT on Cortex-A (snippet only); Hansen, Hissam, Moreno, MBPTA background (snippet only); 2608.21646 power-performance characterisation of TinyML (snippet only: per-access cycle counts, no tail analysis). | No hit combining MBPTA/pWCET with int8 inference on Cortex-M. Search tool was a general web search, not Google Scholar or dblp, so absence is weak evidence. |

## Verdict

**Partly open, leaning open, unverified.** No source read here performs an EVT/MBPTA validity study of int8 TinyML
inference on a cacheless Cortex-M0+ versus a cached Cortex-M, with interrupt, DMA and RTOS interference as the independent
variable. The nearest work (2404.07170) was only read at abstract level.

Honest framing for any write-up: "first open measurement study as far as I could find in a shallow search", plus the
caveat that the result for cacheless cores may be a negative or trivial one (see risks in `design.md`).

## Follow-ups before claiming novelty

1. Read the full PDF of 2404.07170 and of Kumar and Jain (pWCET for DNNs).
2. Search dblp / Google Scholar for: "MBPTA" + "Cortex-M", "pWCET" + "TinyML", "timing jitter" + "TFLite Micro",
   WCET analysis of CMSIS-NN kernels, and the real-time-systems venues (RTSS, RTAS, ECRTS, EMSOFT, WCET workshop).
3. Open the mlcommons/tiny `benchmark` folder and record the exact anomaly-detection model and preprocessing.
4. Rust versus C was cut from scope and is not part of this check.
