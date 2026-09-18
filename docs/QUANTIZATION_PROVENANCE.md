# Quantization provenance and ownership

The [W8A8 checkpoint author's model card](https://huggingface.co/nytopop/Qwen3-8B.w8a8)
describes an Ampere-oriented INT8 artifact produced from Qwen/Qwen3-8B with llm-compressor.
Its published recipe combines SmoothQuant (strength 0.7) and GPTQ in W8A8 mode, leaves the
language-model head unquantized, and uses 256 calibration samples with maximum sequence length
4096 from `neuralmagic/LLM_compression_calibration`. This is **author-reported provenance**,
not a calibration run performed by this project.

We pin the downloaded artifact to revision `13e255a9648ec08d3873bce1c3d9886a76494c43`, verify
the two weight shards by size and SHA-256, inspect quantization metadata and measure quality
against the FP16 model. A hash proves artifact identity, not that the author's entire training or
calibration pipeline has been independently reproduced.

The project's own contributions are runtime integration, the guarded INT8 GEMM, fusion and
fallback experiments, profiling, quality regression and serving measurement. AWQ/Marlin,
SmoothQuant, GPTQ, CUTLASS, FlashInfer and the underlying SGLang engine are upstream work.
Do not put “implemented SmoothQuant/GPTQ calibration” or “invented INT4 inference” on the resume.

Quantization formats and model-file revisions are listed in `configs/model_sources.json`.
Only configurations with independent quality evidence may be recommended from our measurements.
