# Intra-GPU Co-Scheduling — Reproduction Package

This repository contains the source code, build scripts, and measurement campaigns for a
single-GPU engine that **co-schedules the two execution engines inside one commodity GPU**
—the SIMT array (CUDA cores) and the matrix engine (tensor cores)—for exact bitmap
frequent-itemset mining. A measurement-driven router prices every enumeration wave on both
engines under a learned affine cost model and routes it to the cheaper one, under hysteresis;
a double-buffered pipeline overlaps mask construction on the SIMT engine with matrix-engine
execution of the previous wave. This code is the companion of a manuscript currently under
review.

## Datasets

**The datasets are NOT included in this repository** due to licensing. They are publicly
available benchmark corpora; please obtain them yourself from their public sources:

- The FIMI benchmark repository and its SPMF mirror: chess, mushrooms, retail, connect,
  pumsb, accidents, kosarak, T10I4D100K.
- The webdocs corpus (FIMI/SPMF).
- TCGA-derived transaction data from the TCGA public portal (the version used here has
  all-present items stripped; replicated datasets such as `pumsb_x256` are generated from
  the originals by N-times replication).

The engine reads one transaction per line of integer item IDs. Place the dataset files in
a `TransactionSets/` directory at the repository root (the directory is git-ignored).

## Layout

- `src/` — engine source.
  - `kernel_bitmap.cu` — SIMT (CUDA-core) backend, 128-bit segment layout.
  - `kernel_bitmap_diag.cu` — SIMT backend, diagnostic variant.
  - `kernel_bmma.cu` — matrix-engine (tensor-core) backend, one-bit operand MMA.
  - `kernel_router.cu` — measurement-driven dual-engine router (modes D and E).
  - `bmma_bench.cu` — matrix-engine microbenchmark driver.
  - `microbench_contention.cu` — contention-pricing microbenchmark.
- `tools/` — everything needed to build, run, and verify.
  - `build_*.bat` — nvcc builds (CUDA 12.8 + Visual Studio 2022 toolchain, sm_86; adjust
    `-gencode` for your GPU architecture).
  - `campaignA*.sh`, `run_e13*.sh`, `e11_connect.sh`, `run_pumsb_v9_round.sh` —
    interleaved measurement campaigns (serial router, pipelined router, sweeps).
  - `set_dataset.py` — patches the dataset/threshold line in the kernel sources.
  - `gen_synth.py` — synthetic level-wise workload generator (grid configurations).
  - `ablate.py` — ablation runner.
  - `compare_mfis.py` — output set-identity / MD5 comparison for verification.

## Building

From the repository root, create the binary directory and build:

```bat
mkdir run
tools\build_all128.bat    :: SIMT + matrix-engine backends
tools\build_router.bat    :: measurement-driven router
tools\build_micro.bat     :: microbenchmarks (optional)
```

## Running a campaign

Campaign scripts expect the layout `run/` (binaries) and `TransactionSets/` (datasets) at
the repository root, and are invoked from the root, e.g.:

```bash
bash tools/campaignA_v9_router.sh chess 0.80 5
# usage: campaignA_v9_router.sh <dataset> <threshold> <rounds>
```

Router modes are selected with the `ROUTE_MODE` environment variable (`1` = serial router,
`2` = pipelined router); `ROUTE_LOG=<file.csv>` records the per-wave routing trajectory.
Reported timings are medians of interleaved rounds; **compare ratios within one campaign
only** — absolute numbers are machine-specific.

## Verification

Every backend (SIMT, matrix, routed) must produce bit-identical output (set- and
frequency-identical itemsets) regardless of routing history:

```bash
python tools/compare_mfis.py <output_a.txt> <output_b.txt>
```

## Note

This repository intentionally contains no manuscript title, author list, or venue
information; it will be linked from the paper upon acceptance.
