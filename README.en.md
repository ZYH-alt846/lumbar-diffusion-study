# Biplanar lumbar reconstruction experiments

This repository records my experiments on reconstructing L1–L5 segmentation
labels from paired AP/lateral DRRs and calibrated projection matrices.
I started with deterministic reconstruction, then evaluated conditional latent
diffusion, coarse-to-fine bridge refinement, and adaptation of an independently
trained 3D label prior.

## Latest comparison

On the 34-case internal selection set, the historical deterministic reference
(DPR-R) achieved Dice 0.8244, conditional regression (CR) 0.8099, scratch
conditional diffusion (CD-S) 0.8000, and frozen-prior adaptation (CD-F) 0.7878.
CR, CD-S and CD-F were each trained for 150,000 updates. Diffusion scores were
averaged over five independently scored samples per case. All four methods
had zero missing vertebrae.

The conditional model responded to patient-specific image/matrix pairs:
real conditions improved Dice by 0.2436 relative to fixed donor conditions.
This response did not close the reconstruction gap. Label-AE round-trip Dice
was 0.9557. With matched seed-0 outputs, one ground-truth-assisted rigid
transform shared by L1–L5 reduced the gap by 28.0%; independent per-vertebra
registration reduced it by 62.0%. The latter permits changes in inter-vertebra
arrangement and cannot be read as whole-spine pose error. These are diagnostic
procedures that use ground truth.

An archived Swin-X2S-Tiny adaptation scored 0.7907 Dice on the same 34 cases,
with no missing vertebrae; same-cohort HD95 was not saved. It used 128² inputs,
fresh initialization and 100 epochs, rather than the paper's main configuration.
CD-F minus this adaptation was −0.0030 Dice (95% interval, −0.0164 to 0.0162).
See the [public-method comparison](data/aggregate/public_baseline.json).

The 34 cases were used for development and checkpoint selection. The final
105-case test evaluation and independent training repeats remain outstanding.

## Contents

- [Experiment record](docs/EXPERIMENTS.md): setup, progression, results and follow-up decisions.
- [Results](docs/FINDINGS.md): comparisons and error analysis.
- [Code guide](docs/CODE_GUIDE.md): models, objectives, sampling and tensor conventions.
- [Running the code](docs/REPRODUCIBILITY.md).
- `data/aggregate/`: method scores, selection curves and aggregate analyses.

The repository includes core code and aggregate results. Obtain real data
through the [official PPCNet page](https://huggingface.co/datasets/ppcnet-dataset/PPCNet).
Images, case-level data and model weights are not distributed here.
See [data information](docs/DATA_AND_PRIVACY.md) and [license status](LICENSE_STATUS.md).
