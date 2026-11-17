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
was 0.9557. Ground-truth-assisted translation reduced the gap by 36.8%; a
seed-0 rigid-registration analysis reduced its corresponding gap by 62.0%.
These registration results describe alignment error and are not inference procedures.

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
