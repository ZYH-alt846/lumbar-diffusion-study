# Conditional diffusion for biplanar lumbar label reconstruction

A curated research snapshot containing core implementation, aggregate evidence,
and an experiment narrative. The task is calibrated AP/lateral DRR to L1–L5
3D labels. This is an empirical comparison and failure analysis, not a claim
that diffusion improved over a strong deterministic reference.

The repeatedly used 34-case internal selection cohort yielded Dice 0.8244
(historical deterministic reference), 0.8099 (conditional regression),
0.8000 (scratch diffusion), and 0.7878 (frozen-prior conditional diffusion).
Five separately scored diffusion samples were averaged per patient.

Joint-condition replacement confirmed patient-specific dependence (+0.2436
Dice), while AE round-trip fidelity and ground-truth-assisted registration
provided more specific error diagnostics. Registration is not deployable or
a unique causal attribution. No final held-out test or training-seed replication
is presented. Pretraining and freezing have not been causally separated.

See [the Chinese overview](README.md), [experiment history](docs/EXPERIMENTS.md),
[reproduction scope](docs/REPRODUCIBILITY.md), and [data/privacy](docs/DATA_AND_PRIVACY.md).
Real images, labels, latent codes, checkpoints, patient IDs, personal information,
cloud endpoints, and private history are excluded. This package is intended for
private review first; an open-source license has not yet been selected.
