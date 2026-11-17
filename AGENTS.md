# Repository boundaries

This is a curated research snapshot, not a backup of the private working directory.
Preserve negative findings and distinguish internal selection from independent testing.
Never add patient-level data or identifiers, actual camera matrices, latent codes,
checkpoints, cloud credentials, SSH helpers, absolute paths, machine usernames,
raw logs, manuscript binaries, or private repository history.
Add only reviewed files to release_manifest.json, then run scripts/privacy_audit.py.
Never disable the privacy checks to get a commit through. Do not run training or
access held-out data unless separately requested. Public visibility and licensing
require the owner's explicit release decision. Do not infer a license grant.
