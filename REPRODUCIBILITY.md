# Recomputing the results

Run commands from the repository root. CPU audits and historical statistical reproduction use NumPy, pandas, SciPy, scikit-learn and matplotlib. GPU training requires the exact experiment package versions and the original data. Requirements for analysis execution do not retroactively establish historical runtime versions.

## 1. Contemporary results without training or raw text

    python -m pip install -r requirements-audit.txt
    python audit_public_results.py contemporary_multiseed/received_export.zip --source-root contemporary_multiseed/source --destination rerun_public_audit

Expected: PASS; 30 completed neural fits, three epochs each, 15 independently refitted temperatures, 45 evaluation conditions, 32,175 prediction rows and 35 captured deprecation warnings. Compare the five generated metric/summary/history CSV files with contemporary_multiseed/audit/. The audit checks run/code/tokenizer identities and numerical outputs, using published calibration/test identities. It does not recheck raw dataset bytes or omitted weights.

## 2. Historical additions and model-conditional statistics

    python unpack_historical.py
    python analyze_revision.py
    python reproduce_historical_statistics.py --destination rerun_historical_statistics

The first two commands reproduce descriptive domain/comparator/topic-resolution additions. The third executes the original E7 statistical code with 10,000 bootstrap draws, 10,000 cluster sign flips, alpha 0.05 and RNG seed 20260919. It changes only filesystem/Drive configuration. Contemporary training seeds are excluded. Expected tables are preserved in historical_results.zip under 09_E7_STATISTICAL_INFERENCE/Results_Expanded/. Compare numerical columns using a suitable floating-point tolerance; new timestamps and archive hashes differ by design. The E6 public ZIP omits Representative_Docs while retaining all numerical inputs used by E7.

## 3. Raw-input gates and retraining (optional GPU reproduction)

Obtain the original Dreaddit CSV files as described in DATA_ACCESS.md. Preserve their exact bytes.

    python prepare_inputs.py --train /path/dreaddit-train.csv --test /path/dreaddit-test.csv
    python audit_multiseed.py contemporary_multiseed/received_export.zip --source-root contemporary_multiseed/source --destination rerun_full_audit
    python derive_lodo_indices.py --destination rerun_lodo_indices.csv

The original full audit validates the frozen raw input hashes and grouped split as well as every contemporary numerical output. The input preparer also accepts --archive /path/dreaddit.zip. It rejects a mismatching file instead of modifying the protocol or regenerating the split.

For a separately labelled replication, install contemporary_multiseed/source/requirements.txt into a CUDA environment matching the recorded runtime as closely as possible. The recorded experiment used Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.16.1, CUDA 13.0 and a Tesla T4. Full installed packages are retained in received_export.zip. Training uses the fixed final epoch, three epochs, FP32/eager attention, max length 128, learning rate 2e-5, weight decay .01 and effective batch size 16; DeBERTa uses micro-batch 8 and accumulation 2.

    python contemporary_multiseed/source/revision_runner.py data-check --output /path/new_replication
    python contemporary_multiseed/source/revision_runner.py preflight --output /path/new_replication
    python contemporary_multiseed/source/revision_runner.py train --output /path/new_replication --max-runs 2 --save-all-weights
    python contemporary_multiseed/source/revision_runner.py status --output /path/new_replication

Repeat train until all 30 prespecified fits complete, then run:

    python contemporary_multiseed/source/revision_runner.py analyze --output /path/new_replication

Use a new output directory. Preserve all failed attempts and warnings. A separately executed replication may differ across CUDA/device/runtime builds; do not overwrite archived results or treat refitted models as the original instances. The pinned contemporary source hashes authenticate this training implementation, not older missing revisions.
