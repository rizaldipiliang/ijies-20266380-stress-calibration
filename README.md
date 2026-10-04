# Calibration and selective reliability under domain shift

Reproducibility materials for IJIES manuscript **20266380**:
**Separating Probability Calibration from Selective Reliability in Transformer-Based Stress Detection under Domain Shift**.

Repository: https://github.com/rizaldipiliang/ijies-20266380-stress-calibration

The revision evaluates BERT, RoBERTa and DeBERTa on Dreaddit using five prespecified seeds (13, 21, 42, 87, 100). There are 30 neural fits (Vanilla and LS0.10), 15 calibration-only temperature-scaling fits, and 45 evaluation conditions. The official test set contains 715 examples from 586 post clusters. Archived July and September analyses are supplied separately.

## Start here

Install the CPU audit dependencies and recompute the contemporary metrics:

    python -m pip install -r requirements-audit.txt
    python audit_public_results.py contemporary_multiseed/received_export.zip --source-root contemporary_multiseed/source --destination rerun_public_audit

This does not train models and does not require raw Reddit text. It verifies the exported logits, predictions, temperatures, training histories, model/tokenizer files, source-code hashes, and seed summaries. The published identity tables are compared with every logit record. Raw input CSV bytes are checked separately when the original data are supplied.

## Materials

| Path | Contents |
| --- | --- |
| contemporary_multiseed/source/ | Exact pinned training scripts, protocol, requirements, frozen split and identity-only tables |
| contemporary_multiseed/received_export.zip | Complete original 203-file results export, byte-identical |
| contemporary_multiseed/audit/ | Independent per-seed metrics, mean/SD, paired differences, training histories and original audit report |
| historical_results.zip | Stage-specific predictions, logits, temperatures, topic assignments/alignment, configurations, statistical tables and figures |
| historical_notebooks/ | Original July/September source notebooks; outputs and personal execution metadata cleared |
| indices/ | Official row identities and reconstructed, independently checked LODO indices |
| audit_multiseed.py | Original full audit, including raw input checksum validation |
| audit_public_results.py | Results-only audit with an explicitly narrower data-validation scope |
| reproduce_historical_statistics.py | Original E7 bootstrap/sign-flip cells, with filesystem paths adapted |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [DATA_ACCESS.md](DATA_ACCESS.md) and [PROVENANCE.md](PROVENANCE.md).

## Interpretation and scope

TS preserves F1 and within-model AURC in all 15 contemporary pairs. TS and LS improve Brier, NLL and ECE in every paired backbone/seed comparison. LS worsens AURC in all five RoBERTa seeds; BERT and DeBERTa directions are mixed. Reported seed variability is sample SD on one fixed test set, not a confidence interval. The current replication reuses an already evaluated benchmark; it is not a new external holdout.

The original LODO, semantic and Family A/B statistical analyses refer to their archived model instances. They are not pooled with contemporary seeds. Historical E6 HDBSCAN version and fresh E2 BERT/tokenizer commit were not recovered. Neural checkpoint bytes are not included or claimed to be publicly available. These limits are documented in PROVENANCE.md.

Raw Dreaddit text and representative Reddit documents are obtained from the original dataset source rather than redistributed here. Numerical columns are retained. Author correspondence, author-confirmation forms and unpublished submission documents are excluded. No manuscript DOI or archival DOI is claimed.
