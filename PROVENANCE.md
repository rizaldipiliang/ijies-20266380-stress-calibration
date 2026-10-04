# Provenance and limitations

The release separates three experimental stages:

1. July canonical BERT validation selection and historical paired seeds. Selection ranked six predefined candidates by validation ECE, then Brier, then F1. LS epsilon .10 was recorded at 2026-07-28T11:00:28Z; the retained official-test lock is later, at 11:18:51Z. Records and source order support the documented workflow but do not independently establish every earlier execution or every investigator's prior knowledge.
2. September fresh BERT/calibration, cross-backbone, LODO, selective prediction, semantic topic and E7 statistical analyses. E7 inference remains conditional on these fitted model instances and post clusters.
3. Contemporary revision protocol IJIES_20266380_R1_multiseed_v1: three backbones, five paired seeds, 30 neural fits and 15 temperature fits. Its code, model/tokenizer commits, runtime, logits and numerical summaries are frozen and independently audited.

## Known historical gaps

- The original fresh E2 BERT/tokenizer commit was not recorded or recovered. The later E4 BERT lock and contemporary pinned BERT commit do not retroactively identify that E2 model.
- Historical E6 recorded HDBSCAN as unknown, and its installation was not pinned. The actual original distribution version cannot be inferred from that field. Other retained package/embedding records are supplied as recorded.
- Historical attention backend and complete execution history were not fully recorded. Source notebooks alone do not prove that every source variant was run.
- The LODO index CSV in this release is a checked reconstruction from the archived split/taxonomy/code and saved label arrays, not a newly claimed original training log.

## Contemporary model commits

| Backbone | Model/tokenizer revision |
| --- | --- |
| BERT | 86b5e0934494bd15c9632b12f734a8a67f723594 |
| RoBERTa | e2da8e2f811d1448a5b465c236feacd80ffbac7b |
| DeBERTa | 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 |

All contemporary metrics were independently recomputed from the complete export. Thirty fits completed three epochs, with finite losses and logits. The 35 captured torch.jit.script API deprecation warnings are retained. Five-seed sample SD is not a confidence interval, a hierarchical bootstrap or a new significance family. The same already evaluated 715-example test set is reused.

## Public curation and checksum scope

The contemporary received_export.zip is byte-identical to the original export (SHA-256 fccebde75884f34cbe690041a072a505f64cad58bb852990e2282059609bc622). Training scripts and protocol are unchanged. Historical notebook code cells are unchanged, while saved outputs and personal execution metadata are cleared. SOURCE_PROVENANCE.json records code hashes and byte relationships.

Raw-text CSV columns and BERTopic representative documents were removed. PUBLIC_CURATION.json describes each transformed artifact. Archived historical checksum/manifest files describe their original packages; they must not be interpreted as authenticating a transformed public CSV or curated E6 ZIP. SHA256SUMS.csv authenticates the published payloads. Retained numerical fields are unchanged, and public recomputation checks are recorded separately.

Neural checkpoint bytes are absent. The original export lists eight omitted BERT seed-42 checkpoint files; their bytes were not audited here. Model/tokenizer configuration files are included. No weight-download URL, final journal acceptance, manuscript DOI or archival DOI is claimed.
