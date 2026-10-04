# Original data and frozen indices

Dreaddit was released by Elsbeth Turcan and Kathy McKeown. Use the original authors' distribution and its accompanying usage information:

- Paper and citation: https://aclanthology.org/D19-6213/
- Authors' dataset link: https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip
- Dataset source page: https://www.cs.columbia.edu/~eturcan/

This repository does not assert a new license over Dreaddit. Raw Reddit text, validation-text columns and BERTopic Representative_Docs are omitted. IDs and post IDs are dataset identifiers, not a claim of anonymity. Derived predictions, numerical metrics, topic assignments, word summaries and row indices are included for research verification.

## Exact input files

| File | Rows | SHA-256 |
| --- | --- | --- |
| dreaddit-train.csv | 2838 | 4e791e60d3f04dd9f14abb1228b0d4f92bf628cc84097c1c173401e98d355f4e |
| dreaddit-test.csv | 715 | a3b6c9bf5d685f49d57e05d24266dddc52999843256b13e65c07267c543175fe |
| development_split_assignment.csv | 2838 | 843946692fbc72b189549174e55ab561742052b92dcefabd2732721795df1daf |

Stage the original data using prepare_inputs.py. The frozen split comprises fit=1994, validation=421 and calibration=423, grouped by post_id. The contemporary train set combines fit and validation (2415 rows); the calibration set is held apart. Official train/test post IDs do not overlap.

If the original distribution is unavailable or its serialization differs, retain the published hashes and resolve the source-file discrepancy. Do not silently alter the frozen data gate. Numerical results can still be checked using audit_public_results.py without the raw text. Input preparation was tested against the supplied, hash-matching research CSV files; the remote distribution's current bytes are not claimed to have been downloaded and compared here.

indices/lodo_indices.csv reconstructs the September LODO row indices from the frozen split and archived domain taxonomy. Its companion JSON documents agreement with all five original fold manifests and all ten saved fold calibration/target/source label arrays. This is a verified reconstruction, explicitly distinguished from an index file saved during the historical training. Recreate it with derive_lodo_indices.py after staging raw inputs.
