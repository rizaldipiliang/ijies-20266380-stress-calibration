"""Independently audit the IJIES R1 experiment export without training models.

The test endpoints are calculated only for the complete prespecified 30-fit
matrix. Neural weights are intentionally absent from the export and are not
claimed to have been checked here.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

CORE = ["f1", "brier", "nll", "ece", "aurc", "hce_090", "coverage_090"]


def independent_metrics(logits, labels, temperature):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    scaled = z / temperature
    normalizer = np.logaddexp(scaled[:, 0], scaled[:, 1])
    p = np.column_stack([np.exp(scaled[:, k] - normalizer) for k in (0, 1)])
    pred = (z[:, 1] > z[:, 0]).astype(int)
    confidence = p.max(axis=1)
    errors = (pred != y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lower) & ((confidence <= upper) if upper == 1 else (confidence < upper))
        if mask.any():
            ece += float(mask.sum() / len(y) * abs((~errors[mask]).mean() - confidence[mask].mean()))
    margin = np.abs(z[:, 1] - z[:, 0])
    order = np.array(sorted(range(len(y)), key=lambda i: (-margin[i], i)))
    risk = np.cumsum(errors[order]) / np.arange(1, len(y) + 1)
    result = {
        "f1": f1,
        "accuracy": float((~errors).mean()),
        "brier": float(((p[:, 1] - y) ** 2).mean()),
        "nll": float((normalizer - scaled[np.arange(len(y)), y]).mean()),
        "ece": ece,
        "aurc": float(risk.mean()),
        "hce_090": int((errors & (confidence >= .90)).sum()),
        "coverage_090": float((confidence >= .90).mean()),
    }
    for coverage in [.9, .8, .7, .5]:
        k = max(1, round(coverage * len(y)))
        accepted = order[:k]
        result[f"risk_at_{int(coverage * 100)}"] = float(risk[k - 1])
        result[f"stress_recall_at_{int(coverage * 100)}"] = float(((y[accepted] == 1) & (pred[accepted] == 1)).sum() / max(1, (y == 1).sum()))
    ranks = np.empty(len(y), dtype=int)
    ranks[order] = np.arange(1, len(y) + 1)
    return result, p, pred, confidence, ranks


def audit(archive_path, source_root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_root.resolve()))
    from revision_common import load_inputs
    digest = lambda value: hashlib.sha256(value).hexdigest()
    checked_hashes = 0
    omitted_weights = []
    histories = []
    captured_warnings = []
    metrics = []
    temperature_rows = []
    prediction_tables = []
    max_metric_error = 0.0
    max_prediction_error = 0.0
    with zipfile.ZipFile(archive_path) as z:
        assert z.testzip() is None, "ZIP CRC failed"
        names = set(z.namelist())
        protocol = json.loads(z.read("protocol.json"))
        env = json.loads(z.read("environment_lock.json"))
        model_lock = json.loads(z.read("model_identity_lock.json"))
        preflight = json.loads(z.read("PRECHECK_REPORT.json"))
        assert z.read("protocol.json") == (source_root / "protocol.json").read_bytes()
        for filename, key in [("protocol.json", "protocol_sha256"), ("revision_common.py", "common_code_sha256"), ("revision_runner.py", "runner_code_sha256")]:
            assert digest((source_root / filename).read_bytes()) == env[key], filename
        assert preflight["models"] == model_lock
        assert preflight["environment"] == env["environment"]
        assert preflight["identity_hash"] == digest(json.dumps(env, sort_keys=True).encode())
        for backbone, record in model_lock.items():
            assert record["requested_model_revision"] == record["resolved_config_commit"] == protocol["model_specs"][backbone]["revision"]
            for filename, expected in record["tokenizer_files_sha256"].items():
                assert digest(z.read(f"model_configs/{backbone}/{filename}")) == expected
        _, cal, test, _ = load_inputs(source_root, protocol)
        status = pd.read_csv(io.BytesIO(z.read("run_status.csv")))
        expected = {(b, s, c) for b in protocol["model_specs"] for s in protocol["seeds"] for c in protocol["neural_training_conditions"]}
        assert len(status) == 30 and status.status.eq("completed").all()
        assert set(map(tuple, status[["backbone", "seed", "condition"]].to_numpy())) == expected
        assert not status.duplicated(["backbone", "seed", "condition"]).any()
        assert not any("failure_" in name for name in names)
        events = [json.loads(line) for line in z.read("run_events.jsonl").decode().splitlines() if line]
        assert len(events) == 60 and sum(e["status"] == "completed" for e in events) == 30
        assert all(e["status"] in ("started", "completed") for e in events)
        for row in status.itertuples(index=False):
            folder = f"runs/{row.backbone}_seed{row.seed}_{row.condition}/"
            done = json.loads(z.read(folder + "COMPLETED.json"))
            assert done["identity"] == {**env, "backbone": row.backbone, "seed": row.seed, "condition": row.condition}
            run_events = [e for e in events if (e["backbone"], e["seed"], e["condition"]) == (row.backbone, row.seed, row.condition)]
            assert [e["status"] for e in run_events] == ["started", "completed"]
            assert run_events[0]["attempt"] == run_events[1]["attempt"]
            for filename, expected_digest in done["files_sha256"].items():
                if filename.startswith("model_weights/"):
                    assert folder + filename not in names
                    omitted_weights.append(folder + filename)
                else:
                    assert digest(z.read(folder + filename)) == expected_digest, folder + filename
                    checked_hashes += 1
            history = pd.read_csv(io.BytesIO(z.read(folder + "training_history.csv")))
            assert history.epoch.tolist() == [1, 2, 3]
            assert history.training_examples.eq(2415).all()
            assert np.isfinite(history.training_loss).all() and history.training_loss.ge(0).all()
            history.insert(0, "condition", row.condition)
            history.insert(0, "seed", row.seed)
            history.insert(0, "backbone", row.backbone)
            histories.append(history)
            warnings = json.loads(z.read(folder + "warnings.json"))
            captured_warnings.extend(warnings)
            with np.load(io.BytesIO(z.read(folder + "logits.npz")), allow_pickle=False) as arrays:
                for prefix, frame in [("cal", cal), ("test", test)]:
                    assert arrays[prefix + "_logits"].shape == (len(frame), 2)
                    assert np.isfinite(arrays[prefix + "_logits"]).all()
                    for suffix, column in [("y", "label"), ("id", "id"), ("post_id", "post_id")]:
                        expected_values = frame[column].to_numpy() if suffix == "y" else frame[column].to_numpy(dtype=str)
                        assert np.array_equal(arrays[prefix + "_" + suffix], expected_values), (folder, prefix, column)
                modes = [(row.condition, 1.0)]
                fitting = json.loads(z.read(folder + "calibration_fit.json"))
                if row.condition == "Vanilla":
                    cal_logits = arrays["cal_logits"].astype(np.float64)
                    cal_y = arrays["cal_y"].astype(int)
                    def objective(logt):
                        a = cal_logits / np.exp(logt)
                        return float((np.logaddexp(a[:, 0], a[:, 1]) - a[np.arange(len(cal_y)), cal_y]).mean())
                    result = minimize_scalar(objective, bounds=tuple(protocol["temperature_log_bounds"]), method="bounded", options={"xatol": 1e-10})
                    assert result.success and fitting["optimizer_success"] and not fitting["log_temperature_at_boundary"]
                    assert np.isclose(np.exp(result.x), fitting["temperature"], atol=1e-6, rtol=1e-6)
                    assert abs(objective(0) - fitting["calibration_nll_before"]) < 1e-12
                    assert abs(objective(np.log(fitting["temperature"])) - fitting["calibration_nll_after"]) < 1e-12
                    assert fitting["calibration_rows"] == 423
                    temperature_rows.append({"backbone": row.backbone, "seed": row.seed, **fitting})
                    modes.append(("Vanilla+TS", fitting["temperature"]))
                else:
                    assert fitting == {"temperature_fit": "not applicable to LS condition"}
                for condition, temperature in modes:
                    values, probs, pred, conf, ranks = independent_metrics(arrays["test_logits"], arrays["test_y"], temperature)
                    metrics.append({"backbone": row.backbone, "seed": row.seed, "condition": condition, **values})
                    table = pd.DataFrame({"row_index": np.arange(len(test)), "id": arrays["test_id"], "post_id": arrays["test_post_id"], "true_label": arrays["test_y"], "backbone": row.backbone, "seed": row.seed, "condition": condition, "logit_not_stress": arrays["test_logits"][:, 0], "logit_stress": arrays["test_logits"][:, 1], "temperature": temperature, "p_not_stress": probs[:, 0], "p_stress": probs[:, 1], "predicted_label": pred, "confidence": conf, "selection_rank": ranks})
                    prediction_tables.append(table)
                    if condition == "Vanilla":
                        vanilla_values, vanilla_pred, vanilla_ranks = values, pred.copy(), ranks.copy()
                    if condition == "Vanilla+TS":
                        assert np.array_equal(pred, vanilla_pred) and np.array_equal(ranks, vanilla_ranks)
                        assert values["f1"] == vanilla_values["f1"] and values["aurc"] == vanilla_values["aurc"]
        independent = pd.DataFrame(metrics)
        archived = pd.read_csv(io.BytesIO(z.read("analysis/metrics_by_seed.csv")))
        keys = ["backbone", "seed", "condition"]
        a = independent.set_index(keys).sort_index()
        b = archived.set_index(keys).sort_index()
        assert len(a) == 45 and a.index.equals(b.index)
        for column in b.columns:
            error = float(np.abs(a[column] - b[column]).max())
            max_metric_error = max(max_metric_error, error)
            assert error < 1e-12, (column, error)
        grouped = independent.groupby(["backbone", "condition"])
        means = []
        paired = []
        for (backbone, condition), group in grouped:
            assert set(group.seed) == set(protocol["seeds"])
            record = {"backbone": backbone, "condition": condition, "training_seeds": len(group)}
            for metric in CORE:
                record[metric + "_mean"] = float(np.mean(group[metric]))
                record[metric + "_sd"] = float(np.std(group[metric], ddof=1))
            means.append(record)
            if condition != "Vanilla":
                reference = independent[(independent.backbone == backbone) & (independent.condition == "Vanilla")].set_index("seed")
                for item in group.itertuples(index=False):
                    record = {"backbone": backbone, "seed": item.seed, "condition": condition, "reference": "Vanilla"}
                    for metric in CORE:
                        record[metric + "_delta"] = float(getattr(item, metric) - reference.loc[item.seed, metric])
                    paired.append(record)
        means = pd.DataFrame(means)
        paired = pd.DataFrame(paired)
        summaries = []
        for (backbone, condition), group in paired.groupby(["backbone", "condition"]):
            record = {"backbone": backbone, "condition": condition, "training_seeds": len(group)}
            for metric in CORE:
                values = group[metric + "_delta"].to_numpy()
                record[metric + "_delta_mean"] = float(np.mean(values))
                record[metric + "_delta_sd"] = float(np.std(values, ddof=1))
                record[metric + "_negative_seeds"] = int((values < -1e-12).sum())
                record[metric + "_positive_seeds"] = int((values > 1e-12).sum())
                record[metric + "_zero_seeds"] = int((np.abs(values) <= 1e-12).sum())
            summaries.append(record)
        summary = pd.DataFrame(summaries)
        for current, filename, keys in [(means, "metrics_mean_sd.csv", ["backbone", "condition"]), (paired, "paired_deltas_by_seed.csv", ["backbone", "seed", "condition"]), (summary, "paired_delta_summary.csv", ["backbone", "condition"])]:
            archived = pd.read_csv(io.BytesIO(z.read("analysis/" + filename)))
            left = current.set_index(keys).sort_index(); right = archived.set_index(keys).sort_index()
            assert left.index.equals(right.index)
            for column in right.columns:
                if column == "reference":
                    assert left[column].equals(right[column])
                else:
                    assert np.allclose(left[column], right[column], atol=1e-12, rtol=0), (filename, column)
        reference_predictions = pd.read_csv(io.BytesIO(z.read("analysis/test_predictions_long.csv")), dtype={"id": str, "post_id": str})
        computed_predictions = pd.concat(prediction_tables, ignore_index=True)
        assert len(reference_predictions) == len(computed_predictions) == 32175
        for column in reference_predictions:
            if column in ["id", "post_id", "backbone", "condition"]:
                assert np.array_equal(reference_predictions[column].astype(str), computed_predictions[column].astype(str)), column
            elif column.startswith("logit_"):
                # The CSV writes FP32 logits as decimal strings. Restore their
                # original dtype to check a faithful round trip.
                assert np.array_equal(reference_predictions[column].to_numpy(dtype=np.float32), computed_predictions[column].to_numpy(dtype=np.float32)), column
            else:
                error = float(np.abs(reference_predictions[column] - computed_predictions[column]).max())
                max_prediction_error = max(max_prediction_error, error)
                assert error < 1e-12, (column, error)
        # Refit results in the summary must be the exact per-run calibration fit.
        current_t = pd.DataFrame(temperature_rows).set_index(["backbone", "seed"]).sort_index()
        archived_t = pd.read_csv(io.BytesIO(z.read("analysis/temperature_by_seed.csv"))).set_index(["backbone", "seed"]).sort_index()
        for column in archived_t:
            assert np.allclose(current_t[column], archived_t[column], atol=1e-12, rtol=0), column
        for member in z.infolist():
            target = (destination / "received" / member.filename).resolve()
            assert target.is_relative_to((destination / "received").resolve())
        z.extractall(destination / "received")
        report = json.loads(z.read("analysis/ANALYSIS_REPORT.json"))
        assert (report["neural_fits"], report["conditions"], report["prediction_rows"]) == (30, 45, 32175)
    independent.to_csv(destination / "independent_metrics_by_seed.csv", index=False)
    means.to_csv(destination / "independent_mean_sd.csv", index=False)
    paired.to_csv(destination / "independent_paired_deltas.csv", index=False)
    summary.to_csv(destination / "independent_paired_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(destination / "all_training_histories.csv", index=False)
    result = {"audit": "PASS", "archive_sha256": digest(archive_path.read_bytes()), "neural_fits": 30, "training_epochs_per_fit": 3, "train_rows_each_epoch": 2415, "calibration_rows": 423, "test_rows": 715, "test_post_clusters": 586, "evaluation_conditions": 45, "prediction_rows": 32175, "exported_run_file_hashes_verified": checked_hashes, "model_config_and_tokenizer_file_hashes_verified": 9, "all_run_identities_match_environment_and_protocol": True, "all_logit_rows_ids_labels_and_post_ids_match_frozen_inputs": True, "all_losses_and_logits_finite": True, "training_event_counts": {"started": 30, "completed": 30, "failed": 0}, "captured_run_warnings": len(captured_warnings), "warning_message_counts": dict(Counter(captured_warnings)), "temperatures_independently_refitted": 15, "TS_decisions_and_selection_ranks_invariant": True, "all_exported_metrics_and_summaries_reproduced": True, "CSV_logit_FP32_roundtrip": "PASS", "max_metric_absolute_difference": max_metric_error, "max_derived_prediction_absolute_difference": max_prediction_error, "weights_omitted_from_export": len(omitted_weights), "weight_verification_scope": "Weights remain in Colab Drive; omitted weight-file bytes were not audited here.", "uncertainty_scope": "Sample SD across five training seeds on one fixed test set, not a bootstrap CI or external population uncertainty.", "additional_significance_tests": "None", "environment": env["environment"]}
    (destination / "AUDIT_REPORT.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.source_root, args.destination)
