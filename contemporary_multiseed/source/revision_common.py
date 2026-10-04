"""Auditable data gates and seed-level reliability metrics for IJIES 20266380."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score, accuracy_score


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_object(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def load_inputs(root, config):
    root = Path(root)
    names = [("dreaddit-train.csv", "train_sha256"),
             ("dreaddit-test.csv", "test_sha256"),
             ("development_split_assignment.csv", "split_sha256")]
    for name, key in names:
        if sha256(root / "inputs" / name) != config[key]:
            raise ValueError(f"Hash tidak cocok: {name}. Jangan membuat split baru.")
    train = pd.read_csv(root / "inputs/dreaddit-train.csv")
    test = pd.read_csv(root / "inputs/dreaddit-test.csv")
    split = pd.read_csv(root / "inputs/development_split_assignment.csv")
    assert len(train) == 2838 and len(test) == 715 and len(split) == 2838
    assert train.id.is_unique and test.id.is_unique and split.row_index.is_unique
    assert set(split.row_index) == set(range(2838))
    assert split.split.value_counts().to_dict() == {"fit": 1994, "calibration": 423, "validation": 421}
    aligned = train.iloc[split.row_index.to_numpy()].reset_index(drop=True)
    for column in ["id", "post_id", "label"]:
        assert np.array_equal(aligned[column].astype(str), split[column].astype(str)), column
    assert split.groupby("post_id").split.nunique().max() == 1
    assert not (set(train.post_id.astype(str)) & set(test.post_id.astype(str)))
    fit_index = split.loc[split.split.isin(config["train_partition"]), "row_index"].to_numpy()
    cal_index = split.loc[split.split.eq("calibration"), "row_index"].to_numpy()
    fit = train.iloc[fit_index].copy().reset_index(drop=True)
    cal = train.iloc[cal_index].copy().reset_index(drop=True)
    test = test.reset_index(drop=True)
    for frame, expected in [(fit, 2415), (cal, 423), (test, 715)]:
        assert len(frame) == expected and frame.label.isin([0, 1]).all()
        assert frame.text.notna().all()
    return fit, cal, test, split


def fit_temperature(logits, y, bounds=(-4.0, 4.0)):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    assert logits.shape == (len(y), 2) and np.isfinite(logits).all()
    def objective(logt):
        z = logits / np.exp(logt)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
    result = minimize_scalar(objective, bounds=tuple(bounds), method="bounded", options={"xatol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature fit gagal. Jangan lanjutkan ke evaluasi test.")
    return {"temperature": float(np.exp(result.x)), "optimizer_success": True,
            "calibration_nll_before": objective(0.0), "calibration_nll_after": float(result.fun),
            "log_temperature_at_boundary": bool(min(abs(result.x - bounds[0]), abs(result.x - bounds[1])) < 1e-5),
            "calibration_rows": len(y)}


def metrics(logits, y, temperature=1.0, ranking_logits=None, ece_bins=10):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    assert logits.shape == (len(y), 2) and np.isfinite(logits).all()
    assert temperature > 0 and np.isfinite(temperature)
    scaled = logits / temperature
    probability = softmax(scaled, axis=1)
    predicted = logits.argmax(axis=1)  # Positive scalar TS cannot change argmax.
    confidence = probability.max(axis=1)
    correct = (predicted == y).astype(float)
    ece = 0.0
    edges = np.linspace(0, 1, ece_bins + 1)
    for i in range(ece_bins):
        mask = (confidence >= edges[i]) & ((confidence <= edges[i+1]) if i == ece_bins-1 else (confidence < edges[i+1]))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    # Binary max probability is strictly increasing in absolute logit margin.
    # Use the unscaled margin to avoid saturation and new numerical TS ties.
    rank_z = logits if ranking_logits is None else np.asarray(ranking_logits, dtype=np.float64)
    order = np.argsort(-np.abs(rank_z[:, 1] - rank_z[:, 0]), kind="stable")
    errors = (predicted[order] != y[order]).astype(float)
    risk = np.cumsum(errors) / np.arange(1, len(y) + 1)
    high = confidence >= .90
    out = {"f1": float(f1_score(y, predicted, zero_division=0)),
           "accuracy": float(accuracy_score(y, predicted)),
           "brier": float(np.mean((probability[:, 1] - y) ** 2)),
           "nll": float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y])),
           "ece": float(ece), "aurc": float(risk.mean()),
           "hce_090": int(((predicted != y) & high).sum()), "coverage_090": float(high.mean())}
    for coverage in [.9, .8, .7, .5]:
        k = max(1, int(round(coverage * len(y))))
        out[f"risk_at_{int(coverage * 100)}"] = float(risk[k-1])
        out[f"stress_recall_at_{int(coverage * 100)}"] = float(((y[order[:k]] == 1) & (predicted[order[:k]] == 1)).sum() / max(1, (y == 1).sum()))
    return out, probability, predicted, confidence, order


def prediction_table(frame, logits, temperature, backbone, seed, condition):
    values, probs, predicted, conf, order = metrics(logits, frame.label.to_numpy(), temperature)
    rank = np.empty(len(frame), dtype=int)
    rank[order] = np.arange(1, len(frame) + 1)
    table = frame[["id", "post_id", "label"]].copy().rename(columns={"label": "true_label"})
    table.insert(0, "row_index", np.arange(len(frame)))
    table["backbone"] = backbone; table["seed"] = seed; table["condition"] = condition
    table["logit_not_stress"] = logits[:, 0]; table["logit_stress"] = logits[:, 1]
    table["temperature"] = temperature
    table["p_not_stress"] = probs[:, 0]; table["p_stress"] = probs[:, 1]
    table["predicted_label"] = predicted; table["confidence"] = conf; table["selection_rank"] = rank
    return table, values


def summarize(metric_table, expected_seeds):
    columns = ["f1", "brier", "nll", "ece", "aurc", "hce_090", "coverage_090"]
    assert not metric_table.duplicated(["backbone", "seed", "condition"]).any()
    for (_, _), group in metric_table.groupby(["backbone", "condition"]):
        assert set(group.seed.astype(int)) == set(expected_seeds)
    rows, delta_rows = [], []
    for (backbone, condition), group in metric_table.groupby(["backbone", "condition"]):
        item = {"backbone": backbone, "condition": condition, "training_seeds": len(group)}
        for metric in columns:
            item[f"{metric}_mean"] = float(group[metric].mean())
            item[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(item)
    for backbone, group in metric_table.groupby("backbone"):
        base = group[group.condition.eq("Vanilla")].set_index("seed")
        for condition in ["Vanilla+TS", "LS0.10"]:
            comparison = group[group.condition.eq(condition)].set_index("seed").loc[base.index]
            for seed in base.index:
                item = {"backbone": backbone, "seed": int(seed), "condition": condition, "reference": "Vanilla"}
                item.update({metric + "_delta": float(comparison.loc[seed, metric] - base.loc[seed, metric]) for metric in columns})
                delta_rows.append(item)
    deltas = pd.DataFrame(delta_rows)
    summary = []
    for (backbone, condition), group in deltas.groupby(["backbone", "condition"]):
        item = {"backbone": backbone, "condition": condition, "training_seeds": len(group)}
        for metric in columns:
            x = group[metric + "_delta"]
            item[metric + "_delta_mean"] = float(x.mean())
            item[metric + "_delta_sd"] = float(x.std(ddof=1))
            item[metric + "_negative_seeds"] = int((x < -1e-12).sum())
            item[metric + "_positive_seeds"] = int((x > 1e-12).sum())
            item[metric + "_zero_seeds"] = int((abs(x) <= 1e-12).sum())
        summary.append(item)
    return pd.DataFrame(rows), deltas, pd.DataFrame(summary)
