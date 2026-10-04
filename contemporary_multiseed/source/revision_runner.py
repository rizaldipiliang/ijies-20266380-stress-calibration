"""New revision runs. Never treat this script as evidence that training has occurred."""
from pathlib import Path
import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from revision_common import load_inputs, sha256, digest_object, write_json, fit_temperature, prediction_table, summarize

# Must be set before CUDA context creation, including before environment().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def now():
    return datetime.now(timezone.utc).isoformat()


def event(output, **values):
    with open(Path(output) / "run_events.jsonl", "a") as handle:
        handle.write(json.dumps({"utc": now(), **values}) + "\n")


def environment(config):
    import torch
    versions = {name: importlib.metadata.version(name) for name in config["expected_versions"]}
    for name, expected in config["expected_versions"].items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"Versi {name} = {versions[name]}, perlu {expected}. Jalankan sel instalasi, lalu restart session.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA belum aktif. Pilih runtime GPU; jangan jalankan pelatihan di CPU.")
    supporting_versions = {name: importlib.metadata.version(name) for name in ["tokenizers", "huggingface-hub", "safetensors"]}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "supporting_packages": supporting_versions, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0))}


def identity(root, config, env):
    return {"protocol_sha256": sha256(Path(root) / "protocol.json"), "environment": env,
            "common_code_sha256": sha256(Path(root) / "revision_common.py"),
            "runner_code_sha256": sha256(Path(root) / "revision_runner.py")}


def locked_write(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise RuntimeError(f"Lock berbeda: {path.name}. Jangan campur protokol, kode, atau lingkungan. Kirim laporan ini untuk pemeriksaan.")
    else:
        write_json(path, obj)


def check_gpu_models(root, output, config):
    from transformers import AutoConfig, AutoTokenizer
    env = environment(config)
    fit, cal, test, split = load_inputs(root, config)
    run_identity = identity(root, config, env)
    locked_write(output / "environment_lock.json", run_identity)
    shutil.copy2(root / "protocol.json", output / "protocol.json")
    output.joinpath("requirements_actual.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    records = {}
    for name, spec in config["model_specs"].items():
        model_config = AutoConfig.from_pretrained(spec["model_id"], revision=spec["revision"])
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"], use_fast=True)
        commit = getattr(model_config, "_commit_hash", None)
        if commit != spec["revision"]:
            raise RuntimeError(f"Commit {name} tidak terverifikasi: {commit}")
        folder = output / "model_configs" / name
        folder.mkdir(parents=True, exist_ok=True)
        model_config.save_pretrained(folder); tokenizer.save_pretrained(folder)
        records[name] = {"model_id": spec["model_id"], "requested_model_revision": spec["revision"],
                         "resolved_config_commit": commit, "requested_tokenizer_revision": spec["revision"],
                         "tokenizer_class": tokenizer.__class__.__name__, "use_fast_resolved": tokenizer.is_fast,
                         "tokenizer_files_sha256": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
    report = {"utc": now(), "data_integrity": "PASS", "train_rows": len(fit),
              "calibration_rows": len(cal), "test_rows": len(test), "test_post_clusters": int(test.post_id.nunique()),
              "split_post_overlap": 0, "official_train_test_post_overlap": 0,
              "model_config_and_tokenizer_access": "PASS", "model_weights_forward_pass": "not yet checked",
              "environment": env, "models": records, "identity_hash": digest_object(run_identity)}
    write_json(output / "PRECHECK_REPORT.json", report)
    locked_write(output / "model_identity_lock.json", records)
    print("PRECHECK PASS: 2415 train, 423 calibration, 715 test, 586 test post clusters.")
    print("Konfigurasi dan tokenizer terakses. Bobot model diperiksa pada awal setiap run.")
    print("GPU:", env["gpu"])


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train_one(root, output, config, backbone, seed, condition, env, save_weights):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    fit, cal, test, _ = load_inputs(root, config)
    spec = config["model_specs"][backbone]
    folder = output / "runs" / f"{backbone}_seed{seed}_{condition}"
    folder.mkdir(parents=True, exist_ok=True)
    ident = {**identity(root, config, env), "backbone": backbone, "seed": seed, "condition": condition}
    done = folder / "COMPLETED.json"
    if done.exists():
        record = json.loads(done.read_text())
        if record["identity"] != ident:
            raise RuntimeError("Completed run identity berbeda; cache tidak boleh digunakan.")
        for name, digest in record["files_sha256"].items():
            assert sha256(folder / name) == digest, name
        print("REUSE completed:", folder.name)
        return
    attempt = uuid.uuid4().hex
    event(output, backbone=backbone, seed=seed, condition=condition, attempt=attempt, status="started")
    write_json(folder / f"attempt_{attempt}.json", {"utc_started": now(), "identity": ident, "status": "started"})
    start = time.monotonic()
    model = optimizer = scheduler = loader = encoded = tokenizer = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_seed(seed)
            tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"], use_fast=True)
            encoded = []
            keys = None
            for frame in [fit, cal, test]:
                obj = tokenizer(frame.text.astype(str).tolist(), truncation=True, padding="max_length",
                                max_length=config["max_length"], return_tensors="pt")
                current_keys = list(obj.keys())
                if keys is None: keys = current_keys
                assert keys == current_keys
                encoded.append(TensorDataset(*[obj[k] for k in keys], torch.tensor(frame.label.to_numpy(), dtype=torch.long)))
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(encoded[0], batch_size=spec["micro_batch"], shuffle=True,
                                generator=generator, num_workers=0, pin_memory=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                spec["model_id"], revision=spec["revision"], num_labels=2,
                dtype=torch.float32, attn_implementation=config["attention_implementation"]).float().cuda()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Bobot awal tidak finite")
            model.eval()
            with torch.no_grad():
                probe = {k: encoded[0].tensors[i][:2].cuda() for i, k in enumerate(keys)}
                assert torch.isfinite(model(**probe).logits).all(), "Initial logits tidak finite"
            accumulation = spec["gradient_accumulation"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                         eps=config["adam_eps"], betas=tuple(config["adam_betas"]))
            steps = math.ceil(len(loader) / accumulation) * config["epochs"]
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=steps)
            history = []
            for epoch in range(1, config["epochs"] + 1):
                model.train(); optimizer.zero_grad(set_to_none=True); total_loss = 0.; examples = 0
                for step, batch in enumerate(loader, 1):
                    inputs = {k: batch[i].cuda(non_blocking=True) for i, k in enumerate(keys)}
                    y = batch[-1].cuda(non_blocking=True)
                    logits = model(**inputs).logits
                    if condition == "Vanilla":
                        raw_loss = torch.nn.functional.cross_entropy(logits, y)
                    else:
                        target = torch.full_like(logits, config["ls_epsilon"])
                        target.scatter_(1, y[:, None], 1.0 - config["ls_epsilon"])
                        raw_loss = -(target * torch.nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
                    if not torch.isfinite(raw_loss): raise FloatingPointError("Training loss tidak finite")
                    (raw_loss / accumulation).backward()
                    total_loss += float(raw_loss.detach()) * len(y); examples += len(y)
                    if step % accumulation == 0 or step == len(loader):
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                history.append({"epoch": epoch, "training_loss": total_loss / examples, "training_examples": examples})
                pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
                print(f"{backbone} seed {seed} {condition} epoch {epoch}/{config['epochs']} loss {total_loss / examples:.6f}", flush=True)
            model.eval()
            arrays = []
            with torch.no_grad():
                for ds in encoded[1:]:
                    parts = []
                    for batch in DataLoader(ds, batch_size=spec["micro_batch"], shuffle=False, num_workers=0):
                        inputs = {k: batch[i].cuda() for i, k in enumerate(keys)}
                        z = model(**inputs).logits.detach().cpu().numpy()
                        if not np.isfinite(z).all(): raise FloatingPointError("Evaluation logits tidak finite")
                        parts.append(z)
                    arrays.append(np.concatenate(parts))
            temp = folder / "logits.tmp.npz"
            np.savez_compressed(temp, cal_logits=arrays[0], test_logits=arrays[1],
                                cal_y=cal.label.to_numpy(), test_y=test.label.to_numpy(),
                                cal_id=cal.id.to_numpy(dtype=str), test_id=test.id.to_numpy(dtype=str),
                                cal_post_id=cal.post_id.to_numpy(dtype=str), test_post_id=test.post_id.to_numpy(dtype=str))
            temp.replace(folder / "logits.npz")
            temperature = fit_temperature(arrays[0], cal.label.to_numpy(), config["temperature_log_bounds"]) if condition == "Vanilla" else None
            write_json(folder / "calibration_fit.json", temperature or {"temperature_fit": "not applicable to LS condition"})
            if save_weights or (backbone == "BERT" and seed == 42):
                model.save_pretrained(folder / "model_weights", safe_serialization=True)
                tokenizer.save_pretrained(folder / "model_weights")
            write_json(folder / "warnings.json", [str(w.message) for w in caught])
            files = {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob("*") if p.is_file() and p.name != "COMPLETED.json"}
            write_json(done, {"utc_completed": now(), "identity": ident, "files_sha256": files,
                              "seconds": time.monotonic() - start, "model_weights_saved": (folder / "model_weights").exists()})
            event(output, backbone=backbone, seed=seed, condition=condition, attempt=attempt, status="completed")
            print("COMPLETED", folder.name, "Test metrics belum ditampilkan.", flush=True)
    except Exception as exc:
        write_json(folder / f"failure_{attempt}.json", {"utc": now(), "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        event(output, backbone=backbone, seed=seed, condition=condition, attempt=attempt, status="failed", error_type=type(exc).__name__)
        raise
    finally:
        model = optimizer = scheduler = loader = encoded = tokenizer = None
        gc.collect(); torch.cuda.empty_cache()


def progress(output, config):
    rows = []
    for backbone in config["model_specs"]:
        for seed in config["seeds"]:
            for condition in config["neural_training_conditions"]:
                folder = output / "runs" / f"{backbone}_seed{seed}_{condition}"
                done = folder / "COMPLETED.json"
                state = "completed" if done.exists() else "failed_or_interrupted" if folder.exists() else "pending"
                rows.append({"backbone": backbone, "seed": seed, "condition": condition, "status": state})
    table = pd.DataFrame(rows); table.to_csv(output / "run_status.csv", index=False)
    print(table.to_string(index=False)); print("Completed:", int(table.status.eq("completed").sum()), "/ 30")
    return table


def analyze(root, output, config):
    state = progress(output, config)
    if not state.status.eq("completed").all():
        raise RuntimeError("Analisis final diblokir sampai semua 30 neural fits selesai. Jangan memilih hanya seed yang hasilnya baik.")
    fit, cal, test, _ = load_inputs(root, config)
    env_lock = json.loads((output / "environment_lock.json").read_text())
    tables, metric_rows, temp_rows = [], [], []
    for backbone in config["model_specs"]:
        for seed in config["seeds"]:
            for base_condition in config["neural_training_conditions"]:
                folder = output / "runs" / f"{backbone}_seed{seed}_{base_condition}"
                done = json.loads((folder / "COMPLETED.json").read_text())
                assert {k: done["identity"][k] for k in env_lock} == env_lock
                for name, digest in done["files_sha256"].items(): assert sha256(folder / name) == digest, name
                with np.load(folder / "logits.npz", allow_pickle=False) as z:
                    assert np.array_equal(z["test_id"], test.id.astype(str)) and np.array_equal(z["test_y"], test.label)
                    assert np.array_equal(z["cal_id"], cal.id.astype(str)) and np.array_equal(z["cal_y"], cal.label)
                    assert np.array_equal(z["test_post_id"], test.post_id.astype(str))
                    modes = [(base_condition, 1.0)]
                    if base_condition == "Vanilla":
                        fitting = json.loads((folder / "calibration_fit.json").read_text())
                        temp_rows.append({"backbone": backbone, "seed": seed, **fitting})
                        modes.append(("Vanilla+TS", fitting["temperature"]))
                    for condition, temperature in modes:
                        table, values = prediction_table(test, z["test_logits"], temperature, backbone, seed, condition)
                        tables.append(table); metric_rows.append({"backbone": backbone, "seed": seed, "condition": condition, **values})
                        if base_condition == "Vanilla" and condition == "Vanilla": vanilla_table, vanilla_values = table, values
                        if condition == "Vanilla+TS":
                            assert np.array_equal(table.predicted_label, vanilla_table.predicted_label)
                            assert np.array_equal(table.selection_rank, vanilla_table.selection_rank)
                            assert values["f1"] == vanilla_values["f1"] and values["aurc"] == vanilla_values["aurc"]
    result = output / "analysis"; result.mkdir(exist_ok=True)
    long_metrics = pd.DataFrame(metric_rows)
    means, deltas, delta_summary = summarize(long_metrics, config["seeds"])
    assert len(long_metrics) == 45 and sum(len(x) for x in tables) == 32175
    for table, name in [(long_metrics, "metrics_by_seed.csv"), (means, "metrics_mean_sd.csv"),
                        (deltas, "paired_deltas_by_seed.csv"), (delta_summary, "paired_delta_summary.csv"),
                        (pd.DataFrame(temp_rows), "temperature_by_seed.csv"), (pd.concat(tables), "test_predictions_long.csv")]:
        table.to_csv(result / name, index=False)
    write_json(result / "ANALYSIS_REPORT.json", {"utc": now(), "neural_fits": 30, "conditions": 45,
               "prediction_rows": 32175, "seed_level_sd": "sample SD ddof=1", "uncertainty_scope": "training seed dispersion on one frozen test set",
               "bootstrap_confidence_intervals": "not computed here; original cluster intervals remain a separate result",
               "significance_tests": "none added", "historical_main_models": "not pooled into these new runs",
               "nll_evaluation": "stable logsumexp on float64 logits", "aurc_order": "stable absolute binary logit margin; TS inherits Vanilla order",
               "TS_decision_and_rank_invariance": "PASS for every seed and backbone"})
    print("ANALYSIS COMPLETE: 45 conditions, 32175 test prediction rows.")
    print(means[["backbone", "condition", "training_seeds", "brier_mean", "nll_mean", "f1_mean", "aurc_mean"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["data-check", "preflight", "train", "status", "analyze"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-runs", type=int, default=2)
    parser.add_argument("--save-all-weights", action="store_true")
    args = parser.parse_args(); config = json.loads((args.root / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "data-check":
        fit, cal, test, split = load_inputs(args.root, config)
        print("DATA CHECK PASS", len(fit), len(cal), len(test), test.post_id.nunique()); return
    if args.action == "preflight": check_gpu_models(args.root, args.output, config); return
    if args.action == "status": progress(args.output, config); return
    if args.action == "analyze": analyze(args.root, args.output, config); return
    if not (args.output / "PRECHECK_REPORT.json").exists(): raise RuntimeError("Jalankan preflight terlebih dahulu")
    env = environment(config)
    locked_write(args.output / "environment_lock.json", identity(args.root, config, env))
    backbones = args.backbones or list(config["model_specs"])
    seeds = args.seeds or config["seeds"]
    assert set(backbones) <= set(config["model_specs"]) and set(seeds) <= set(config["seeds"])
    count = 0
    for backbone in backbones:
        for seed in seeds:
            for condition in config["neural_training_conditions"]:
                done = args.output / "runs" / f"{backbone}_seed{seed}_{condition}" / "COMPLETED.json"
                was_complete = done.exists()
                if not was_complete and count >= args.max_runs: progress(args.output, config); return
                train_one(args.root, args.output, config, backbone, seed, condition, env, args.save_all_weights)
                if not done.exists(): raise RuntimeError("Completed file missing")
                if not was_complete: count += 1
    progress(args.output, config)


if __name__ == "__main__":
    main()
