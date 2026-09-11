"""Small, exploratory fusion follow-up using cached features and grouped CV.

Selection uses only the original train + validation participants. The historical
test has already been inspected in previous experiments and is explicitly a
follow-up comparison, never a new untouched test. Existing artifacts are kept.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from src.evaluation.recruitview_final_evaluation import _group_splits
from src.models.model1_protocol import DIRECT_TARGET_COLUMNS
from src.models.train_model1_feature_fusion import _load_numeric_table, _read, _targets
from src.data.extract_recruitview_llm_features import FEATURE_COLUMNS


@dataclass(frozen=True)
class Config:
    version: str = "gated-v1"
    folds: int = 3
    seeds: tuple[int, ...] = (17, 43, 89)
    checkpoints: tuple[int, ...] = (50, 100, 150)
    hidden: int = 16
    lr: float = 0.002
    weight_decay: float = 0.01
    dropout: float = 0.1
    ranking_weight: float = 0.1
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    bootstrap_resamples: int = 1000


VARIANTS = ("fixed_mse", "gated_mse", "gated_huber", "gated_huber_rank")


class GatedFusion(nn.Module):
    """Equal-width text/audio projections with optional per-response gating."""

    def __init__(self, text_dim: int, audio_dim: int, config: Config, gated: bool = True):
        super().__init__()
        h = config.hidden
        self.text = nn.Sequential(nn.Linear(text_dim, h), nn.Tanh())
        self.audio = nn.Sequential(nn.Linear(audio_dim, h), nn.Tanh())
        self.gate = nn.Linear(2 * h, 2) if gated else None
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 6))

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        t, a = self.text(text), self.audio(audio)
        weights = (torch.softmax(self.gate(torch.cat((t, a), dim=1)), dim=1)
                   if self.gate is not None else t.new_full((len(t), 2), 0.5))
        fused = weights[:, :1] * t + weights[:, 1:] * a
        return self.head(fused), weights


def training_loss(pred, target, variant, ranking_weight=0.1):
    if variant.endswith("mse"):
        return F.mse_loss(pred, target)
    loss = F.huber_loss(pred, target, delta=1.0)
    if variant.endswith("rank"):
        # Pairs are generated inside the training batch only; ties are excluded.
        other = torch.randperm(len(target), device=target.device)
        sign = torch.sign(target - target[other])
        mask = sign != 0
        if mask.any():
            loss = loss + ranking_weight * F.softplus(-sign[mask] * (pred - pred[other])[mask]).mean()
    return loss


def correlations(y, p):
    a, b = rankdata(y, axis=0), rankdata(p, axis=0)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), denominator, out=np.full(y.shape[1], np.nan), where=denominator > 0)


def metrics(y, p):
    r = correlations(y, p)
    if not np.isfinite(r).all():
        raise ValueError("Undefined correlation: constant predictions or targets.")
    error = y - p
    return {"big_five_spearman": float(r[:5].mean()), "speaking_spearman": float(r[5]),
            "per_target": {name: {"spearman": float(r[i]), "mae": float(np.abs(error[:, i]).mean()),
                                   "rmse": float(np.sqrt((error[:, i] ** 2).mean()))}
                           for i, name in enumerate(DIRECT_TARGET_COLUMNS)}}


def split_indices(rows):
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate prepared record IDs.")
    sets = {s: {r["participant_id"] for r in rows if r["split"] == s} for s in ("train", "validation", "test")}
    if any(not s for s in sets.values()) or any(r["split"] not in sets for r in rows):
        raise ValueError("Expected nonempty train, validation and test splits.")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("Participant leakage across original splits.")
    return (np.array([i for i, r in enumerate(rows) if r["split"] != "test"]),
            np.array([i for i, r in enumerate(rows) if r["split"] == "test"]))


def fit_neural(text, audio, y, train, evaluation, variant, config, seed, epochs):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    scalers = [StandardScaler().fit(x[train]) for x in (text, audio, y)]
    t, a, target = [torch.tensor(s.transform(x[train]), dtype=torch.float32) for s, x in zip(scalers, (text, audio, y))]
    et, ea = [torch.tensor(s.transform(x[evaluation]), dtype=torch.float32) for s, x in zip(scalers, (text, audio))]
    model = GatedFusion(text.shape[1], audio.shape[1], config, gated=variant != "fixed_mse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    results = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        prediction, _ = model(t, a)
        loss = training_loss(prediction, target, variant, config.ranking_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.no_grad():
                pred, weights = model(et, ea)
            results[epoch] = {"prediction": scalers[2].inverse_transform(pred.numpy()), "gates": weights.numpy(),
                              "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    return results, scalers


def group_bootstrap(y, prediction, baseline, groups, resamples):
    rng = np.random.default_rng(20260911)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    values = []
    for _ in range(resamples):
        indices = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        current, reference = correlations(y[indices], prediction[indices]), correlations(y[indices], baseline[indices])
        values.append([current[:5].mean(), current[5], (current[:5] - reference[:5]).mean(), current[5] - reference[5]])
    bounds = np.nanpercentile(values, [2.5, 97.5], axis=0)
    return {key: bounds[:, i].tolist() for i, key in enumerate(("big_five", "speaking", "big_five_delta_vs_ridge", "speaking_delta_vs_ridge"))}


def predict_artifact(path, text, audio):
    """Predict with saved preprocessing and all fixed-seed ensemble members."""
    artifact = joblib.load(path)
    config = Config(**artifact["config"])
    predictions = []
    for member in artifact["members"]:
        model = GatedFusion(text.shape[1], audio.shape[1], config, artifact["variant"] != "fixed_mse")
        model.load_state_dict(member["state"])
        model.eval()
        scalers = member["scalers"]
        with torch.no_grad():
            p, _ = model(torch.tensor(scalers[0].transform(text), dtype=torch.float32),
                         torch.tensor(scalers[1].transform(audio), dtype=torch.float32))
        predictions.append(scalers[2].inverse_transform(p.numpy()))
    return np.mean(predictions, axis=0)


def run(data_dir: Path, output_dir: Path, report_path: Path, config=Config()):
    torch.set_num_threads(1)
    paths = [data_dir / name for name in ("recruitview_prepared.csv", "recruitview_llm_features.csv", "recruitview_audio_features.csv")]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    signature = {"config": asdict(config), "inputs": input_hashes,
                 "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    run_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["run_hash"] != run_hash:
        raise ValueError("Output directory belongs to another protocol/input version; choose a new directory.")
    manifest.write_text(json.dumps({**signature, "run_hash": run_hash,
        "selection": "Mean fold Big Five Spearman on development participants only; seeds averaged before scoring.",
        "test_status": "Previously inspected historical split. Exploratory follow-up, not untouched confirmation.",
        "targets": DIRECT_TARGET_COLUMNS, "variants": VARIANTS}, indent=2))
    rows = _read(paths[0])
    text, text_columns = _load_numeric_table(paths[1], rows, FEATURE_COLUMNS)
    audio, audio_columns = _load_numeric_table(paths[2], rows)
    y = _targets(rows, DIRECT_TARGET_COLUMNS)
    development, test = split_indices(rows)
    groups = np.array([r["participant_id"] for r in rows])
    folds = [(development[a], development[b]) for a, b in _group_splits(groups[development], config.folds, 20260911)]
    candidates = {}
    for variant in VARIANTS:
        for epoch in config.checkpoints:
            candidates[f"{variant}:{epoch}"] = {"folds": [], "seed_scores": [], "variant": variant, "epoch": epoch}
    for alpha in config.ridge_alphas:
        candidates[f"ridge:{alpha}"] = {"folds": [], "alpha": alpha, "variant": "ridge"}
    x = np.hstack((text, audio))
    for fold, (train, validation) in enumerate(folds):
        for variant in VARIANTS:
            outputs = []
            for seed in config.seeds:
                cache = output_dir / f"fold{fold}-{variant}-seed{seed}.joblib"
                if cache.exists():
                    result = joblib.load(cache)
                else:
                    result, _ = fit_neural(text, audio, y, train, validation, variant, config, seed, config.checkpoints)
                    joblib.dump(result, cache)
                outputs.append(result)
            for epoch in config.checkpoints:
                candidate = candidates[f"{variant}:{epoch}"]
                prediction = np.mean([o[epoch]["prediction"] for o in outputs], axis=0)
                candidate["folds"].append(metrics(y[validation], prediction))
                candidate["seed_scores"].append([metrics(y[validation], o[epoch]["prediction"])["big_five_spearman"] for o in outputs])
            print(f"Completed development fold {fold + 1}/{config.folds}: {variant}", flush=True)
        scaler = StandardScaler().fit(x[train])
        for alpha in config.ridge_alphas:
            model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
            candidates[f"ridge:{alpha}"]["folds"].append(metrics(y[validation], model.predict(scaler.transform(x[validation]))))
    for candidate in candidates.values():
        for metric in ("big_five_spearman", "speaking_spearman"):
            scores = [f[metric] for f in candidate["folds"]]
            candidate[metric] = float(np.mean(scores))
            candidate[metric + "_fold_std"] = float(np.std(scores, ddof=1))
    best_by_variant = {variant: max((k for k, v in candidates.items() if v["variant"] == variant),
                                    key=lambda k: candidates[k]["big_five_spearman"])
                       for variant in (*VARIANTS, "ridge")}
    winner = max(best_by_variant.values(), key=lambda k: candidates[k]["big_five_spearman"])
    # Persist decisions BEFORE any predictions or scores on the historical test.
    selection = {"winner": winner, "best_by_variant": best_by_variant, "candidates": candidates, "run_hash": run_hash}
    (output_dir / "selection-before-test.json").write_text(json.dumps(selection, indent=2))
    print(f"Development selection locked: {winner}", flush=True)

    scaler = StandardScaler().fit(x[development])
    ridge = Ridge(alpha=candidates[best_by_variant["ridge"]]["alpha"]).fit(scaler.transform(x[development]), y[development])
    baseline = ridge.predict(scaler.transform(x[test]))
    joblib.dump({"model": ridge, "scaler": scaler, "text_columns": text_columns, "audio_columns": audio_columns}, output_dir / "ridge.joblib")
    results = {"ridge": {"metrics": metrics(y[test], baseline)}}
    predictions = {"ridge": baseline}
    for variant in VARIANTS:
        epoch = candidates[best_by_variant[variant]]["epoch"]
        members, pred, gates, seed_metrics = [], [], [], []
        for seed in config.seeds:
            result, scalers = fit_neural(text, audio, y, development, test, variant, config, seed, (epoch,))
            members.append({"state": result[epoch]["state"], "scalers": scalers, "seed": seed})
            pred.append(result[epoch]["prediction"])
            gates.append(result[epoch]["gates"])
            seed_metrics.append(metrics(y[test], pred[-1]))
        prediction = np.mean(pred, axis=0)
        predictions[variant] = prediction
        artifact = output_dir / f"{variant}.joblib"
        joblib.dump({"members": members, "config": asdict(config), "epoch": epoch, "variant": variant,
                     "text_columns": text_columns, "audio_columns": audio_columns, "targets": DIRECT_TARGET_COLUMNS,
                     "run_hash": run_hash}, artifact)
        np.testing.assert_allclose(predict_artifact(artifact, text[test], audio[test]), prediction, atol=1e-6)
        weights = np.mean(gates, axis=0)[:, 0]
        results[variant] = {"metrics": metrics(y[test], prediction), "seed_metrics": seed_metrics,
                            "text_gate": {"mean": float(weights.mean()), "p10": float(np.percentile(weights, 10)),
                                          "p90": float(np.percentile(weights, 90))}}
        print(f"Historical comparison saved: {variant}", flush=True)
    for variant, prediction in predictions.items():
        results[variant]["participant_bootstrap_95ci"] = group_bootstrap(y[test], prediction, baseline, groups[test], config.bootstrap_resamples)
    with (output_dir / "historical-test-predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "participant_id", "variant", *[f"true_{t}" for t in DIRECT_TARGET_COLUMNS], *[f"pred_{t}" for t in DIRECT_TARGET_COLUMNS]])
        for variant, prediction in predictions.items():
            for index, p in zip(test, prediction):
                writer.writerow([rows[index]["record_id"], rows[index]["participant_id"], variant, *y[index], *p])
    report = {"status": "completed", "protocol": signature, "run_hash": run_hash,
              "development_rows": len(development), "development_participants": len(set(groups[development])),
              "historical_test_rows": len(test), "historical_test_participants": len(set(groups[test])),
              "selection": selection, "historical_test": results,
              "fold_participants": [{"train": sorted(set(groups[a])), "validation": sorted(set(groups[b]))} for a, b in folds],
              "original_primary_artifact_unchanged": True}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    lines = ["# Small gated-fusion follow-up", "", "Exploratory extension after viewing the original test results. This is not a fresh confirmatory test or a CRMF reproduction.", "",
             f"Inputs: {len(text_columns)} cached LLM text columns + {len(audio_columns)} engineered audio columns. No new API calls or media extraction.",
             f"Development: {len(development)} responses / {len(set(groups[development]))} participants; {config.folds} grouped folds. Each neural prediction averages {len(config.seeds)} fixed seeds.",
             "Preprocessing is fitted separately inside each training fold. All six outputs share a small network. Huber uses training-standardised targets; ranking uses training-label order, not original human pairwise judgments.",
             "Epoch and model selection maximise mean fold Big Five Spearman. Speaking skills is secondary and never chooses the winning variant. Decisions are saved before historical test scoring.", "",
             "| Variant | Epoch / alpha | Development Big Five (fold SD) | Development speaking | Historical test Big Five | Historical test speaking |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for variant, key in best_by_variant.items():
        c, m = candidates[key], results[variant]["metrics"]
        lines.append(f"| {variant} | {c.get('epoch', c.get('alpha'))} | {c['big_five_spearman']:.4f} ({c['big_five_spearman_fold_std']:.4f}) | {c['speaking_spearman']:.4f} | {m['big_five_spearman']:.4f} | {m['speaking_spearman']:.4f} |")
    lines.extend(["", f"Development winner: **{winner}**. Existing primary Model 1 and downstream BESSI artifacts are preserved.", "",
                  "## Paired participant-bootstrap uncertainty on the historical test", "",
                  "Intervals below are score differences against the Ridge refit using this same development protocol. They do not account for prior researcher exposure to test results.", "",
                  "| Variant | Big Five difference, 95% CI | Speaking difference, 95% CI |", "| --- | ---: | ---: |"])
    for variant in VARIANTS:
        ci = results[variant]["participant_bootstrap_95ci"]
        bf, sp = ci["big_five_delta_vs_ridge"], ci["speaking_delta_vs_ridge"]
        lines.append(f"| {variant} | [{bf[0]:+.4f}, {bf[1]:+.4f}] | [{sp[0]:+.4f}, {sp[1]:+.4f}] |")
    lines.extend(["", "## Interpretation", "",
                  "The gate learns response-dependent weights; these weights are not calibrated reliability estimates or causal explanations. Comparisons isolate fixed versus learned fusion under MSE, then Huber and the additional ranking term. All settings were fixed before running this follow-up.",
                  "The original Ridge report uses a different selection procedure. Use the matched Ridge refit here to assess this extension; retain the original results as the historical reference.",
                  "No claim of better true-personality measurement, BESSI validation, or superiority to published CRMF follows from these correlations. A fresh external dataset would be needed for new independent confirmation.", ""])
    report_path.with_suffix(".md").write_text("\n".join(lines))
    return {"winner": winner, "report": str(report_path), "historical_test": {v: r["metrics"] for v, r in results.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/recruitview"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark/recruitview-gated-fusion"))
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark/recruitview-gated-fusion.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir, args.report), indent=2))


if __name__ == "__main__":
    main()
