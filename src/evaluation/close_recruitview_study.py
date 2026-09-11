"""Close the fixed study with descriptive uncertainty and confound diagnostics.

Never selects a new model or overwrites existing model artifacts. The historical
test is reused for descriptive analyses and is not an untouched confirmation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.train_model1_feature_fusion import _read, _load_numeric_table, _length_features, _targets
from src.models.train_model1_gated_fusion import correlations, metrics, split_indices
from src.evaluation.recruitview_final_evaluation import _group_splits
from src.models.model1_protocol import DIRECT_TARGET_COLUMNS


def clustered_intervals(y, predictions, groups, resamples=1000):
    rng = np.random.default_rng(20260911)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    samples = {name: [] for name in predictions}
    for _ in range(resamples):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        for name, p in predictions.items():
            r = correlations(y[idx], p[idx])
            samples[name].append(np.r_[r[:5].mean(), r])
    labels = ("big_five_average", *DIRECT_TARGET_COLUMNS)
    values = {name: np.asarray(items) for name, items in samples.items()}
    def interval(array):
        return {name: np.nanpercentile(array[:, j], [2.5, 97.5]).tolist() for j, name in enumerate(labels)}
    return {"absolute": {name: interval(v) for name, v in values.items()},
            "primary_minus_baseline": {name: interval(values["llm_fused"] - v) for name, v in values.items() if name != "llm_fused"}}


def partial_rank_correlations(y, prediction, controls):
    """Descriptive within-sample residual rank correlation; not causal adjustment."""
    design = np.column_stack((np.ones(len(y)), rankdata(controls, axis=0)))
    ranked_y, ranked_p = rankdata(y, axis=0), rankdata(prediction, axis=0)
    ry = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    rp = ranked_p - design @ np.linalg.lstsq(design, ranked_p, rcond=None)[0]
    ny, npred = np.linalg.norm(ry, axis=0), np.linalg.norm(rp, axis=0)
    den = ny * npred
    # Exact control relationships can leave floating-point residuals after OLS.
    # Do not report correlations of that numerical noise as remaining signal.
    valid = ((ny > 1e-10 * np.linalg.norm(ranked_y, axis=0)) &
             (npred > 1e-10 * np.linalg.norm(ranked_p, axis=0)))
    return np.divide((ry * rp).sum(axis=0), den, out=np.full(y.shape[1], np.nan), where=valid)


def run():
    data = Path("data/processed/recruitview")
    report_path = Path("reports/benchmark/recruitview-final-review.json")
    artifact_path = Path("outputs/models/model1_feature_fusion.joblib")
    artifact = joblib.load(artifact_path)
    original = json.loads(Path("reports/benchmark/recruitview-feature-fusion.json").read_text())
    if original["protocol"]["primary_variant"] != "llm_fused":
        raise ValueError("Closing protocol expects the frozen llm_fused primary.")
    rows = _read(data / "recruitview_prepared.csv")
    llm, _ = _load_numeric_table(data / "recruitview_llm_features.csv", rows, tuple(artifact["llm_feature_columns"]))
    audio, columns = _load_numeric_table(data / "recruitview_audio_features.csv", rows, tuple(artifact["audio_feature_columns"]))
    length = _length_features(rows)
    y = _targets(rows, DIRECT_TARGET_COLUMNS)
    dev, test = split_indices(rows)
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    groups = np.array([r["participant_id"] for r in rows])
    raw = {"length_only": length, "audio_only": audio, "llm_fused": np.hstack((llm, audio))}
    scaled = {"length_only": artifact["length_scaler"].transform(length),
              "audio_only": artifact["audio_scaler"].transform(audio),
              "llm_fused": np.hstack((artifact["llm_scaler"].transform(llm), artifact["audio_scaler"].transform(audio)))}
    predictions = {}
    for name, x in scaled.items():
        models = artifact["models"][name]
        predictions[name] = np.column_stack((models["big_five"].predict(x[test]), models["speaking_skills"].predict(x[test])))
    saved = {r["record_id"]: r for r in _read(Path("outputs/benchmark/recruitview-feature-fusion/predictions.csv"))}
    expected = np.array([[float(saved[rows[i]["record_id"]]["prediction_" + t]) for t in DIRECT_TARGET_COLUMNS] for i in test])
    np.testing.assert_allclose(predictions["llm_fused"], expected, atol=1e-8)
    scores = {name: metrics(y[test], p) for name, p in predictions.items()}
    intervals = clustered_intervals(y[test], predictions, groups[test])
    print("Verified frozen predictions and completed participant-bootstrap intervals.", flush=True)
    folds = []
    for seed in (42, 73, 111):
        for fold, (a, b) in enumerate(_group_splits(groups[dev], 5, seed)):
            a, b = dev[a], dev[b]
            entry = {"seed": seed, "fold": fold, "train_participants": sorted(set(groups[a])), "validation_participants": sorted(set(groups[b])), "scores": {}}
            for name, x in raw.items():
                scaler = StandardScaler().fit(x[a])
                p = []
                for target_slice, key in ((slice(0, 5), "big_five"), (slice(5, 6), "speaking_skills")):
                    alpha = original["variants"][name][key]["selected_alpha"]
                    model = Ridge(alpha=alpha).fit(scaler.transform(x[a]), y[a, target_slice])
                    p.append(model.predict(scaler.transform(x[b])))
                entry["scores"][name] = metrics(y[b], np.column_stack(p))
            folds.append(entry)
    cv = {name: {key: {"mean": float(np.mean([f["scores"][name][key] for f in folds])),
                       "fold_sd": float(np.std([f["scores"][name][key] for f in folds], ddof=1))}
                 for key in ("big_five_spearman", "speaking_spearman")} for name in raw}
    duration = audio[:, columns.index("audio_duration_seconds")]
    boundaries = np.quantile(duration[train], [1 / 3, 2 / 3])
    bins = np.digitize(duration[test], boundaries)
    subsets = {}
    def summarise(idx):
        p, target = predictions["llm_fused"][idx], y[test][idx]
        base = {"rows": len(idx), "participants": len(set(groups[test][idx]))}
        if base["participants"] < 10:
            return {**base, "status": "Too few participants for subgroup correlation (minimum 10)."}
        return {**base, "status": "descriptive_only", "metrics": metrics(target, p)}
    for j, label in enumerate(("short", "medium", "long")):
        subsets["duration_" + label] = summarise(np.flatnonzero(bins == j))
    questions = np.array([rows[i]["question_id"] for i in test])
    for q in np.unique(questions):
        subsets["question_" + q] = summarise(np.flatnonzero(questions == q))
    controls = np.column_stack((length[test, 0], duration[test]))
    partial = partial_rank_correlations(y[test], predictions["llm_fused"], controls)
    error = (predictions["llm_fused"] - y[test]) / np.maximum(y[dev].std(axis=0), 1e-8)
    worst = np.argsort(-np.abs(error[:, :5]).mean(axis=1))[:5]
    error_examples = []
    for j in worst:
        target_index = int(np.argmax(np.abs(error[j, :5])))
        error_examples.append({"record_id": rows[test[j]]["record_id"], "question_id": questions[j],
                               "duration_seconds": float(duration[test[j]]), "largest_error_trait": DIRECT_TARGET_COLUMNS[target_index],
                               "observed": float(y[test[j], target_index]), "predicted": float(predictions["llm_fused"][j, target_index]),
                               "error_in_development_sd": float(error[j, target_index])})
    # Reuse saved Model 2 predictions only. The source does not support verified participant grouping.
    m2rows = _read(Path("outputs/model-evaluation/model2_selected_test_predictions.csv"))
    m2targets = [k.removeprefix("observed_") for k in m2rows[0] if k.startswith("observed_")]
    my = np.array([[float(r["observed_" + k]) for k in m2targets] for r in m2rows])
    mp = np.array([[float(r["prediction_" + k]) for k in m2targets] for r in m2rows])
    def m2metrics(a, b):
        err = a - b
        r2 = 1 - (err ** 2).sum(axis=0) / ((a - a.mean(axis=0)) ** 2).sum(axis=0)
        return np.array([np.abs(err).mean(), np.sqrt((err ** 2).mean()), r2.mean()])
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(1000):
        idx = rng.integers(0, len(my), len(my))
        samples.append(m2metrics(my[idx], mp[idx]))
    m2ci = np.percentile(samples, [2.5, 97.5], axis=0)
    tracked = [artifact_path, data / "recruitview_prepared.csv", data / "recruitview_audio_features.csv", data / "recruitview_llm_features.csv",
               Path("outputs/benchmark/recruitview-feature-fusion/predictions.csv"), Path("outputs/model-evaluation/model2_selected_test_predictions.csv"), Path(__file__)]
    report = {"status": "completed", "scope": "Exploratory closing diagnostics; no model selection or promotion.",
              "historical_test_rows": len(test), "historical_test_participants": len(set(groups[test])),
              "metrics": scores, "participant_bootstrap_95ci": intervals, "grouped_development_sensitivity": cv,
              "fold_details": folds, "duration_cutoffs_from_train_seconds": boundaries.tolist(), "subgroups": subsets,
              "partial_rank_correlation_controlling_word_count_and_duration": dict(zip(DIRECT_TARGET_COLUMNS, partial.tolist())),
              "error_examples": error_examples,
              "model2_row_bootstrap": {k: {"score": float(v), "ci95": m2ci[:, i].tolist()} for i, (k, v) in enumerate(zip(("mae", "rmse", "mean_r2"), m2metrics(my, mp)))},
              "demographic_analysis": "Not performed: prepared RecruitView table has no verified demographic columns; never infer demographics from media.",
              "limitations": ["Historical test previously inspected; descriptive results cannot restore independent confirmation.",
                              "Repeated development folds use previously selected alphas; sensitivity analysis, not nested unbiased model selection.",
                              "Length adjustment is descriptive and cannot establish causal skill measurement.",
                              "Model 2 intervals assume independent rows, an assumption not verified from the non-unique Case field.",
                              "RecruitView contains perceived interview ratings, not measured BESSI skills."],
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}}
    report_path.write_text(json.dumps(report, indent=2))
    def ci(v):
        return f"[{v[0]:+.4f}, {v[1]:+.4f}]"
    lines = ["# Final evaluation: compact RecruitView study", "", "Model expansion is closed. These analyses reuse existing data and saved predictions; no primary model is changed.", "",
             "## Historical test results", "", f"{len(test)} responses from {len(set(groups[test]))} participants. Scores are correlations, not accuracy percentages.", "",
             "| Model | Big Five average | Speaking skills |", "| --- | ---: | ---: |"]
    for name, m in scores.items():
        lines.append(f"| {name} | {m['big_five_spearman']:.4f} | {m['speaking_spearman']:.4f} |")
    lines += ["", "## Does the full feature set add more than response length?", "",
              "Paired 95% intervals resample whole participants (1,000 draws). An interval containing zero does not establish an improvement. They are descriptive after prior test inspection.", "",
              "| Primary model minus baseline | Big Five difference interval | Speaking difference interval |", "| --- | ---: | ---: |"]
    for name, result in intervals["primary_minus_baseline"].items():
        lines.append(f"| {name} | {ci(result['big_five_average'])} | {ci(result['speaking_skills'])} |")
    lines += ["", "## Primary per-target results", "", "| Target | Spearman | 95% participant CI | MAE | RMSE |", "| --- | ---: | --- | ---: | ---: |"]
    for target, m in scores["llm_fused"]["per_target"].items():
        lines.append(f"| {target} | {m['spearman']:.4f} | {ci(intervals['absolute']['llm_fused'][target])} | {m['mae']:.4f} | {m['rmse']:.4f} |")
    lines += ["", "## Stability across development participants", "", "Five participant-grouped folds repeated with three fixed seeds. Previously selected alphas are held fixed; these folds provide sensitivity checks, not fresh unbiased validation. Fold SD is descriptive and is not a confidence interval.", "",
              "| Model | Mean Big Five (fold SD) | Mean speaking (fold SD) |", "| --- | ---: | ---: |"]
    for name, result in cv.items():
        a, b = result["big_five_spearman"], result["speaking_spearman"]
        lines.append(f"| {name} | {a['mean']:.4f} ({a['fold_sd']:.4f}) | {b['mean']:.4f} ({b['fold_sd']:.4f}) |")
    lines += ["", "## Response length and duration", "", "Descriptive partial rank correlations remove linear associations with ranked word count and duration from both labels and predictions. They do not remove every length effect or prove causation.", ""]
    for name, v in zip(DIRECT_TARGET_COLUMNS, partial):
        lines.append(f"- {name}: {v:.4f}")
    lines += ["", f"Duration groups use training-only boundaries of {boundaries[0]:.2f} and {boundaries[1]:.2f} seconds. Question/duration subgroup scores are descriptive and are suppressed below 10 participants. No demographic attributes are inferred.", "",
              "| Subgroup | Rows | Participants | Big Five | Speaking |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, s in subsets.items():
        if "metrics" in s:
            lines.append(f"| {name} | {s['rows']} | {s['participants']} | {s['metrics']['big_five_spearman']:.4f} | {s['metrics']['speaking_spearman']:.4f} |")
    lines += ["", "## Error examples", "", "Five largest mean Big Five errors, scaled using development-label standard deviations. IDs support local audit; transcripts and identifying details are omitted. Large signed errors show the direction of the mismatch, not its psychological cause.", "",
              "| Record | Question | Seconds | Largest-error trait | Observed | Predicted | Error (SD) |", "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for e in error_examples:
        lines.append(f"| {e['record_id']} | {e['question_id']} | {e['duration_seconds']:.1f} | {e['largest_error_trait']} | {e['observed']:.3f} | {e['predicted']:.3f} | {e['error_in_development_sd']:+.2f} |")
    lines += ["", "## Model 2 uncertainty", "", "Computed from the 47 saved test rows; no model refit. Row bootstrap assumes independent rows, which the source identifier cannot verify. These intervals are exploratory.", ""]
    for name, item in report["model2_row_bootstrap"].items():
        lines.append(f"- {name}: {item['score']:.4f}; 95% interval {ci(item['ci95'])}.")
    lines += ["", "## Boundaries", "", *[f"- {x}" for x in report["limitations"]], "",
              "Full fold membership, subgroup suppression, input hashes and machine-readable metrics are in the companion JSON. Run `make close-recruitview-study` to regenerate.", ""]
    report_path.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"report": str(report_path), "baseline_differences": intervals["primary_minus_baseline"], "cv": cv}, indent=2))


if __name__ == "__main__":
    run()
