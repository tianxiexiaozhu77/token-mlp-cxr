#!/usr/bin/env python3
"""Paired patient-cluster bootstrap of image-level binary ranking metrics.

Patients, not images, are resampled. An image receives its patient's draw
multiplicity as an integer weight. This is exactly equivalent to expanding all
images of each sampled patient, including repeated patients, but avoids sorting
the expanded scores for every replicate. Score ties are grouped *before*
computing ROC and PR curves; zero-weight score groups are removed.

This quantifies test-patient uncertainty conditional on a fixed fitted method
and support set. Variation over support seeds must be reported separately.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

METRIC_NAMES = ("auroc", "auprc", "average_precision")
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20261024


def _patient_array(patient_ids: Sequence[object]) -> np.ndarray:
    raw = np.asarray(patient_ids, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("patient_ids must be a nonempty one-dimensional sequence")
    if any(x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x))
           for x in raw):
        raise ValueError("Missing/nonfinite patient identity")
    values = np.asarray([str(x) for x in raw])
    if np.any(values == "") or any("\x00" in x for x in values):
        raise ValueError("Empty or NUL-containing patient identity")
    return values


@dataclass(frozen=True)
class PatientDrawPlan:
    """Patient multiplicities; row r is the shared paired bootstrap draw r."""

    patient_ids: np.ndarray  # Canonical lexicographic order, distinct patients.
    counts: np.ndarray       # [replicate, unique_patient], each row sums to P.
    seed: int

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.patient_ids.tolist(), ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
        h.update(np.asarray(self.counts, dtype="<u4").tobytes(order="C"))
        h.update(str(self.seed).encode("ascii"))
        return h.hexdigest()

    def validate(self) -> None:
        ids = _patient_array(self.patient_ids)
        if not np.array_equal(ids, np.unique(ids)):
            raise ValueError("Plan patient identities must be unique and sorted")
        counts = np.asarray(self.counts)
        if counts.ndim != 2 or counts.shape[1] != len(ids) or not len(counts):
            raise ValueError("Invalid patient draw count shape")
        if counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("Patient draw multiplicities must be nonnegative integers")
        if not np.all(counts.sum(axis=1, dtype=np.int64) == len(ids)):
            raise ValueError("Every replicate must draw exactly P patients")


def make_patient_draw_plan(patient_ids: Sequence[object], *,
                           n_resamples: int = DEFAULT_RESAMPLES,
                           seed: int = DEFAULT_SEED) -> PatientDrawPlan:
    """Make one plan and share it across branches, tasks, models and seeds."""
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    ids = np.unique(_patient_array(patient_ids))
    p = len(ids)
    # Store uint32 to make archives portable even for larger future datasets.
    counts = np.empty((n_resamples, p), dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, n_resamples, 128):
        draws = rng.integers(0, p, size=(min(128, n_resamples - start), p))
        for j, draw in enumerate(draws):
            counts[start + j] = np.bincount(draw, minlength=p)
    plan = PatientDrawPlan(ids, counts, int(seed))
    plan.validate()
    return plan


def save_patient_draw_plan(plan: PatientDrawPlan, path: str | Path) -> None:
    plan.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the path explicitly: numpy otherwise adds .npz to arbitrary suffixes.
    with path.open("wb") as handle:
        np.savez_compressed(handle, patient_ids=plan.patient_ids, counts=plan.counts,
                            seed=np.asarray(plan.seed), sha256=np.asarray(plan.sha256),
                            generator=np.asarray("numpy.Generator(PCG64)"))


def load_patient_draw_plan(path: str | Path) -> PatientDrawPlan:
    with np.load(path, allow_pickle=False) as data:
        plan = PatientDrawPlan(data["patient_ids"], data["counts"], int(data["seed"]))
        recorded_hash = str(data["sha256"])
    plan.validate()
    if plan.sha256 != recorded_hash:
        raise ValueError("Patient draw plan SHA256 mismatch")
    return plan


@dataclass(frozen=True)
class _PreparedBinary:
    order: np.ndarray
    group_starts: np.ndarray
    sorted_labels: np.ndarray

    def metrics(self, weights: np.ndarray) -> np.ndarray:
        ordered_weights = weights[self.order]
        positive = np.add.reduceat(ordered_weights * self.sorted_labels, self.group_starts)
        total = np.add.reduceat(ordered_weights, self.group_starts)
        # sklearn excludes zero-weight observations before constructing curves.
        nonempty = total > 0
        total, positive = total[nonempty], positive[nonempty]
        if not len(total):
            return np.full(3, np.nan)
        tp = np.cumsum(positive, dtype=np.float64)
        fp = np.cumsum(total - positive, dtype=np.float64)
        n_positive, n_negative = tp[-1], fp[-1]
        if n_positive <= 0 or n_negative <= 0:
            return np.full(3, np.nan)
        recall = np.r_[0.0, tp / n_positive]
        fpr = np.r_[0.0, fp / n_negative]
        precision = np.r_[1.0, tp / (tp + fp)]
        dr = np.diff(recall)
        auroc = np.sum(np.diff(fpr) * (recall[:-1] + recall[1:]) * .5)
        auprc = np.sum(dr * (precision[:-1] + precision[1:]) * .5)
        ap = np.sum(dr * precision[1:])
        return np.asarray([auroc, auprc, ap], dtype=np.float64)


def _prepare_binary(labels: np.ndarray, scores: np.ndarray) -> _PreparedBinary:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    return _PreparedBinary(order, starts, labels[order].astype(np.float64))


def binary_weighted_metrics(labels: np.ndarray, scores: np.ndarray,
                            weights: np.ndarray | None = None) -> np.ndarray:
    """Return AUROC, trapezoidal PR AUC, AP; invalid draws return three NaNs."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Require nonempty aligned one-dimensional labels and scores")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != labels.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be aligned, finite and nonnegative")
    return _prepare_binary(labels, scores).metrics(weights)


def interval_summary(point: float, replicates: np.ndarray,
                     confidence: float = .95) -> dict:
    """JSON-safe percentile interval, explicitly reporting discarded draws."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Require one-dimensional nonempty bootstrap replicates")
    finite = values[np.isfinite(values)]
    tail = (1.0 - confidence) / 2
    ci = np.quantile(finite, [tail, 1 - tail], method="linear").tolist() if len(finite) else None
    return {"point": float(point) if np.isfinite(point) else None,
            "confidence_level": confidence, "ci": ci,
            "valid_replicates": int(len(finite)), "invalid_replicates": int(len(values) - len(finite)),
            "valid_fraction": float(len(finite) / len(values))}


@dataclass
class BootstrapResult:
    summary: dict
    # Axis order is recorded in summary and archives to avoid implicit reshaping.
    point_metrics: np.ndarray       # [branch, task, metric]
    replicate_metrics: np.ndarray   # [replicate, branch, task, metric]
    branch_names: tuple[str, ...]
    task_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    plan: PatientDrawPlan


def paired_cluster_bootstrap(labels: np.ndarray, scores: Mapping[str, np.ndarray],
                             patient_ids: Sequence[object], *, baseline: str = "original",
                             task_names: Sequence[str] | None = None,
                             plan: PatientDrawPlan | None = None,
                             n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                             confidence: float = .95) -> BootstrapResult:
    """Evaluate paired branches using one shared patient bootstrap draw plan.

    `scores` preserves insertion order as the branch axis. The same plan can be
    passed to subsequent model/support-seed calls; an exact patient-universe
    match is mandatory. Labels and scores retain image-level weighting.
    """
    labels = np.asarray(labels)
    patients = _patient_array(patient_ids)
    if labels.ndim != 2 or labels.shape[0] != len(patients) or labels.shape[1] == 0:
        raise ValueError("Require labels [number_of_images, number_of_tasks]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not scores or baseline not in scores:
        raise ValueError("scores must contain the named original baseline")
    branches = tuple(scores)
    if any(not isinstance(x, str) or not x for x in branches):
        raise ValueError("Branch names must be nonempty strings")
    names = tuple(str(x) for x in (task_names if task_names is not None
                                  else [f"task_{j}" for j in range(labels.shape[1])]))
    if len(names) != labels.shape[1] or len(set(names)) != len(names):
        raise ValueError("Task names must be unique and match the label columns")
    values = []
    for name in branches:
        value = np.asarray(scores[name], dtype=np.float64)
        if value.shape != labels.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid score shape/nonfinite scores for {name}")
        values.append(value)
    if plan is None:
        plan = make_patient_draw_plan(patients, n_resamples=n_resamples, seed=seed)
    plan.validate()
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    if not np.array_equal(plan.patient_ids, unique_patients):
        raise ValueError("Shared draw plan and prediction patient universe differ")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    n, c = labels.shape
    r, b = len(plan.counts), len(branches)
    prepared = [[_prepare_binary(labels[:, j], value[:, j]) for j in range(c)]
                for value in values]
    weights = np.ones(n, dtype=np.float64)
    points = np.asarray([[p.metrics(weights) for p in by_task] for by_task in prepared])
    samples = np.empty((r, b, c, len(METRIC_NAMES)), dtype=np.float64)
    for replicate, counts in enumerate(plan.counts):
        weights = counts[inverse].astype(np.float64)
        for ib in range(b):
            for j in range(c):
                samples[replicate, ib, j] = prepared[ib][j].metrics(weights)

    task_counts = {}
    for j, name in enumerate(names):
        has_positive = np.zeros(len(unique_patients), dtype=bool)
        has_negative = np.zeros(len(unique_patients), dtype=bool)
        np.logical_or.at(has_positive, inverse, labels[:, j] == 1)
        np.logical_or.at(has_negative, inverse, labels[:, j] == 0)
        task_counts[name] = {
            "positive_images": int(labels[:, j].sum()),
            "negative_images": int(n - labels[:, j].sum()),
            "patients_with_positive_image": int(has_positive.sum()),
            "patients_with_negative_image": int(has_negative.sum()),
            "patients_with_all_images_negative": int((~has_positive).sum()),
            "patients_with_both_labels": int((has_positive & has_negative).sum())}

    summary = {
        "schema_version": 1, "status": "complete", "metric_unit": "image",
        "resampling_unit": "patient", "interval_method": "paired percentile; linear quantiles",
        "uncertainty_scope": "test-patient sampling conditional on the fixed support set and method",
        "confidence_level": confidence, "images": n, "patients": len(unique_patients),
        "n_resamples": r, "bootstrap_seed": plan.seed, "draw_plan_sha256": plan.sha256,
        "baseline": baseline, "branch_names": list(branches), "task_names": list(names),
        "metric_names": list(METRIC_NAMES), "metric_scale": "fraction; multiply by 100 for percent or percentage-point differences",
        "point_metrics_axes": ["branch", "task", "metric"],
        "replicate_metrics_axes": ["replicate", "branch", "task", "metric"],
        "auprc_definition": "trapezoidal area under precision-recall curve; not average precision",
        "invalid_draw_policy": "a task without positive or negative image weight has all three metrics NaN; excluded from its interval with counts reported",
        "macro_policy": "unweighted task mean; invalid if any task is invalid in that replicate",
        "test_counts": task_counts, "branches": {}, "paired_differences": {}}

    def summarize_values(point_array, sample_array):
        per_task = {name: {metric: interval_summary(point_array[j, im], sample_array[:, j, im], confidence)
                           for im, metric in enumerate(METRIC_NAMES)}
                    for j, name in enumerate(names)}
        macro_point = point_array.mean(axis=0)
        macro_samples = sample_array.mean(axis=1)  # Propagate invalid tasks, never nanmean.
        return {"per_task": per_task,
                "macro": {metric: interval_summary(macro_point[im], macro_samples[:, im], confidence)
                          for im, metric in enumerate(METRIC_NAMES)}}

    original = branches.index(baseline)
    for ib, name in enumerate(branches):
        summary["branches"][name] = summarize_values(points[ib], samples[:, ib])
        if name != baseline:
            summary["paired_differences"][name] = summarize_values(
                points[ib] - points[original], samples[:, ib] - samples[:, original])
    return BootstrapResult(summary, points, samples, branches, names, METRIC_NAMES, plan)


def save_bootstrap(result: BootstrapResult, output_dir: str | Path) -> None:
    """Save strict JSON summary plus auditable draws/metrics, without pickle."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(result.summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(folder / "bootstrap_metrics.npz", point_metrics=result.point_metrics,
                        replicate_metrics=result.replicate_metrics,
                        branch_names=np.asarray(result.branch_names), task_names=np.asarray(result.task_names),
                        metric_names=np.asarray(result.metric_names),
                        draw_plan_sha256=np.asarray(result.plan.sha256))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", type=Path, help="Load, validate and describe a saved patient draw plan")
    args = parser.parse_args()
    if args.check_plan:
        draw_plan = load_patient_draw_plan(args.check_plan)
        print(json.dumps({"patients": len(draw_plan.patient_ids), "replicates": len(draw_plan.counts),
                          "seed": draw_plan.seed, "sha256": draw_plan.sha256}, indent=2))
    else:
        parser.print_help()
