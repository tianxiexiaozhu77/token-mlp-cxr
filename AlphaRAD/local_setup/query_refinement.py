"""Frozen, support-derived query refinement through AlphaRAD's original FLaS.

The fixed eta=0.3 is transferred from the earlier RadZero experiment; it is not
selected on AlphaRAD test results. Only labelled support images enter the fit.
No optimizer, gradient updates, or changes to the released model are involved.
For each class, the direction is the unit positive-minus-negative mean of its
original FLaS visual context. The refined text query is then passed back through
the unchanged FLaS branch, preserving its head aggregation and temperatures.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VERSION = "alpharad-support-query-refinement-v1"
REAGGREGATION_VERSION = "alpharad-reaggregation-aware-query-refinement-v1"
ABLATION_VERSION = "alpharad-fixed-point-ablation-v1"
UNCERTAINTY_VERSION = "alpharad-support-uncertainty-query-refinement-v1"
JACKKNIFE_VERSION = "alpharad-jackknife-stable-reaggregation-v1"
ECQA_VERSION = "alpharad-evidence-calibrated-query-adaptation-v1"
ECQT_VERSION = "alpharad-evidence-calibrated-query-transport-v1"
SEPCAL_VERSION = "alpharad-support-separation-calibrated-transport-v1"
TDQA_VERSION = "alpharad-token-discriminative-query-adaptation-v1"
CGT_VERSION = "alpharad-contrastive-grounding-transport-v1"
GATE_VERSION = "alpharad-support-only-pre-gate-v1"
DEFAULT_ETAS = (0.0, 0.3)


def _native_support_margin(support_vision, support_labels, query, matcher):
    """Return the balanced native support margin at ``query``.

    This deliberately consumes only the prepared support images and labels.
    It is used by :func:`fit_support_only_gate` before a fixed-point pass is
    launched, so the gate cannot depend on evaluation images or labels.
    """
    import copy
    labels = _cpu_float(support_labels)
    query = _cpu_float(query)
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    device = matcher.device

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    logits_batches = []
    with torch.no_grad():
        for vision_cpu in support_vision:
            result = matcher.match_prepared(
                on_device(vision_cpu), query.to(device), return_features=False)
            logits_batches.append(result["logits"])
        logits = torch.cat(logits_batches, dim=0)
        margin = (logits * labels.to(device)).sum(0) / positives.to(device) - (
            logits * (1 - labels.to(device))).sum(0) / negatives.to(device)
    return margin.detach().cpu().numpy()


def _native_support_observation(support_vision, support_labels, query, matcher):
    """Compute native direction and balanced margin in one support pass."""
    import copy
    labels = _cpu_float(support_labels)
    query = _cpu_float(query)
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    device = matcher.device

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    feature_batches, logits_batches = [], []
    with torch.no_grad():
        for vision_cpu in support_vision:
            result = matcher.match_prepared(
                on_device(vision_cpu), query.to(device), return_features=True)
            feature_batches.append(result["class_local"])
            logits_batches.append(result["logits"])
        features, logits = torch.cat(feature_batches), torch.cat(logits_batches)
        pos = (features * labels.to(device)[..., None]).sum(0) / positives.to(device)[:, None]
        neg = (features * (1 - labels.to(device))[..., None]).sum(0) / negatives.to(device)[:, None]
        direction = F.normalize(pos - neg, p=2, dim=-1)
        margin = (logits * labels.to(device)).sum(0) / positives.to(device) - (
            logits * (1 - labels.to(device))).sum(0) / negatives.to(device)
    return direction.cpu().numpy(), margin.cpu().numpy()


def fit_support_only_gate(
    support_features, support_vision, support_labels, original_queries, matcher,
    eta=0.3, gamma_margin=0.0, gamma_kappa=1.0, gamma_residual=0.0,
    max_iter=3, query_tol=1e-4, patience=1,
):
    """Select one-shot or native fixed-point using a pre-gate on support only.

    The gate first fits the ordinary one-shot query, then evaluates one cheap
    native probe step on the support set.  The probe estimates
    ``Delta M = M(q_probe)-M(q^(1))`` and the local map ratio
    ``kappa = ||q_probe-q^(1)|| / ||q^(1)-q^(0)||``.  Only when the support
    margin improvement exceeds ``gamma_margin``, the map is locally stable,
    and the native direction residual exceeds ``gamma_residual`` is the full
    fixed-point iteration launched.  Thus the expensive branch is never run
    before this decision, and no test tensor enters the gate.

    Thresholds are explicit arguments and are recorded in the returned fit;
    callers should freeze them using an independent validation split or
    leave-one-dataset-out calibration before running final evaluation.
    """
    gate_started = time.perf_counter()
    gate_probe_started = time.perf_counter()
    if not math.isfinite(float(gamma_margin)):
        raise ValueError("gamma_margin must be finite")
    if not math.isfinite(float(gamma_kappa)) or float(gamma_kappa) < 0:
        raise ValueError("gamma_kappa must be finite and nonnegative")
    if not math.isfinite(float(gamma_residual)) or float(gamma_residual) < 0:
        raise ValueError("gamma_residual must be finite and nonnegative")
    one_shot = fit_query_directions(
        support_features, support_labels, original_queries, etas=(0.0, float(eta)))
    q0, q1 = one_shot["queries"][0], one_shot["queries"][-1]
    # q^(0) and q^(1) are the two inexpensive support observations already
    # needed to make the pre-gate decision.  q_probe is one native map step;
    # a selected task then starts the full solver from the original q^(0).
    d0, margin0 = _native_support_observation(support_vision, support_labels, q0, matcher)
    d1, margin1 = _native_support_observation(support_vision, support_labels, q1, matcher)
    q_probe = F.normalize(
        F.normalize(_cpu_float(original_queries), p=2, dim=-1)
        + float(eta) * _cpu_float(d1), p=2, dim=-1).numpy()
    _, margin_probe = _native_support_observation(
        support_vision, support_labels, q_probe, matcher)
    probe_elapsed = time.perf_counter() - gate_started
    q_step = np.linalg.norm(q1 - q0, axis=-1)
    probe_step = np.linalg.norm(q_probe - q1, axis=-1)
    kappa = probe_step / np.maximum(q_step, 1e-8)
    residual = np.linalg.norm(d1 - d0, axis=-1)
    delta_margin = margin_probe - margin1
    selected_full = bool(
        float(np.mean(delta_margin)) > float(gamma_margin)
        and float(np.mean(kappa)) < float(gamma_kappa)
        and float(np.mean(residual)) >= float(gamma_residual))
    if selected_full:
        fit = fit_reaggregation_query_directions(
            support_vision, support_labels, original_queries, matcher,
            eta=eta, max_iter=max_iter, query_tol=query_tol, patience=patience)
    else:
        fit = one_shot
        # Ensure one-shot reports the true d(q0)-d(q1) representation gap.
        gap = np.linalg.norm(d0 - d1, axis=-1)
        cosine = (d0 * d1).sum(-1)
        fit.update({
            "direction_initial": d0, "direction_final": d1,
            "direction_gap_l2": gap,
            "direction_gap_l2_mean": float(gap.mean()),
            "direction_gap_l2_median": float(np.median(gap)),
            "direction_gap_l2_max": float(gap.max()),
            "direction_gap_cos": cosine,
            "direction_gap_cos_mean": float(cosine.mean()),
            "direction_gap_cos_min": float(cosine.min()),
            "native_reaggregation_passes": 3,
        })
    fit.update({
        "gate_name": "support_only_pre_gate_v1",
        "gate_selected_full": selected_full,
        "gate_thresholds": {"gamma_margin": float(gamma_margin),
                             "gamma_kappa": float(gamma_kappa),
                             "gamma_residual": float(gamma_residual)},
        "gate_probe_steps": 1,
        "gate_probe_native_passes": 3,
        "gate_probe_elapsed_seconds": float(time.perf_counter() - gate_probe_started),
        "gate_support_margin_q0": margin0,
        "gate_support_margin_q1": margin1,
        "gate_support_margin_probe": margin_probe,
        "gate_delta_margin": delta_margin,
        "gate_local_map_ratio": kappa,
        "gate_direction_residual_l2": residual,
        "gate_selected_full_rate": float(selected_full),
        "native_reaggregation_passes": (3 + int(fit.get("native_reaggregation_passes", 0))
                                        if selected_full else 3),
        "gate_total_native_passes": (3 + int(fit.get("native_reaggregation_passes", 0))
                                     if selected_full else 3),
        "estimated_fit_flops": float(
            (3 + int(fit.get("native_reaggregation_passes", 0)) if selected_full else 3)
            * len(support_labels) * _cpu_float(original_queries).shape[0]
            * _cpu_float(original_queries).shape[1] * 2),
        "gate_test_labels_used": False,
        "gate_pre_full_decision": True,
        # Wall-clock is measured around the complete pre-gate protocol,
        # including the three native support probes.  This keeps cost
        # accounting honest for rejected episodes as well as selected ones.
        "gate_probe_elapsed_seconds": float(probe_elapsed),
        "fit_elapsed_seconds": float(time.perf_counter() - gate_started),
        "wall_clock_seconds": float(time.perf_counter() - gate_started),
    })
    return fit


def validate_etas(etas):
    values = tuple(float(value) for value in etas)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Etas must be finite nonnegative values")
    if len(set(values)) != len(values) or 0.0 not in values:
        raise ValueError("Etas must be unique and include the unchanged zero baseline")
    return values


def _cpu_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).clone()
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)


def fit_query_directions(support_features, support_labels, original_queries,
                         etas=DEFAULT_ETAS):
    """Fit from support [N,C,D], binary labels [N,C], and text queries [C,D].

    The interface and equations follow RadZero's fit_query_directions. Input
    class-local contexts must already have unit norm over their full D-vector.
    Every class requires both positive and negative support images. Test labels,
    validation scores, and evaluation images are not inputs to this function.
    """
    features, labels, original = map(_cpu_float,
        (support_features, support_labels, original_queries))
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Empty or nonfinite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    norms = torch.linalg.vector_norm(features, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=1e-5):
        raise ValueError("Class-local support features must have unit vector norm")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each class needs positive and negative support examples")
    positive_mean = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_mean = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = positive_mean - negative_mean
    direction_norms = torch.linalg.vector_norm(difference, dim=-1)
    directions = F.normalize(difference, p=2, dim=-1)
    normalized_query = F.normalize(original, p=2, dim=-1)
    queries = torch.stack([
        original.clone() if eta == 0 else F.normalize(
            normalized_query + eta * directions, p=2, dim=-1
        ) for eta in etas
    ])
    if not torch.isfinite(queries).all():
        raise ValueError("Nonfinite refined queries")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(), "positive_means": positive_mean.numpy(),
        "negative_means": negative_mean.numpy(),
        "direction_norms": direction_norms.numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (direction_norms < 1e-12).numpy(),
    }


def reaggregated_direction(support_vision, support_labels, query, matcher):
    """Evaluate the native support direction at an arbitrary query.

    This helper is used for the representation-gap diagnostic of one-shot and
    cached controls.  It deliberately performs a fresh native aggregation at
    ``query`` instead of treating the cached q^(0) features as d(query).
    """
    import copy
    labels = _cpu_float(support_labels)
    query = _cpu_float(query)
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    device = matcher.device

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    batches = []
    with torch.no_grad():
        q = query.to(device)
        for vision_cpu in support_vision:
            result = matcher.match_prepared(on_device(vision_cpu), q, return_features=True)
            batches.append(result["class_local"])
        features = torch.cat(batches, dim=0)
        pos = (features * labels.to(device)[..., None]).sum(0) / positives.to(device)[:, None]
        neg = (features * (1 - labels.to(device))[..., None]).sum(0) / negatives.to(device)[:, None]
        return F.normalize(pos - neg, p=2, dim=-1).cpu().numpy()


def fit_cached_multistep_query_directions(
    support_features, support_labels, original_queries, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1,
):
    """Run the query map repeatedly while *holding support features fixed*.

    This is the explicit no-reaggregation ablation for the native fixed-point
    rule.  ``support_features`` are computed once at the original query and
    reused on every step; only the query state changes.  Keeping this path
    separate from :func:`fit_reaggregation_query_directions` makes it possible
    to attribute any gain to query iteration rather than to native
    query-conditioned aggregation.
    """
    features, labels, original = map(_cpu_float,
        (support_features, support_labels, original_queries))
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Empty or nonfinite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    if not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("eta must be finite and positive")
    if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("Invalid cached fixed-point iteration settings")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    if not torch.allclose(torch.linalg.vector_norm(features, dim=-1),
                          torch.ones(features.shape[:2]), atol=2e-5, rtol=1e-5):
        raise ValueError("Class-local support features must have unit vector norm")
    positive_means = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_means = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = positive_means - negative_means
    directions = F.normalize(difference, p=2, dim=-1)
    direction_norms = torch.linalg.vector_norm(difference, dim=-1)
    anchor = F.normalize(original, p=2, dim=-1)
    query = original.clone()
    trajectory = [query.clone()]
    history, stable_steps = [], 0
    previous_query = previous_next = None
    baseline_margin = None
    stop_reason = "max_iter"
    eta_values = torch.full((original.shape[0],), float(eta), dtype=torch.float32)
    for iteration in range(int(max_iter)):
        normalized = F.normalize(query, p=2, dim=-1)
        projections = torch.einsum("ncd,cd->nc", features, normalized)
        margin = (projections * labels).sum(0) / positives - (
            projections * (1 - labels)).sum(0) / negatives
        if baseline_margin is None:
            baseline_margin = margin.clone()
        next_query = F.normalize(anchor + float(eta) * directions, p=2, dim=-1)
        delta = torch.linalg.vector_norm(next_query - normalized, dim=-1)
        if previous_query is None:
            input_delta = map_delta = local_lipschitz = direction_delta = None
        else:
            input_step = torch.linalg.vector_norm(normalized - previous_query, dim=-1)
            map_step = torch.linalg.vector_norm(next_query - previous_next, dim=-1)
            input_delta = float(input_step.mean())
            map_delta = float(map_step.mean())
            local_lipschitz = float((map_step / input_step.clamp_min(1e-8)).mean())
            direction_delta = 0.0
        history.append({
            "iteration": iteration,
            "query_delta_max": float(delta.max()),
            "query_delta_mean": float(delta.mean()),
            "direction_alignment_mean": float((directions * anchor).sum(-1).mean()),
            "support_margin_mean": float(margin.mean()),
            "support_margin_min": float(margin.min()),
            "support_margin_per_class": margin.tolist(),
            "support_utility": float((margin - baseline_margin).mean()),
            "support_utility_per_class": (margin - baseline_margin).tolist(),
            "input_delta_mean": input_delta,
            "map_delta_mean": map_delta,
            "direction_delta_mean": direction_delta,
            "local_lipschitz_mean": local_lipschitz,
            "zero_direction_classes": int((direction_norms < 1e-12).sum()),
            "cached_support_features": True,
            "native_reaggregation": False,
        })
        previous_query, previous_next = normalized.clone(), next_query.clone()
        query = next_query
        trajectory.append(query.clone())
        stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
        if stable_steps >= int(patience):
            stop_reason = "query_stability"
            break
    gap = np.linalg.norm(directions.numpy() - directions.numpy(), axis=-1)
    return {
        "etas": np.asarray((0.0, float(eta)), dtype=np.float64),
        "original_queries": original.numpy(),
        "queries": np.stack((original.numpy(), query.numpy())),
        "directions": directions.numpy(),
        "positive_means": positive_means.numpy(), "negative_means": negative_means.numpy(),
        "direction_norms": direction_norms.numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (direction_norms < 1e-12).numpy(),
        "reaggregation_iterations": len(history), "reaggregation_max_iter": int(max_iter),
        "reaggregation_query_tol": float(query_tol), "reaggregation_patience": int(patience),
        "reaggregation_stop_reason": stop_reason, "reaggregation_history": history,
        "query_trajectory": torch.stack(trajectory).numpy(),
        "support_margin_baseline": baseline_margin.numpy(),
        "support_utility_final": float(history[-1]["support_utility"]),
        "local_lipschitz_max": float(max((x["local_lipschitz_mean"] for x in history
                                           if x["local_lipschitz_mean"] is not None), default=0.0)),
        "selected_eta_per_class": eta_values.numpy(),
        "ablation_mode": "cached_multistep_no_reaggregation",
        "direction_gap_l2": gap, "direction_gap_l2_mean": float(gap.mean()),
        "direction_gap_l2_median": float(np.median(gap)), "direction_gap_l2_max": float(gap.max()),
        "direction_gap_cos": np.ones_like(gap), "direction_gap_cos_mean": 1.0,
        "native_reaggregation_passes": 1, "fit_elapsed_seconds": 0.0,
    }


def fit_reaggregation_only_query_directions(
    support_vision, support_labels, original_queries, matcher,
    eta=0.3, max_iter=3, query_tol=1e-4, patience=1,
):
    """Recompute native support aggregation while keeping ``q=q0`` fixed.

    The branch isolates the aggregation effect: support contexts and margins
    are recomputed at every step, but no query update is applied.  It therefore
    returns two identical query branches so the evaluator can measure the
    aggregation-only support diagnostics without silently changing inference.
    """
    import copy
    labels, original = _cpu_float(support_labels), _cpu_float(original_queries)
    if not isinstance(support_vision, (list, tuple)) or not support_vision:
        raise ValueError("support_vision must contain at least one prepared batch")
    if labels.ndim != 2 or original.ndim != 2 or labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be finite binary values")
    if not torch.isfinite(original).all() or not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("Expected finite original query and positive eta")
    if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("Invalid aggregation-only iteration settings")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    device = matcher.device
    query = original.to(device)
    labels_d = labels.to(device)
    history, feature_batches = [], []
    directions = positive_means = negative_means = direction_norms = None
    baseline_margin = None
    with torch.no_grad():
        for iteration in range(int(max_iter)):
            feature_batches, logit_batches = [], []
            for vision_cpu in support_vision:
                def on_device(value):
                    if isinstance(value, torch.Tensor):
                        return value.to(device=device, dtype=torch.float32, non_blocking=True)
                    if isinstance(value, dict):
                        return {key: on_device(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return type(value)(on_device(item) for item in value)
                    return copy.deepcopy(value)
                result = matcher.match_prepared(on_device(vision_cpu), query, return_features=True)
                feature_batches.append(result["class_local"])
                logit_batches.append(result["logits"])
            features, logits = torch.cat(feature_batches), torch.cat(logit_batches)
            if features.shape[:2] != labels.shape:
                raise ValueError("Reaggregated support feature/label shape mismatch")
            positive_means = (features * labels_d[..., None]).sum(0) / positives.to(device)[:, None]
            negative_means = (features * (1 - labels_d)[..., None]).sum(0) / negatives.to(device)[:, None]
            difference = positive_means - negative_means
            directions = F.normalize(difference, p=2, dim=-1)
            direction_norms = torch.linalg.vector_norm(difference, dim=-1)
            margin = (logits * labels_d).sum(0) / positives.to(device) - (
                logits * (1 - labels_d)).sum(0) / negatives.to(device)
            if baseline_margin is None:
                baseline_margin = margin.clone()
            history.append({
                "iteration": iteration,
                "query_delta_max": 0.0, "query_delta_mean": 0.0,
                "direction_alignment_mean": float((directions * F.normalize(query, dim=-1)).sum(-1).mean()),
                "support_margin_mean": float(margin.mean()),
                "support_margin_min": float(margin.min()),
                "support_margin_per_class": margin.detach().cpu().tolist(),
                "support_utility": float((margin - baseline_margin).mean()),
                "support_utility_per_class": (margin - baseline_margin).detach().cpu().tolist(),
                "input_delta_mean": 0.0, "map_delta_mean": 0.0,
                "direction_delta_mean": None, "local_lipschitz_mean": 0.0,
                "zero_direction_classes": int((direction_norms < 1e-12).sum()),
                "cached_support_features": False, "native_reaggregation": True,
            })
    if directions is None:
        raise ValueError("Aggregation-only fit produced no support direction")
    gap = np.linalg.norm(directions.cpu().numpy() - directions.cpu().numpy(), axis=-1)
    return {
        "etas": np.asarray((0.0, float(eta)), dtype=np.float64),
        "original_queries": original.numpy(), "queries": np.stack((original.numpy(), original.numpy())),
        "directions": directions.cpu().numpy(),
        "positive_means": positive_means.cpu().numpy(), "negative_means": negative_means.cpu().numpy(),
        "direction_norms": direction_norms.cpu().numpy(),
        "positive_counts": positives.numpy().astype(np.int64), "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (direction_norms < 1e-12).cpu().numpy(),
        "reaggregation_iterations": len(history), "reaggregation_max_iter": int(max_iter),
        "reaggregation_query_tol": float(query_tol), "reaggregation_patience": int(patience),
        "reaggregation_stop_reason": "fixed_query", "reaggregation_history": history,
        "query_trajectory": np.repeat(original.numpy()[None], len(history) + 1, axis=0),
        "support_margin_baseline": baseline_margin.cpu().numpy(),
        "support_utility_final": float(history[-1]["support_utility"]),
        "local_lipschitz_max": 0.0, "ablation_mode": "reaggregation_only_fixed_query",
        "direction_gap_l2": gap, "direction_gap_l2_mean": float(gap.mean()),
        "direction_gap_l2_median": float(np.median(gap)), "direction_gap_l2_max": float(gap.max()),
        "direction_gap_cos": np.ones_like(gap), "direction_gap_cos_mean": 1.0,
        "native_reaggregation_passes": len(history), "fit_elapsed_seconds": 0.0,
    }


def fit_support_selected_query_directions(
    support_features, support_labels, original_queries,
    eta_candidates=(0.0, 0.1, 0.2, 0.3),
):
    """Select an adaptation magnitude using support-only leave-one-out margins.

    For each held-out support image, a class direction is fitted on the other
    support images and the held-out signed cosine margin is evaluated for every
    candidate ``eta``.  The best eta is selected per class; a macro support
    margin provides a single global eta for reporting and compatibility with
    the existing evaluator.  The final query itself uses the class-specific
    eta values, so this is a genuine per-class selection rather than a test
    tuned scalar.  It is intentionally feature-only and does not require an
    optimizer or model weights.
    """
    features, labels, original = map(_cpu_float,
        (support_features, support_labels, original_queries))
    candidates = validate_etas(eta_candidates)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if len(features) < 4 or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Invalid support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives < 2) or torch.any(negatives < 2):
        raise ValueError("Leave-one-out selection needs at least two positives and negatives")
    normalized_query = F.normalize(original, p=2, dim=-1)
    n_classes, n_candidates = features.shape[1], len(candidates)
    margin_curve = torch.zeros(n_classes, n_candidates, dtype=features.dtype)
    counts = torch.zeros(n_classes, n_candidates, dtype=features.dtype)
    # Leave-one-out fitting is tiny (ten support images in the standard
    # episode) and the explicit loops keep class/sample bookkeeping auditable.
    for i in range(len(features)):
        keep = torch.ones(len(features), dtype=torch.bool)
        keep[i] = False
        y_keep = labels[keep]
        for c in range(n_classes):
            pos = features[keep, c][y_keep[:, c] == 1]
            neg = features[keep, c][y_keep[:, c] == 0]
            direction = F.normalize(pos.mean(0) - neg.mean(0), p=2, dim=0)
            signed_label = 2 * labels[i, c] - 1
            for j, eta in enumerate(candidates):
                query = normalized_query[c] if eta == 0 else F.normalize(
                    normalized_query[c] + float(eta) * direction, p=2, dim=0)
                margin_curve[c, j] += signed_label * torch.dot(query, features[i, c])
                counts[c, j] += 1
    margin_curve = margin_curve / counts.clamp_min(1)
    selected_indices = margin_curve.argmax(dim=1)
    selected_eta = np.asarray([candidates[int(i)] for i in selected_indices], dtype=np.float64)
    global_scores = margin_curve.mean(0)
    global_index = int(global_scores.argmax())
    global_eta = float(candidates[global_index])
    # Refit once on all support examples, while retaining the per-class eta.
    positive_mean = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_mean = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    directions = F.normalize(positive_mean - negative_mean, p=2, dim=-1)
    selected_queries = torch.stack([
        normalized_query[c] if selected_eta[c] == 0 else F.normalize(
            normalized_query[c] + float(selected_eta[c]) * directions[c], p=2, dim=0)
        for c in range(n_classes)
    ])
    queries = torch.stack((original, selected_queries))
    return {
        "etas": np.asarray((0.0, global_eta), dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(),
        "positive_means": positive_mean.numpy(), "negative_means": negative_mean.numpy(),
        "direction_norms": torch.linalg.vector_norm(positive_mean - negative_mean, dim=-1).numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "selected_eta_per_class": selected_eta,
        "selected_eta_global": global_eta,
        "eta_candidates": np.asarray(candidates, dtype=np.float64),
        "support_margin_curve": margin_curve.numpy(),
        "support_margin_global": global_scores.numpy(),
        "zero_direction_classes": (torch.linalg.vector_norm(directions, dim=-1) < 1e-12).numpy(),
    }


def fit_separation_calibrated_query_directions(
    support_features, support_labels, original_queries, etas=DEFAULT_ETAS,
    separation_temperature=1.0, reliability_floor=0.25,
):
    """Transport the query with a support-separation reliability gate.

    The positive-minus-negative residual is calibrated by a support-only
    standardized separation score.  The score compares the class margin along
    the learned direction with its pooled within-class standard error; classes
    whose support cannot reliably separate the two labels receive a smaller
    query step.  This is a diagonal, episode-level shrinkage rule and does not
    fit a classifier or inspect evaluation images.
    """
    features, labels, original = map(_cpu_float,
        (support_features, support_labels, original_queries))
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2] or original.shape != features.shape[1:]:
        raise ValueError("Expected support features [N,C,D], labels [N,C], and matching queries")
    if not torch.isfinite(features).all() or not torch.isfinite(original).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Invalid support/query features or labels")
    if not math.isfinite(float(separation_temperature)) or separation_temperature <= 0:
        raise ValueError("separation_temperature must be finite and positive")
    if not math.isfinite(float(reliability_floor)) or not 0 < reliability_floor <= 1:
        raise ValueError("reliability_floor must lie in (0, 1]")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives < 1) or torch.any(negatives < 1):
        raise ValueError("Each class needs positive and negative support examples")
    pos = (features * labels[..., None]).sum(0) / positives[:, None]
    neg = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = pos - neg
    directions = F.normalize(difference, p=2, dim=-1)
    projections = (features * directions[None]).sum(-1)
    pos_proj = projections * labels
    neg_proj = projections * (1 - labels)
    pos_var = ((pos_proj - (pos_proj.sum(0) / positives) * labels) ** 2).sum(0) / positives.clamp_min(2)
    neg_var = ((neg_proj - (neg_proj.sum(0) / negatives) * (1 - labels)) ** 2).sum(0) / negatives.clamp_min(2)
    margin = (pos_proj.sum(0) / positives) - (neg_proj.sum(0) / negatives)
    standard_error = torch.sqrt((pos_var / positives + neg_var / negatives).clamp_min(0))
    separation = margin / (standard_error + 1e-6)
    reliability = torch.sigmoid(float(separation_temperature) * separation.clamp(-10, 10))
    reliability = float(reliability_floor) + (1.0 - float(reliability_floor)) * reliability
    normalized_query = F.normalize(original, p=2, dim=-1)
    weighted = directions * reliability[:, None]
    queries = torch.stack([
        original.clone() if eta == 0 else F.normalize(
            normalized_query + eta * weighted, p=2, dim=-1)
        for eta in etas
    ])
    return {
        "etas": np.asarray(etas, dtype=np.float64), "original_queries": original.numpy(),
        "queries": queries.numpy(), "directions": directions.numpy(),
        "weighted_directions": weighted.numpy(), "positive_means": pos.numpy(),
        "negative_means": neg.numpy(), "direction_norms": torch.linalg.vector_norm(difference, dim=-1).numpy(),
        "support_margin": margin.numpy(), "support_standard_error": standard_error.numpy(),
        "support_separation": separation.numpy(), "support_reliability": reliability.numpy(),
        "positive_counts": positives.numpy().astype(np.int64), "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (torch.linalg.vector_norm(difference, dim=-1) < 1e-12).numpy(),
        "separation_temperature": float(separation_temperature), "reliability_floor": float(reliability_floor),
    }


def fit_evidence_calibrated_query_directions(
    support_features, support_labels, original_queries, support_evidence,
    etas=DEFAULT_ETAS, temperature=1.0, weight_floor=0.25,
):
    """Fit a native-grounding-evidence calibrated support query.

    ``support_evidence[n, c]`` is computed by the frozen native FLaS branch
    (before any query update) from the spatial grounding logits for support
    image ``n`` and class ``c``.  The positive and negative class means are
    then formed with class-conditional reliability weights: an image receives
    a high weight when its native evidence agrees with its support label and a
    low weight when it is ambiguous or contradictory.  This keeps the query
    transport in the model's own evidence space instead of introducing a
    separately trained confidence head.

    Evidence is robustly standardized per class with the median and MAD.  The
    signed evidence is transformed with a sigmoid and mixed with a uniform
    floor, so no support item can be discarded completely.  Everything is
    support-only and the released model remains frozen.
    """
    features, labels, original, evidence = map(_cpu_float,
        (support_features, support_labels, original_queries, support_evidence))
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if evidence.shape != labels.shape:
        raise ValueError("Support evidence must have shape [N,C]")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all() or not torch.isfinite(evidence).all():
        raise ValueError("Empty or nonfinite support/query/evidence")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(weight_floor)) or not 0 < float(weight_floor) <= 1:
        raise ValueError("weight_floor must lie in (0, 1]")
    norms = torch.linalg.vector_norm(features, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=1e-5):
        raise ValueError("Class-local support features must have unit vector norm")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each class requires positive and negative support examples")

    # Robustly normalize each class's native grounding evidence.  The tiny
    # MAD fallback is important for classes whose support maps have nearly
    # identical concentration.
    median = evidence.median(dim=0).values
    mad = (evidence - median).abs().median(dim=0).values
    scale = (1.4826 * mad).clamp_min(torch.finfo(evidence.dtype).eps)
    standardized = (evidence - median) / scale
    signed = standardized * (2 * labels - 1)
    agreement = torch.sigmoid(float(temperature) * signed)
    weights = float(weight_floor) + (1.0 - float(weight_floor)) * agreement
    # Normalize within each class and label group, preserving the unweighted
    # mean scale while allowing evidence to change which supports dominate.
    positive_weights = weights * labels
    negative_weights = weights * (1 - labels)
    pos_den = positive_weights.sum(0).clamp_min(torch.finfo(features.dtype).eps)
    neg_den = negative_weights.sum(0).clamp_min(torch.finfo(features.dtype).eps)
    positive_mean = (features * positive_weights[..., None]).sum(0) / pos_den[:, None]
    negative_mean = (features * negative_weights[..., None]).sum(0) / neg_den[:, None]
    difference = positive_mean - negative_mean
    direction_norms = torch.linalg.vector_norm(difference, dim=-1)
    directions = F.normalize(difference, p=2, dim=-1)
    normalized_query = F.normalize(original, p=2, dim=-1)
    queries = torch.stack([
        original.clone() if eta == 0 else F.normalize(
            normalized_query + eta * directions, p=2, dim=-1)
        for eta in etas
    ])
    if not torch.isfinite(queries).all():
        raise ValueError("Nonfinite evidence-calibrated refined queries")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(),
        "positive_means": positive_mean.numpy(), "negative_means": negative_mean.numpy(),
        "direction_norms": direction_norms.numpy(),
        "support_evidence": evidence.numpy(), "evidence_median": median.numpy(),
        "evidence_scale": scale.numpy(), "evidence_standardized": standardized.numpy(),
        "evidence_agreement": agreement.numpy(), "evidence_weights": weights.numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (direction_norms < 1e-12).numpy(),
        "evidence_temperature": float(temperature), "evidence_weight_floor": float(weight_floor),
    }


def fit_reaggregation_evidence_query_directions(
    support_vision, support_labels, original_queries, matcher, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1, temperature=1.0,
    weight_floor=0.25,
):
    """Evidence-calibrated query transport with native reaggregation.

    At each support-only fixed-point step the current query is passed through
    the original FLaS branch.  Both class-local contexts and their native
    grounding logits are recomputed, then the latter determines reliability
    weights for the positive/negative means.  The next query is anchored to
    the original text query, so the procedure is a bounded intervention rather
    than an unconstrained recurrent update.
    """
    import copy
    labels, original = map(_cpu_float, (support_labels, original_queries))
    if not support_vision or labels.ndim != 2 or original.ndim != 2:
        raise ValueError("Invalid support vision, labels, or queries")
    if labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if eta <= 0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be finite and positive")
    if max_iter < 1 or max_iter > 32 or patience < 1 or patience > max_iter:
        raise ValueError("Invalid fixed-point iteration settings")
    if not math.isfinite(float(query_tol)) or query_tol < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(weight_floor)) or not 0 < weight_floor <= 1:
        raise ValueError("weight_floor must lie in (0, 1]")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    device = matcher.device
    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    history, stable_steps = [], 0
    final = None

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    with torch.no_grad():
        for iteration in range(int(max_iter)):
            feature_batches, evidence_batches = [], []
            for prepared_cpu in support_vision:
                result = matcher.match_prepared(
                    on_device(prepared_cpu), query, return_maps=True, return_features=True)
                feature_batches.append(result["class_local"])
                grounding = result["grounding_map"]
                evidence_batches.append(grounding.amax(dim=-1) - grounding.mean(dim=-1))
            features = torch.cat(feature_batches, dim=0)
            evidence = torch.cat(evidence_batches, dim=0)
            median = evidence.median(dim=0).values
            mad = (evidence - median).abs().median(dim=0).values
            scale = (1.4826 * mad).clamp_min(torch.finfo(evidence.dtype).eps)
            standardized = (evidence - median) / scale
            signed = standardized * (2 * labels.to(device) - 1)
            agreement = torch.sigmoid(float(temperature) * signed)
            weights = float(weight_floor) + (1.0 - float(weight_floor)) * agreement
            pos_weights, neg_weights = weights * labels.to(device), weights * (1 - labels.to(device))
            pos_den = pos_weights.sum(0).clamp_min(torch.finfo(features.dtype).eps)
            neg_den = neg_weights.sum(0).clamp_min(torch.finfo(features.dtype).eps)
            pos = (features * pos_weights[..., None]).sum(0) / pos_den[:, None]
            neg = (features * neg_weights[..., None]).sum(0) / neg_den[:, None]
            difference = pos - neg
            directions = F.normalize(difference, p=2, dim=-1)
            next_query = F.normalize(anchor + float(eta) * directions, p=2, dim=-1)
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            history.append({"iteration": iteration,
                            "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "evidence_agreement_mean": float(agreement.mean().cpu()),
                            "evidence_agreement_min": float(agreement.min().cpu())})
            final = {"directions": directions, "positive_means": pos, "negative_means": neg,
                     "evidence": evidence, "median": median, "scale": scale,
                     "standardized": standardized, "agreement": agreement, "weights": weights}
            query = next_query
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                break
    if final is None or not torch.isfinite(query).all():
        raise ValueError("Reaggregation evidence transport produced nonfinite queries")
    return {
        "etas": np.asarray((0.0, float(eta)), dtype=np.float64),
        "original_queries": original.cpu().numpy(),
        "queries": np.stack((original.cpu().numpy(), query.cpu().numpy())),
        "directions": final["directions"].cpu().numpy(),
        "positive_means": final["positive_means"].cpu().numpy(),
        "negative_means": final["negative_means"].cpu().numpy(),
        "direction_norms": torch.linalg.vector_norm(final["positive_means"] - final["negative_means"], dim=-1).cpu().numpy(),
        "support_evidence": final["evidence"].cpu().numpy(),
        "evidence_median": final["median"].cpu().numpy(),
        "evidence_scale": final["scale"].cpu().numpy(),
        "evidence_standardized": final["standardized"].cpu().numpy(),
        "evidence_agreement": final["agreement"].cpu().numpy(),
        "evidence_weights": final["weights"].cpu().numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (torch.linalg.vector_norm(final["directions"], dim=-1) < 1e-12).cpu().numpy(),
        "evidence_temperature": float(temperature), "evidence_weight_floor": float(weight_floor),
        "reaggregation_iterations": len(history), "reaggregation_history": history,
        "reaggregation_query_tol": float(query_tol), "reaggregation_patience": int(patience),
    }


def fit_reaggregation_query_directions(support_vision, support_labels,
                                       original_queries, matcher, eta=0.3,
                                       max_iter=3, query_tol=1e-4,
                                       patience=1):
    """Fit a query while re-running query-conditioned support aggregation.

    ``support_vision`` is a sequence of the projected, frozen vision dicts
    returned by :meth:`QueryRefinementMatcher.encode_vision`.  At every fixed
    point step the original AlphaRAD branch is called with the current query;
    its newly aggregated class-local features define the next support
    direction.  Only support labels are consumed and no model parameter is
    updated.  The original query remains the update anchor, matching the
    one-step residual rule, while the direction is recomputed from the
    reaggregated support features.

    A support-only stopping rule ends the iteration when the normalized query
    update is below ``query_tol`` for ``patience`` consecutive steps.  The
    returned dictionary has the same query arrays as ``fit_query_directions``
    plus per-step diagnostics, allowing the evaluator to audit convergence.
    """
    import copy
    fit_started = time.perf_counter()

    features_labels = (_cpu_float(support_labels), _cpu_float(original_queries))
    labels, original = features_labels
    if not isinstance(support_vision, (list, tuple)) or not support_vision:
        raise ValueError("support_vision must contain at least one prepared batch")
    if labels.ndim != 2 or labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if original.ndim != 2 or not torch.isfinite(original).all():
        raise ValueError("Expected finite original queries [class, hidden]")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary and finite")
    eta_values = torch.as_tensor(eta, dtype=torch.float32).flatten()
    if eta_values.numel() == 1:
        eta_values = eta_values.repeat(original.shape[0])
    if eta_values.numel() != original.shape[0] or not torch.isfinite(eta_values).all() \
            or torch.any(eta_values < 0) or torch.all(eta_values <= 0):
        raise ValueError("eta must be a positive scalar or a nonnegative per-class vector")
    if int(max_iter) < 1 or int(max_iter) > 32:
        raise ValueError("max_iter must be in [1, 32]")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    if int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("patience must be in [1, max_iter]")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")

    # Copy the nested dict/list structure so callers may safely reuse cached
    # support features; tensors are moved to the matcher device per step.
    device = matcher.device
    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    directions = None
    initial_directions = None
    positive_means = negative_means = direction_norms = None
    history = []
    # Keep the actual fixed-point states (including q^(0)) so convergence
    # plots can be generated without rerunning the frozen encoder.  The first
    # state is intentionally the unnormalised query supplied by the evaluator;
    # AlphaRAD's native branch performs its own per-head normalisation.
    query_trajectory = [query.detach().cpu().clone()]
    previous_query = None
    previous_next_query = None
    previous_direction = None
    support_margin_baseline = None
    stable_steps = 0
    stop_reason = "max_iter"
    with torch.no_grad():
        for iteration in range(int(max_iter)):
            feature_batches, logit_batches = [], []
            for vision_cpu in support_vision:
                vision = on_device(vision_cpu)
                result = matcher.match_prepared(vision, query, return_features=True)
                feature_batches.append(result["class_local"])
                logit_batches.append(result["logits"])
            features = torch.cat(feature_batches, dim=0)
            logits = torch.cat(logit_batches, dim=0)
            if features.shape[:2] != labels.shape:
                raise ValueError("Reaggregated support feature/label shape mismatch")
            pos = (features * labels.to(device)[..., None]).sum(0) / positives.to(device)[:, None]
            neg = (features * (1 - labels.to(device))[..., None]).sum(0) / negatives.to(device)[:, None]
            difference = pos - neg
            direction_norms = torch.linalg.vector_norm(difference, dim=-1)
            directions = F.normalize(difference, p=2, dim=-1)
            if initial_directions is None:
                initial_directions = directions.detach().clone()
            next_query = F.normalize(anchor + eta_values.to(device)[:, None] * directions, p=2, dim=-1)
            normalized_current = F.normalize(query, p=2, dim=-1)
            delta = torch.linalg.vector_norm(next_query - normalized_current, dim=-1)
            direction_cosine = (directions * F.normalize(anchor, p=2, dim=-1)).sum(-1)
            pos_logits = (logits * labels.to(device)).sum(0) / positives.to(device)
            neg_logits = (logits * (1 - labels.to(device))).sum(0) / negatives.to(device)
            support_margin = pos_logits - neg_logits
            if support_margin_baseline is None:
                support_margin_baseline = support_margin.detach().clone()

            # The native branch defines the fixed-point map
            # F(q)=normalize(q_0 + eta*d(q)).  Estimate its local Lipschitz
            # ratio directly from consecutive support-only evaluations.  This
            # is a diagnostic only; it never affects the update or stopping
            # rule and therefore cannot leak test information.
            if previous_query is None:
                input_delta = None
                map_delta = None
                local_lipschitz = None
                direction_delta = None
            else:
                input_delta_tensor = torch.linalg.vector_norm(
                    normalized_current - previous_query, dim=-1)
                map_delta_tensor = torch.linalg.vector_norm(
                    next_query - previous_next_query, dim=-1)
                direction_delta_tensor = torch.linalg.vector_norm(
                    directions - previous_direction, dim=-1)
                input_delta = float(input_delta_tensor.mean().cpu())
                map_delta = float(map_delta_tensor.mean().cpu())
                direction_delta = float(direction_delta_tensor.mean().cpu())
                local_lipschitz = float(
                    (map_delta_tensor / input_delta_tensor.clamp_min(1e-8)).mean().cpu())

            support_utility_per_class = support_margin - support_margin_baseline
            support_utility = float(support_utility_per_class.mean().cpu())
            history.append({
                "iteration": iteration,
                "query_delta_max": float(delta.max().cpu()),
                "query_delta_mean": float(delta.mean().cpu()),
                "direction_alignment_mean": float(direction_cosine.mean().cpu()),
                "support_margin_mean": float(support_margin.mean().cpu()),
                "support_margin_min": float(support_margin.min().cpu()),
                "support_margin_per_class": support_margin.detach().cpu().tolist(),
                "support_utility": support_utility,
                "support_utility_per_class": support_utility_per_class.detach().cpu().tolist(),
                "input_delta_mean": input_delta,
                "map_delta_mean": map_delta,
                "direction_delta_mean": direction_delta,
                "local_lipschitz_mean": local_lipschitz,
                "zero_direction_classes": int((direction_norms < 1e-12).sum().cpu()),
            })
            previous_query = normalized_current.detach().clone()
            previous_next_query = next_query.detach().clone()
            previous_direction = directions.detach().clone()
            query = next_query
            query_trajectory.append(query.detach().cpu().clone())
            if float(delta.max()) <= float(query_tol):
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= int(patience):
                stop_reason = "query_stability"
                break

    # One final native support pass evaluates d(q_hat), so the reported gap is
    # the actual representation inconsistency between the initial and adapted
    # query-conditioned spaces rather than a proxy from the penultimate step.
    with torch.no_grad():
        final_batches = []
        for vision_cpu in support_vision:
            vision = on_device(vision_cpu)
            final_batches.append(matcher.match_prepared(vision, query, return_features=True)["class_local"])
        final_features = torch.cat(final_batches, dim=0)
        final_pos = (final_features * labels.to(device)[..., None]).sum(0) / positives.to(device)[:, None]
        final_neg = (final_features * (1 - labels.to(device))[..., None]).sum(0) / negatives.to(device)[:, None]
        final_directions = F.normalize(final_pos - final_neg, p=2, dim=-1)
    if directions is None or not torch.isfinite(query).all() or not torch.isfinite(final_directions).all():
        raise ValueError("Reaggregation produced nonfinite query vectors")
    gap_l2 = torch.linalg.vector_norm(initial_directions - final_directions, dim=-1)
    gap_cos = (F.normalize(initial_directions, dim=-1) * F.normalize(final_directions, dim=-1)).sum(-1)
    fit_elapsed = time.perf_counter() - fit_started
    return {
        "etas": np.asarray((0.0, float(eta_values.max())), dtype=np.float64),
        "original_queries": original.cpu().numpy(),
        "queries": np.stack((original.cpu().numpy(), query.cpu().numpy())),
        "directions": final_directions.cpu().numpy(),
        "positive_means": pos.cpu().numpy(), "negative_means": neg.cpu().numpy(),
        "direction_norms": direction_norms.cpu().numpy(),
        "positive_counts": positives.cpu().numpy().astype(np.int64),
        "negative_counts": negatives.cpu().numpy().astype(np.int64),
        "zero_direction_classes": (direction_norms < 1e-12).cpu().numpy(),
        "reaggregation_iterations": len(history),
        "reaggregation_max_iter": int(max_iter),
        "reaggregation_query_tol": float(query_tol),
        "reaggregation_patience": int(patience),
        "reaggregation_stop_reason": stop_reason,
        "reaggregation_history": history,
        "query_trajectory": torch.stack(query_trajectory).numpy(),
        "support_margin_baseline": support_margin_baseline.cpu().numpy(),
        "support_utility_final": float(history[-1]["support_utility"]),
        "local_lipschitz_max": float(max(
            (item["local_lipschitz_mean"] for item in history
             if item["local_lipschitz_mean"] is not None), default=0.0)),
        "selected_eta_per_class": eta_values.cpu().numpy(),
        "direction_initial": initial_directions.cpu().numpy(),
        "direction_final": final_directions.cpu().numpy(),
        "direction_gap_l2": gap_l2.cpu().numpy(),
        "direction_gap_l2_mean": float(gap_l2.mean().cpu()),
        "direction_gap_l2_median": float(gap_l2.median().cpu()),
        "direction_gap_l2_max": float(gap_l2.max().cpu()),
        "direction_gap_cos": gap_cos.cpu().numpy(),
        "direction_gap_cos_mean": float(gap_cos.mean().cpu()),
        "direction_gap_cos_min": float(gap_cos.min().cpu()),
        "native_reaggregation_passes": len(history) + 1,
        "fit_elapsed_seconds": float(fit_elapsed),
        "estimated_fit_flops": float((len(history) + 1) * len(labels) * original.shape[0] * original.shape[1] * 2),
    }


def _jackknife_consensus(features, labels, kappa=8.0):
    """Return a leave-one-support-out consensus direction and stability.

    The operation is deliberately performed independently for every class.  A
    leave-one-out direction is only formed when the held-out example leaves at
    least one positive and one negative support item, which is the setting of
    the released five-shot episodes.  ``stability`` is a bounded, support-only
    estimate of how much the direction survives removal of one support image.
    """
    n, classes, dim = features.shape
    positive = labels.sum(0)
    negative = n - positive
    if torch.any(positive < 2) or torch.any(negative < 2):
        raise ValueError("Jackknife fitting needs at least two positives and negatives per class")
    directions = torch.zeros(classes, n, dim, dtype=features.dtype, device=features.device)
    valid = torch.zeros(classes, n, dtype=torch.bool, device=features.device)
    consensus_all = torch.zeros(classes, dim, dtype=features.dtype, device=features.device)
    stability_all = torch.zeros(classes, dtype=features.dtype, device=features.device)
    for c in range(classes):
        pos_sum = (features[:, c] * labels[:, c, None]).sum(0)
        neg_sum = (features[:, c] * (1 - labels[:, c, None])).sum(0)
        for i in range(n):
            pcount = positive[c] - labels[i, c]
            ncount = negative[c] - (1 - labels[i, c])
            if pcount < 1 or ncount < 1:
                continue
            diff = (pos_sum - labels[i, c] * features[i, c]) / pcount
            diff = diff - (neg_sum - (1 - labels[i, c]) * features[i, c]) / ncount
            norm = torch.linalg.vector_norm(diff)
            if norm > 1e-12 and torch.isfinite(norm):
                directions[c, i] = diff / norm
                valid[c, i] = True
        if not valid[c].any():
            continue
        initial = directions[c, valid[c]].mean(0)
        initial = F.normalize(initial, p=2, dim=0)
        cosine = (directions[c] * initial).sum(-1).clamp(-1, 1)
        weights = torch.where(valid[c], torch.exp(float(kappa) * cosine), torch.zeros_like(cosine))
        consensus = (weights[:, None] * directions[c]).sum(0)
        consensus = F.normalize(consensus, p=2, dim=0)
        agreement = (directions[c] * consensus).sum(-1).clamp(0, 1)
        denom = valid[c].sum().clamp_min(1)
        stability = (agreement * valid[c]).sum() / denom
        consensus_all[c] = consensus
        stability_all[c] = stability
    if classes == 0:
        raise ValueError("No classes in support features")
    return consensus_all, stability_all


def fit_jackknife_stable_query_directions(
    support_features, support_labels, original_queries, etas=DEFAULT_ETAS,
    stability_power=1.0, kappa=8.0, geometry="linear",
):
    """Support-stable query transport using leave-one-support-out consensus.

    Unlike a single positive-minus-negative prototype residual, this rule
    measures directional survival under every support deletion.  The consensus
    direction is used for the update and its nonnegative jackknife agreement
    gates the step size.  This gives a frozen, gradient-free trust region that
    is robust to one mislabeled or acquisition-corrupted support image.
    """
    features, labels, original = map(_cpu_float,
        (support_features, support_labels, original_queries))
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not math.isfinite(float(stability_power)) or float(stability_power) < 0:
        raise ValueError("stability_power must be finite and nonnegative")
    if not math.isfinite(float(kappa)) or float(kappa) < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if geometry not in ("linear", "tangent"):
        raise ValueError("geometry must be linear or tangent")
    if not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Nonfinite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    directions, stability = _jackknife_consensus(features, labels, kappa=kappa)
    normalized_query = F.normalize(original, p=2, dim=-1)
    gate = stability.clamp(0, 1).pow(float(stability_power))
    weighted = directions * gate[:, None]
    queries = []
    for eta in etas:
        if eta == 0:
            queries.append(original.clone())
            continue
        if geometry == "linear":
            queries.append(F.normalize(normalized_query + eta * weighted, p=2, dim=-1))
            continue
        tangent = weighted - (weighted * normalized_query).sum(-1, keepdim=True) * normalized_query
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        tangent_unit = F.normalize(tangent, p=2, dim=-1)
        theta = float(eta) * tangent_norm.squeeze(-1)
        queries.append(torch.cos(theta)[:, None] * normalized_query
                      + torch.sin(theta)[:, None] * tangent_unit)
    queries = torch.stack(queries)
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if not torch.isfinite(queries).all():
        raise ValueError("Nonfinite jackknife refined queries")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(), "weighted_directions": weighted.numpy(),
        "support_stability": stability.numpy(), "stability_power": float(stability_power),
        "jackknife_kappa": float(kappa), "geometry": geometry,
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (torch.linalg.vector_norm(directions, dim=-1) < 1e-12).numpy(),
    }


def fit_token_discriminative_query_directions(
    support_head_features, support_labels, original_queries, num_heads,
    etas=DEFAULT_ETAS, reliability_temperature=1.0, reliability_power=1.0,
):
    """Native-head support transport for query-dependent RadZero/AlphaRAD.

    The released classifier normalizes and scores every RadZero head
    independently before averaging.  TDQA therefore estimates a
    positive-minus-negative direction in each head subspace, then gates the
    update with a support-only Fisher-style utility (between-class direction
    divided by pooled within-class dispersion).  The gate is class-specific,
    mean-one across heads, and the update is scaled by ``1/sqrt(H)`` so its
    total magnitude remains comparable to a global residual update.
    """
    features, labels, original = map(_cpu_float,
        (support_head_features, support_labels, original_queries))
    etas = validate_etas(etas)
    if features.ndim != 4 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support head features [N,C,H,D] and labels [N,C]")
    n, classes, heads, head_dim = features.shape
    if int(num_heads) != heads:
        raise ValueError("num_heads does not match support feature heads")
    if original.ndim == 3:
        if original.shape != (classes, heads, head_dim):
            raise ValueError("Query and support head dimensions differ")
        original_heads = original.clone()
        original_flat = original.reshape(classes, heads * head_dim)
    elif original.ndim == 2:
        if original.shape != (classes, heads * head_dim):
            raise ValueError("Query and support dimensions differ")
        original_flat = original.clone()
        original_heads = original.reshape(classes, heads, head_dim)
    else:
        raise ValueError("Expected original queries [C,D] or [C,H,D]")
    if n == 0 or not torch.isfinite(features).all() or not torch.isfinite(original_flat).all():
        raise ValueError("Empty or nonfinite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    if not math.isfinite(float(reliability_temperature)) or reliability_temperature <= 0:
        raise ValueError("reliability_temperature must be finite and positive")
    if not math.isfinite(float(reliability_power)) or reliability_power < 0:
        raise ValueError("reliability_power must be finite and nonnegative")
    head_norms = torch.linalg.vector_norm(features, dim=-1)
    if not torch.allclose(head_norms, torch.ones_like(head_norms), atol=2e-4, rtol=2e-4):
        raise ValueError("Class-local head features must have unit norm")
    positive = labels.sum(0)
    negative = n - positive
    if torch.any(positive == 0) or torch.any(negative == 0):
        raise ValueError("Each class needs positive and negative support examples")
    positive_mean = (features * labels[..., None, None]).sum(0) / positive[:, None, None]
    negative_mean = (features * (1 - labels)[..., None, None]).sum(0) / negative[:, None, None]
    difference = positive_mean - negative_mean
    direction_norms = torch.linalg.vector_norm(difference, dim=-1)
    directions = F.normalize(difference, p=2, dim=-1)
    # Pooled within-class dispersion in each native head.  This is stable for
    # five-shot support because the denominator is clamped for singleton class
    # subsets and the final precision-like ratio is regularized.
    pos_residual = (features - positive_mean[None]) * labels[..., None, None]
    neg_residual = (features - negative_mean[None]) * (1 - labels)[..., None, None]
    pos_var = pos_residual.square().sum(0) / positive[:, None, None].clamp_min(1)
    neg_var = neg_residual.square().sum(0) / negative[:, None, None].clamp_min(1)
    within = torch.sqrt((pos_var + neg_var).mean(-1).clamp_min(1e-8))
    utility = (direction_norms / within.clamp_min(1e-4)).clamp_min(0)
    gates = heads * F.softmax(utility / float(reliability_temperature), dim=-1)
    if reliability_power != 1.0:
        gates = gates.pow(float(reliability_power))
        gates = gates / gates.mean(-1, keepdim=True).clamp_min(1e-6)
    weighted = directions * gates[..., None] / math.sqrt(float(heads))
    normalized_query = F.normalize(original_heads, p=2, dim=-1)
    queries = []
    for eta in etas:
        if eta == 0:
            queries.append(original_flat.clone())
        else:
            adapted = F.normalize(normalized_query + float(eta) * weighted, p=2, dim=-1)
            queries.append(adapted.reshape(classes, heads * head_dim))
    queries = torch.stack(queries)
    if not torch.isfinite(queries).all():
        raise ValueError("Nonfinite TDQA refined queries")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original_flat.numpy(), "queries": queries.numpy(),
        "directions": directions.reshape(classes, heads * head_dim).numpy(),
        "head_directions": directions.numpy(), "head_weighted_directions": weighted.numpy(),
        "positive_means": positive_mean.reshape(classes, heads * head_dim).numpy(),
        "negative_means": negative_mean.reshape(classes, heads * head_dim).numpy(),
        "direction_norms": direction_norms.mean(-1).numpy(),
        "head_direction_norms": direction_norms.numpy(),
        "head_utility": utility.numpy(), "head_gates": gates.numpy(),
        "within_dispersion": within.numpy(),
        "positive_counts": positive.numpy().astype(np.int64),
        "negative_counts": negative.numpy().astype(np.int64),
        "num_heads": int(heads), "head_dim": int(head_dim),
        "reliability_temperature": float(reliability_temperature),
        "reliability_power": float(reliability_power),
        "zero_direction_classes": (direction_norms.mean(-1) < 1e-12).numpy(),
    }


def fit_reaggregation_token_discriminative_query_directions(
    support_vision, support_labels, original_queries, matcher, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1, reliability_temperature=1.0,
    reliability_power=1.0,
):
    """Fixed-point TDQA: recompute native head evidence after each transport.

    This combines the native per-head update with AlphaRAD's query-conditioned
    aggregation.  The support contexts are regenerated at the current query,
    head utilities are refit in that same representation, and the next query
    is anchored to the original query.  The stopping rule is support-only.
    """
    import copy
    labels = _cpu_float(support_labels)
    original = _cpu_float(original_queries)
    if not support_vision or labels.ndim != 2 or original.ndim != 2:
        raise ValueError("Invalid support vision, labels, or queries")
    if eta <= 0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be finite and positive")
    if max_iter < 1 or max_iter > 32 or patience < 1 or patience > max_iter:
        raise ValueError("Invalid fixed-point iteration settings")
    device = matcher.device
    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    stable_steps, history, fit = 0, [], None

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    with torch.no_grad():
        for iteration in range(int(max_iter)):
            batches = []
            for prepared_cpu in support_vision:
                result = matcher.match_prepared(on_device(prepared_cpu), query, return_features=True)
                batches.append(result["class_local_heads"])
            head_features = torch.cat(batches, dim=0).cpu()
            fit = fit_token_discriminative_query_directions(
                head_features, labels, original, matcher.model.num_radzero_heads,
                etas=(0., eta), reliability_temperature=reliability_temperature,
                reliability_power=reliability_power)
            next_query = torch.as_tensor(fit["queries"][1], dtype=torch.float32, device=device)
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            history.append({"iteration": iteration,
                            "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "head_gate_mean": float(np.asarray(fit["head_gates"]).mean()),
                            "head_gate_max": float(np.asarray(fit["head_gates"]).max())})
            query = next_query
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                break
    if fit is None:
        raise ValueError("No reaggregated TDQA direction was produced")
    fit["queries"] = np.stack((original.numpy(), query.cpu().numpy()))
    fit["reaggregation_iterations"] = len(history)
    fit["reaggregation_max_iter"] = int(max_iter)
    fit["reaggregation_query_tol"] = float(query_tol)
    fit["reaggregation_patience"] = int(patience)
    fit["reaggregation_history"] = history
    fit["reaggregation_stop_reason"] = "query_stability" if stable_steps >= int(patience) else "max_iter"
    return fit


def fit_reaggregation_jackknife_query_directions(
    support_vision, support_labels, original_queries, matcher, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1, stability_power=1.0,
    kappa=8.0, geometry="tangent",
):
    """Query-conditioned jackknife consensus fixed point.

    This is the full native-path variant: support contexts are reaggregated at
    the current query, then leave-one-support-out directions are recomputed in
    that same representation before the next transport step.  The stability
    gate and optional tangent geometry are support-only; model weights remain
    frozen.
    """
    import copy
    labels = _cpu_float(support_labels)
    original = _cpu_float(original_queries)
    if not support_vision or labels.ndim != 2 or original.ndim != 2:
        raise ValueError("Invalid support vision, labels, or queries")
    if labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if geometry not in ("linear", "tangent"):
        raise ValueError("geometry must be linear or tangent")
    if max_iter < 1 or max_iter > 32 or patience < 1 or patience > max_iter:
        raise ValueError("Invalid fixed-point iteration settings")
    if eta <= 0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be finite and positive")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives < 2) or torch.any(negatives < 2):
        raise ValueError("Jackknife fitting needs at least two positives and negatives")
    device = matcher.device
    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    history, stable_steps = [], 0
    directions = stability = None

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    with torch.no_grad():
        for iteration in range(int(max_iter)):
            batches = []
            for prepared_cpu in support_vision:
                result = matcher.match_prepared(on_device(prepared_cpu), query, return_features=True)
                batches.append(result["class_local"])
            features = torch.cat(batches, dim=0)
            directions, stability = _jackknife_consensus(features, labels.to(device), kappa=kappa)
            weighted = directions * stability.clamp(0, 1).pow(float(stability_power))[:, None]
            linear = F.normalize(anchor + float(eta) * weighted, p=2, dim=-1)
            if geometry == "linear":
                next_query = linear
            else:
                tangent = weighted - (weighted * anchor).sum(-1, keepdim=True) * anchor
                tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
                tangent_unit = F.normalize(tangent, p=2, dim=-1)
                theta = float(eta) * tangent_norm
                next_query = torch.cos(theta)[:, None] * anchor + torch.sin(theta)[:, None] * tangent_unit
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            history.append({"iteration": iteration,
                            "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "support_stability_mean": float(stability.mean().cpu()),
                            "support_stability_min": float(stability.min().cpu())})
            query = next_query
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                break
    if directions is None:
        raise ValueError("No reaggregated support direction was produced")
    return {
        "etas": np.asarray((0.0, float(eta)), dtype=np.float64),
        "original_queries": original.cpu().numpy(),
        "queries": np.stack((original.cpu().numpy(), query.cpu().numpy())),
        "directions": directions.cpu().numpy(),
        "support_stability": stability.cpu().numpy(),
        "stability_power": float(stability_power), "jackknife_kappa": float(kappa),
        "geometry": geometry, "reaggregation_iterations": len(history),
        "reaggregation_history": history,
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (torch.linalg.vector_norm(directions, dim=-1) < 1e-12).cpu().numpy(),
    }


def fit_contrastive_grounding_query_directions(
    support_vision, support_labels, original_queries, matcher, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1, token_weight=1.0,
    adaptive_token_gate=False, geometry="linear", global_jackknife=False,
    support_margin_gate=False, support_margin_grid=False, support_cv_grid=False,
):
    """Transport queries with positive-minus-negative native token maps."""
    import copy
    labels, original = map(_cpu_float, (support_labels, original_queries))
    if not support_vision or labels.ndim != 2 or original.ndim != 2:
        raise ValueError("Invalid support vision, labels, or queries")
    if labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if not math.isfinite(float(eta)) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if not math.isfinite(float(token_weight)) or token_weight < 0:
        raise ValueError("token_weight must be finite and nonnegative")
    if geometry not in ("linear", "tangent", "adaptive"):
        raise ValueError("geometry must be linear, tangent, or adaptive")
    if max_iter < 1 or max_iter > 32 or patience < 1 or patience > max_iter:
        raise ValueError("Invalid fixed-point iteration settings")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    device = matcher.device
    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    history, stable_steps, final = [], 0, None

    def on_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(on_device(item) for item in value)
        return copy.deepcopy(value)

    with torch.no_grad():
        for iteration in range(int(max_iter)):
            attn_batches, value_batches, global_batches = [], [], []
            for prepared_cpu in support_vision:
                result = matcher.match_prepared(on_device(prepared_cpu), query,
                                                return_features=True,
                                                return_token_features=True)
                attn_batches.append(result["attention_heads"])
                value_batches.append(result["image_values_heads"])
                global_batches.append(result["class_local"])
            attention = torch.cat(attn_batches, dim=0)
            values = torch.cat(value_batches, dim=0)
            global_features = torch.cat(global_batches, dim=0)
            labels_device = labels.to(device)
            pos_map = (attention * labels_device[:, :, None, None]).sum(0) / positives.to(device)[:, None, None]
            neg_map = (attention * (1 - labels_device)[:, :, None, None]).sum(0) / negatives.to(device)[:, None, None]
            delta_map = pos_map - neg_map
            delta_map = delta_map / torch.linalg.vector_norm(delta_map, dim=1, keepdim=True).clamp_min(1e-6)
            contrast = torch.einsum("bckh,ckh,bkhd->bchd", attention, delta_map, values)
            contrast = F.normalize(contrast, p=2, dim=-1).flatten(start_dim=-2)
            contrast = F.normalize(contrast, p=2, dim=-1)
            pos = (contrast * labels_device[..., None]).sum(0) / positives.to(device)[:, None]
            neg = (contrast * (1 - labels_device)[..., None]).sum(0) / negatives.to(device)[:, None]
            difference = pos - neg
            token_directions = F.normalize(difference, p=2, dim=-1)
            global_pos = (global_features * labels_device[..., None]).sum(0) / positives.to(device)[:, None]
            global_neg = (global_features * (1 - labels_device)[..., None]).sum(0) / negatives.to(device)[:, None]
            global_directions = F.normalize(global_pos - global_neg, p=2, dim=-1)
            if global_jackknife:
                global_directions, global_stability = _jackknife_consensus(
                    global_features, labels_device, kappa=8.0)
                global_stability = global_stability.clamp(0, 1)
            else:
                global_stability = torch.ones_like(global_directions[:, 0])
            alignment = (global_directions * token_directions).sum(-1).clamp(-1, 1)
            positives_device, negatives_device = positives.to(device), negatives.to(device)
            if adaptive_token_gate:
                token_gate = alignment.clamp_min(0) * (alignment >= 0.1).to(alignment.dtype)
            else:
                token_gate = torch.ones_like(alignment)
            step_direction = (global_stability[:, None] * global_directions
                              + float(token_weight) * token_gate[:, None] * token_directions)
            directions = F.normalize(step_direction, p=2, dim=-1)
            linear = F.normalize(anchor + float(eta) * step_direction, p=2, dim=-1)
            tangent = step_direction - (step_direction * anchor).sum(-1, keepdim=True) * anchor
            tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
            tangent_unit = F.normalize(tangent, p=2, dim=-1)
            theta = float(eta) * tangent_norm
            tangent_query = torch.cos(theta)[:, None] * anchor + torch.sin(theta)[:, None] * tangent_unit
            if geometry == "linear":
                next_query = linear
            elif geometry == "tangent":
                next_query = tangent_query
            else:
                # Support-only path selection: use tangent transport only when
                # token and global contrastive directions agree; otherwise
                # preserve the stable linear reaggregation path.
                use_tangent = (alignment >= 0.1)[:, None]
                next_query = torch.where(use_tangent, tangent_query, linear)
            if support_cv_grid:
                # Leave-one-support-out utility selection.  Each class picks
                # among global, half-token, and full-token transport using
                # only held-out support examples; the test set is never used
                # to select a path or a weight.
                global_sum_pos = (global_features * labels_device[..., None]).sum(0)
                global_sum_neg = (global_features * (1 - labels_device)[..., None]).sum(0)
                token_sum_pos = (contrast * labels_device[..., None]).sum(0)
                token_sum_neg = (contrast * (1 - labels_device)[..., None]).sum(0)
                weights = (0.0, 0.5 * float(token_weight), float(token_weight))
                eta_grid = tuple(dict.fromkeys((float(eta), 0.5)))
                utility = torch.zeros(len(eta_grid), 3, len(query), device=device)
                valid_counts = torch.zeros(len(query), device=device)
                for sample_idx in range(len(global_features)):
                    for class_idx in range(len(query)):
                        y = labels_device[sample_idx, class_idx]
                        pcount = positives_device[class_idx] - y
                        ncount = negatives_device[class_idx] - (1 - y)
                        if pcount <= 0 or ncount <= 0:
                            continue
                        gd = F.normalize(
                            (global_sum_pos[class_idx] - y * global_features[sample_idx, class_idx]) / pcount
                            - (global_sum_neg[class_idx] - (1 - y) * global_features[sample_idx, class_idx]) / ncount,
                            p=2, dim=0)
                        td = F.normalize(
                            (token_sum_pos[class_idx] - y * contrast[sample_idx, class_idx]) / pcount
                            - (token_sum_neg[class_idx] - (1 - y) * contrast[sample_idx, class_idx]) / ncount,
                            p=2, dim=0)
                        align = torch.dot(gd, td).clamp(-1, 1)
                        gate = align.clamp_min(0) * (align >= 0.1).to(align.dtype)
                        for eta_idx, eta_value in enumerate(eta_grid):
                          for weight_idx, weight in enumerate(weights):
                            step = F.normalize(gd + weight * gate * td, p=2, dim=0)
                            linear_i = F.normalize(anchor[class_idx] + eta_value * step, p=2, dim=0)
                            tangent_i = step - torch.dot(step, anchor[class_idx]) * anchor[class_idx]
                            tangent_unit_i = F.normalize(tangent_i, p=2, dim=0)
                            tangent_i = (torch.cos(eta_value * torch.linalg.vector_norm(tangent_i)) * anchor[class_idx]
                                         + torch.sin(eta_value * torch.linalg.vector_norm(tangent_i)) * tangent_unit_i)
                            q_i = tangent_i if align >= 0.1 else linear_i
                            signed_score = (2 * y - 1) * torch.dot(global_features[sample_idx, class_idx], q_i)
                            # Leave-one-out logistic support utility is less
                            # rank-only than a raw margin and better reflects
                            # the precision/recall tradeoff in few-shot data.
                            utility[eta_idx, weight_idx, class_idx] -= F.softplus(-signed_score)
                        valid_counts[class_idx] += 1
                selected_flat = utility.reshape(-1, len(query)).argmax(0)
                selected_eta_idx = selected_flat // len(weights)
                selected_weight_idx = selected_flat % len(weights)
                selected_eta = torch.as_tensor(eta_grid, device=device)[selected_eta_idx]
                # A larger transport radius is exposed only when both sides
                # have at least five support examples; sparse classes retain
                # the conservative eta supplied by the caller.
                eta_min_count = 25 if len(labels) > 20 else 5
                eta_is_supported = ((torch.minimum(positives_device, negatives_device) >= eta_min_count)
                                    & ((alignment < 0.05) | (alignment >= 0.1)))
                selected_eta = torch.where(eta_is_supported, selected_eta,
                                            torch.full_like(selected_eta, float(eta)))
                # Require a meaningful held-out support utility gain before
                # spending the larger radius; tiny gains are treated as
                # sampling noise and keep the conservative eta.
                base_utility = utility[0, 0]
                chosen_utility = utility[selected_eta_idx, selected_weight_idx,
                                         torch.arange(len(query), device=device)]
                meaningful_gain = (chosen_utility - base_utility) >= 0.05
                selected_eta = torch.where(meaningful_gain, selected_eta,
                                            torch.full_like(selected_eta, float(eta)))
                selected_weight = torch.as_tensor(weights, device=device)[selected_weight_idx]
                global_step = F.normalize(global_stability[:, None] * global_directions, p=2, dim=-1)
                step = F.normalize(global_step + selected_weight[:, None] * token_gate[:, None] * token_directions, p=2, dim=-1)
                linear = F.normalize(anchor + selected_eta[:, None] * step, p=2, dim=-1)
                tangent = step - (step * anchor).sum(-1, keepdim=True) * anchor
                tangent_unit = F.normalize(tangent, p=2, dim=-1)
                tangent_query = torch.cos(selected_eta * torch.linalg.vector_norm(tangent, dim=-1))[:, None] * anchor + torch.sin(selected_eta * torch.linalg.vector_norm(tangent, dim=-1))[:, None] * tangent_unit
                next_query = torch.where((alignment >= 0.1)[:, None], tangent_query, linear)
                choose_sgtt = selected_weight_idx > 0
                transport_choice = selected_weight_idx
                selected_weight_record = selected_weight
                selected_eta_record = selected_eta
            elif support_margin_gate or support_margin_grid:
                # Support-only utility gate: compare the native support
                # margin of the token transport against the global path using
                # the current query-conditioned class-local features.  This
                # keeps token transport only when it improves the frozen
                # support separation, with no test-label access.
                global_step = F.normalize(global_stability[:, None] * global_directions, p=2, dim=-1)
                global_linear = F.normalize(anchor + float(eta) * global_step, p=2, dim=-1)
                candidate_queries = [global_linear]
                if support_margin_grid:
                    half_step = F.normalize(global_step + 0.5 * float(token_weight) * token_gate[:, None] * token_directions, p=2, dim=-1)
                    half_linear = F.normalize(anchor + float(eta) * half_step, p=2, dim=-1)
                    half_tangent = half_step - (half_step * anchor).sum(-1, keepdim=True) * anchor
                    half_unit = F.normalize(half_tangent, p=2, dim=-1)
                    half_query = torch.cos(float(eta) * torch.linalg.vector_norm(half_tangent, dim=-1))[:, None] * anchor + torch.sin(float(eta) * torch.linalg.vector_norm(half_tangent, dim=-1))[:, None] * half_unit
                    half_query = torch.where((alignment >= 0.1)[:, None], half_query, half_linear)
                    candidate_queries.append(half_query)
                candidate_queries.append(next_query)
                candidate_scores = torch.stack([torch.einsum("bcd,cd->bc", global_features, q) for q in candidate_queries])
                candidate_margins = ((candidate_scores * labels_device[None]).sum(1) / positives.to(device)[None]
                                     - (candidate_scores * (1 - labels_device)[None]).sum(1) / negatives.to(device)[None])
                choice = candidate_margins.argmax(0)
                next_query = torch.stack(candidate_queries)[choice, torch.arange(len(choice), device=device)]
                choose_sgtt = choice > 0
                if support_margin_grid:
                    transport_choice = choice
                else:
                    transport_choice = choose_sgtt.to(torch.long) * 2
                selected_weight_record = torch.where(transport_choice == 0, torch.zeros_like(alignment),
                                                     torch.where(transport_choice == 1,
                                                                 torch.full_like(alignment, 0.5 * float(token_weight)),
                                                                 torch.full_like(alignment, float(token_weight))))
                selected_eta_record = torch.full_like(alignment, float(eta))
            else:
                choose_sgtt = torch.ones_like(alignment, dtype=torch.bool)
                transport_choice = torch.full_like(alignment, 2, dtype=torch.long)
                selected_weight_record = torch.full_like(alignment, float(token_weight))
                selected_eta_record = torch.full_like(alignment, float(eta))
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            history.append({"iteration": iteration,
                            "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "token_contrast_norm_mean": float(torch.linalg.vector_norm(difference, dim=-1).mean().cpu())})
            final = {"directions": directions, "positive_means": pos, "negative_means": neg,
                     "delta_map": delta_map, "token_directions": token_directions,
                     "global_directions": global_directions, "token_alignment": alignment,
                     "token_gate": token_gate, "global_stability": global_stability,
                     "support_path_gate": choose_sgtt,
                     "transport_choice": transport_choice,
                     "selected_token_weight": selected_weight_record,
                     "selected_eta": selected_eta_record,
                     "support_cv_utility": utility if support_cv_grid else None}
            query = next_query
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                break
    if final is None or not torch.isfinite(query).all():
        raise ValueError("Contrastive grounding transport produced nonfinite queries")
    return {
        "etas": np.asarray((0.0, float(eta)), dtype=np.float64),
        "original_queries": original.numpy(), "queries": np.stack((original.cpu().numpy(), query.cpu().numpy())),
        "directions": final["directions"].cpu().numpy(),
        "positive_means": final["positive_means"].cpu().numpy(),
        "negative_means": final["negative_means"].cpu().numpy(),
        "token_delta_map": final["delta_map"].cpu().numpy(),
        "token_directions": final["token_directions"].cpu().numpy(),
        "global_directions": final["global_directions"].cpu().numpy(),
        "token_weight": float(token_weight),
        "adaptive_token_gate": bool(adaptive_token_gate),
        "geometry": geometry,
        "global_jackknife": bool(global_jackknife),
        "global_stability": final["global_stability"].cpu().numpy() if "global_stability" in final else np.ones(len(final["directions"])),
        "token_alignment": final["token_alignment"].cpu().numpy(),
        "token_gate": final["token_gate"].cpu().numpy(),
        "support_path_gate": final["support_path_gate"].cpu().numpy(),
        "support_margin_gate": bool(support_margin_gate),
        "support_margin_grid": bool(support_margin_grid),
        "transport_choice": final["transport_choice"].cpu().numpy(),
        "selected_token_weight": final["selected_token_weight"].cpu().numpy(),
        "selected_eta": final["selected_eta"].cpu().numpy(),
        "support_cv_utility": (final["support_cv_utility"].cpu().numpy()
                               if final.get("support_cv_utility") is not None else None),
        "support_cv_grid": bool(support_cv_grid),
        "positive_counts": positives.cpu().numpy().astype(np.int64),
        "negative_counts": negatives.cpu().numpy().astype(np.int64),
        "zero_direction_classes": (torch.linalg.vector_norm(final["directions"], dim=-1) < 1e-12).cpu().numpy(),
        "reaggregation_iterations": len(history), "reaggregation_max_iter": int(max_iter),
        "reaggregation_query_tol": float(query_tol), "reaggregation_patience": int(patience),
        "reaggregation_stop_reason": "query_stability" if stable_steps >= int(patience) else "max_iter",
        "reaggregation_history": history,
    }


def fit_uncertainty_query_directions(
    support_features,
    support_labels,
    original_queries,
    etas=DEFAULT_ETAS,
    covariance_shrinkage=0.2,
    ridge=1e-3,
    reliability_power=1.0,
):
    """Fit support directions with covariance shrinkage and reliability weighting.

    This is a frozen, support-only alternative to ``fit_query_directions``.  A
    class's positive and negative context means are compared with a pooled
    covariance estimate, then whitened using a regularized precision matrix:

    ``v_c = (Sigma_c + ridge * tau_c I)^-1 (mu+_c - mu-_c)``.

    ``Sigma_c`` is a within-class pooled covariance.  Since five-shot support
    sets make the empirical covariance rank deficient, Ledoit--Wolf-style
    shrinkage towards its trace target is applied before the ridge term.  The
    residual standard error of the mean difference provides a scalar support
    reliability; it scales only the update magnitude and leaves the direction
    itself unchanged.  No test images, labels, gradients, or model parameters
    enter this fit.

    Parameters are deliberately explicit so an experiment can audit the exact
    regularization.  ``covariance_shrinkage`` lies in [0, 1], ``ridge`` is a
    nonnegative multiple of the trace target, and ``reliability_power`` allows
    a controlled ablation (1.0 is the proposed rule; 0 reproduces an
    unweighted Mahalanobis direction).
    """
    features, labels, original = map(_cpu_float,
        (support_features, support_labels, original_queries))
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Empty or nonfinite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    if not math.isfinite(float(covariance_shrinkage)) or not 0 <= float(covariance_shrinkage) <= 1:
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if not math.isfinite(float(ridge)) or float(ridge) < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if not math.isfinite(float(reliability_power)) or float(reliability_power) < 0:
        raise ValueError("reliability_power must be finite and nonnegative")
    norms = torch.linalg.vector_norm(features, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=1e-5):
        raise ValueError("Class-local support features must have unit vector norm")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each class requires positive and negative support examples")

    n_classes, dim = features.shape[1], features.shape[2]
    positive_mean = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_mean = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = positive_mean - negative_mean
    raw_norms = torch.linalg.vector_norm(difference, dim=-1)
    target_scale = torch.empty(n_classes, dtype=features.dtype)
    directions = torch.empty_like(difference)
    reliabilities = torch.empty(n_classes, dtype=features.dtype)
    mahalanobis_norms = torch.empty(n_classes, dtype=features.dtype)
    covariance_traces = torch.empty(n_classes, dtype=features.dtype)

    # A batched solve would require materializing C x D x D matrices.  The
    # class loop keeps peak memory bounded and is negligible next to the image
    # encoder (there are only a handful of classes and ten support examples).
    eye = torch.eye(dim, dtype=features.dtype)
    for class_idx in range(n_classes):
        positive = features[labels[:, class_idx] == 1, class_idx]
        negative = features[labels[:, class_idx] == 0, class_idx]
        pos_residual = positive - positive_mean[class_idx]
        neg_residual = negative - negative_mean[class_idx]
        residuals = torch.cat((pos_residual, neg_residual), dim=0)
        denominator = max(int(residuals.shape[0]) - 2, 1)
        covariance = residuals.transpose(0, 1).matmul(residuals) / float(denominator)
        # Shrink towards an isotropic target.  The target is detached from the
        # class direction, so all uncertainty statistics remain support-only.
        scale = torch.trace(covariance) / float(dim)
        scale = torch.clamp(scale, min=torch.finfo(features.dtype).eps)
        target_scale[class_idx] = scale
        covariance_traces[class_idx] = torch.trace(covariance)
        covariance = (1.0 - float(covariance_shrinkage)) * covariance
        covariance = covariance + float(covariance_shrinkage) * scale * eye
        covariance = covariance + float(ridge) * scale * eye
        # Cholesky is more stable than explicit matrix inversion and gives a
        # clear failure if an invalid user-supplied regularizer is requested.
        try:
            chol = torch.linalg.cholesky(covariance)
            whitened = torch.cholesky_solve(
                difference[class_idx, :, None], chol).squeeze(-1)
        except RuntimeError as exc:
            raise ValueError("Covariance regularization produced a singular matrix") from exc
        mah_sq = torch.dot(difference[class_idx], whitened).clamp_min(0)
        mahalanobis_norms[class_idx] = torch.sqrt(mah_sq)
        directions[class_idx] = F.normalize(whitened, p=2, dim=0)
        # Standard error of the positive-minus-negative mean along the learned
        # direction.  This drops toward zero for a stable, well-separated class.
        standard_error_sq = torch.dot(
            directions[class_idx], covariance.matmul(directions[class_idx]))
        standard_error = torch.sqrt(standard_error_sq.clamp_min(0) *
                                    (1.0 / positives[class_idx] + 1.0 / negatives[class_idx]))
        reliability = raw_norms[class_idx] / (raw_norms[class_idx] + standard_error + 1e-8)
        reliabilities[class_idx] = reliability.clamp(0, 1).pow(float(reliability_power))

    normalized_query = F.normalize(original, p=2, dim=-1)
    weighted_directions = directions * reliabilities[:, None]
    queries = torch.stack([
        original.clone() if eta == 0 else F.normalize(
            normalized_query + eta * weighted_directions, p=2, dim=-1)
        for eta in etas
    ])
    if not torch.isfinite(queries).all():
        raise ValueError("Nonfinite uncertainty-aware refined queries")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(),
        "weighted_directions": weighted_directions.numpy(),
        "positive_means": positive_mean.numpy(),
        "negative_means": negative_mean.numpy(),
        "direction_norms": raw_norms.numpy(),
        "mahalanobis_norms": mahalanobis_norms.numpy(),
        "support_reliability": reliabilities.numpy(),
        "covariance_trace": covariance_traces.numpy(),
        "covariance_target_scale": target_scale.numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (raw_norms < 1e-12).numpy(),
        "covariance_shrinkage": float(covariance_shrinkage),
        "ridge": float(ridge),
        "reliability_power": float(reliability_power),
    }


@torch.no_grad()
def original_queries_from_evaluator(evaluator, class_names):
    """Return raw projected text queries and exactly the author's class prompts."""
    from cxrclip.evaluator_utils import disease_name_mapping
    from cxrclip.util.utils import curate_dqn_input_labels

    prompts = [curate_dqn_input_labels(
        disease_name_mapping(name), evaluator.prompt_template,
        evaluator.observation_explanation,
        evaluator.anonymize_observation_explanation,
    ) for name in class_names]
    encoded = evaluator.encode_text(
        prompts, text_max_length=evaluator.config["base"]["dqn_label_max_length"])
    queries = np.asarray(encoded["cls_emb_raw"], dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if queries.ndim != 2 or len(queries) != len(class_names) or not np.isfinite(queries).all():
        raise ValueError("Unexpected original text-query shape or values")
    return queries.copy(), prompts


class QueryRefinementMatcher(nn.Module):
    """One frozen image forward, then the author's FLaS for each fixed query.

    ``grounding_map`` is the unchanged scaled cosine map [B,C,K], including
    CLS if the model attends CLS. It goes directly into the author's metrics.
    ``similarity_scores`` removes CLS and reshapes to [B,C,S,S] for exporting.
    Neither is a softmax heatmap. ``class_local`` is the original per-head unit
    context, concatenated in head order and normalized over its full D-vector.
    """

    def __init__(self, model, queries, etas=DEFAULT_ETAS, use_all_image_tokens=True):
        super().__init__()
        model = model.module if hasattr(model, "module") else model
        if not getattr(model, "enable_multihead_radzero_forward_pass", False):
            raise ValueError("Requires the released multi-head AlphaRAD FLaS model")
        self.model = model.requires_grad_(False).eval()
        self.etas = validate_etas(etas)
        self.use_all_image_tokens = bool(use_all_image_tokens)
        self.register_buffer("queries", _cpu_float(queries).to(next(model.parameters()).device))
        if self.queries.ndim != 3 or len(self.queries) != len(self.etas):
            raise ValueError("Expected queries [eta,class,hidden]")
        if not torch.isfinite(self.queries).all():
            raise ValueError("Nonfinite query features")
        if self.queries.shape[-1] % model.num_radzero_heads:
            raise ValueError("Query dimension is not divisible by the number of FLaS heads")

    @property
    def device(self):
        return self.queries.device

    def _prepare_projected(self, cls_raw, patches_raw):
        # Reproduce BaseEvaluator helper token selection exactly, including the
        # order of its use_all_image_tokens slice and re-splitting of CLS/patches.
        combined = [torch.cat((cls[:, None, :], patches), dim=1)
                    for cls, patches in zip(cls_raw, patches_raw)]
        if not self.use_all_image_tokens:
            combined = [layer[:, 1:] for layer in combined]
        if len(combined) != 1 or combined[0].shape[1] < 2:
            raise ValueError("Released FLaS requires one nonempty image feature layer")
        return {"image_cls_raw_features": [layer[:, 0, :] for layer in combined],
                "image_patch_raw_features": [layer[:, 1:, :] for layer in combined]}

    def prepare_vision_results(self, image_results):
        """Accept the unmodified BaseEvaluator.encode_image output for audits."""
        cls_raw = image_results.get("cls2_token_raw")
        if cls_raw is None:
            cls_raw = image_results["cls_token_raw"]
        to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return self._prepare_projected(
            [to_tensor(value) for value in cls_raw],
            [to_tensor(value) for value in image_results["patch_tokens_raw"]])

    @torch.no_grad()
    def encode_vision(self, pixels):
        """Author image encoder and frozen projection, without a CPU round-trip."""
        model = self.model
        model.eval()
        values = model.encode_image(pixels.to(self.device), model.use_last_n_layer_features)
        cls_tokens = values["cls_token"]
        cls, patches = cls_tokens[:, :1, :], values["patch_tokens"]
        second_cls = cls_tokens[:, 1:2, :] if cls_tokens.shape[1] == 2 else None
        if not model.projection:
            raise ValueError("Released-checkpoint compatibility requires projection heads")
        if len(model.use_last_n_layer_features) == 1:
            cls_raw, _ = model._project_and_normalize(
                cls, model.image_cls_projection, return_raw_and_normed=True)
            if second_cls is not None:
                cls_raw, _ = model._project_and_normalize(
                    second_cls, model.image_cls2_projection, return_raw_and_normed=True)
            patches_raw, _ = model._project_and_normalize(
                patches, model.image_patch_projection, return_raw_and_normed=True)
            cls_raw, patches_raw = [cls_raw], [patches_raw]
        else:
            if second_cls is not None:
                raise ValueError("Author evaluator does not support multilayer dual CLS")
            cls_raw, _ = model.custom_project_vision_cls_features(cls, values["hidden_states"])
            patches_raw, _ = model.custom_project_vision_patch_features(patches, values["hidden_states"])
        return self._prepare_projected(cls_raw, patches_raw)

    @torch.no_grad()
    def match_prepared(self, vision, query, return_maps=False, return_features=False,
                       return_token_features=False):
        """Call the original FLaS branch; do not reimplement its scoring formula."""
        self.model.eval()
        query = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        results = self.model.multihead_radzero_classification_branch(
            image_embeddings=vision,
            label_embeddings={"text_cls_raw_features": query},
            generate_grounding_map=return_maps)
        logits = results["logits_label_to_image"].squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite query-conditioned logits")
        output = {"logits": logits}
        if return_features:
            context_heads = results["radzero_enriched_image_context"]
            # Preserve the native [B, C, H, D_head] representation for
            # token/head-discriminative query adaptation.  ``class_local`` is
            # retained byte-for-byte for all existing methods.
            output["class_local_heads"] = F.normalize(context_heads, p=2, dim=-1)
            features = context_heads.flatten(start_dim=-2)
            features = F.normalize(features, p=2, dim=-1)
            if features.shape != (*logits.shape, query.shape[-1]) or not torch.isfinite(features).all():
                raise ValueError("Unexpected class-local feature shape or values")
            output["class_local"] = features
        if return_token_features:
            attention = results.get("radzero_attention_weights_h")
            image_values = results.get("radzero_image_kv_h")
            if attention is None or image_values is None:
                raise ValueError("Released FLaS branch did not return token features")
            output["attention_heads"] = attention
            output["image_values_heads"] = image_values
        if return_maps:
            raw_map = results["grounding_map"]
            spatial = raw_map[..., 1:] if self.model.radzero_head_enable_cross_attend_cls_token else raw_map
            side = math.isqrt(spatial.shape[-1])
            if side * side != spatial.shape[-1] or not torch.isfinite(raw_map).all():
                raise ValueError("Expected a finite square spatial similarity map")
            output["grounding_map"] = raw_map
            output["grounding_per_head"] = results["grounding_per_head"]
            output["similarity_scores"] = spatial.reshape(*spatial.shape[:2], side, side)
        return output

    @torch.no_grad()
    def match_prepared_readout_only(self, vision, aggregation_query, readout_query,
                                    return_maps=False, return_features=False,
                                    return_token_features=False):
        """Run QCA with separate aggregation and readout queries.

        This is the readout-only control used by the independent AlphaRAD
        protocol.  ``aggregation_query`` determines the softmax over image
        tokens; ``readout_query`` is used only for the final per-head cosine
        score.  The frozen model branch remains the single source of truth
        for both operations, so this cannot silently change the native
        scaling or head aggregation.
        """
        self.model.eval()
        aggregation_query = torch.as_tensor(
            aggregation_query, dtype=torch.float32, device=self.device)
        readout_query = torch.as_tensor(
            readout_query, dtype=torch.float32, device=self.device)
        results = self.model.multihead_radzero_classification_branch(
            image_embeddings=vision,
            label_embeddings={"text_cls_raw_features": readout_query},
            aggregation_label_embeddings={"text_cls_raw_features": aggregation_query},
            generate_grounding_map=return_maps)
        logits = results["logits_label_to_image"].squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite readout-only query-conditioned logits")
        output = {"logits": logits}
        if return_features:
            context_heads = results["radzero_enriched_image_context"]
            output["class_local_heads"] = F.normalize(context_heads, p=2, dim=-1)
            features = F.normalize(context_heads.flatten(start_dim=-2), p=2, dim=-1)
            if features.shape != (*logits.shape, readout_query.shape[-1]):
                raise ValueError("Unexpected readout-only class-local feature shape")
            output["class_local"] = features
        if return_token_features:
            output["attention_heads"] = results["radzero_attention_weights_h"]
            output["image_values_heads"] = results["radzero_image_kv_h"]
        if return_maps:
            raw_map = results["grounding_map"]
            spatial = raw_map[..., 1:] if self.model.radzero_head_enable_cross_attend_cls_token else raw_map
            side = math.isqrt(spatial.shape[-1])
            if side * side != spatial.shape[-1] or not torch.isfinite(raw_map).all():
                raise ValueError("Expected a finite square spatial similarity map")
            output["grounding_map"] = raw_map
            output["grounding_per_head"] = results["grounding_per_head"]
            output["similarity_scores"] = spatial.reshape(*spatial.shape[:2], side, side)
        return output

    @torch.no_grad()
    def forward(self, pixels, return_maps=False, return_features=False):
        vision = self.encode_vision(pixels)
        return {eta: self.match_prepared(vision, query, return_maps, return_features)
                for eta, query in zip(self.etas, self.queries)}


def _self_test():
    """CPU proof using the actual author's branch, without loading any encoder."""
    import ast
    import json

    source = Path(__file__).resolve().parents[1] / "cxrclip/model/cxrclip_plus_dqn.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "CXRClipPlusWithDQN")
    branch = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "multihead_radzero_classification_branch")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
                 str(source), "exec"), namespace)
    class SyntheticModel(nn.Module):
        multihead_radzero_classification_branch = namespace[branch.name]

    torch.set_num_threads(1)
    torch.manual_seed(42)
    checks = []
    for batch, classes in ((2, 3), (1, 1)):
        for attend_cls in (False, True):
            for aggregation in ("mean", "learned", "one_head"):
                model = SyntheticModel()
                model.radzero_logit_scale = nn.Parameter(torch.tensor(1.3))
                model.enable_multihead_radzero_forward_pass = True
                model.num_radzero_heads = 3
                model.use_learnable_head_weights = aggregation == "learned"
                model.radzero_head_weights = nn.Parameter(torch.tensor([-.5, .3, 1.]))
                model.head_index = 1 if aggregation == "one_head" else None
                model.radzero_head_enable_cross_attend_cls_token = attend_cls
                model.radzero_head_enable_bce_loss = True
                model.radzero_head_enable_softmax_loss = False
                query = torch.randn(classes, 12)
                matcher = QueryRefinementMatcher(model, query[None], (0.,))
                image = {"image_cls_raw_features": [torch.randn(batch, 12)],
                         "image_patch_raw_features": [torch.randn(batch, 9, 12)]}
                expected = model.multihead_radzero_classification_branch(
                    image, {"text_cls_raw_features": query}, generate_grounding_map=True)
                dual_same = model.multihead_radzero_classification_branch(
                    image, {"text_cls_raw_features": query},
                    aggregation_label_embeddings={"text_cls_raw_features": query},
                    generate_grounding_map=True)
                assert torch.equal(dual_same["logits_label_to_image"],
                                   expected["logits_label_to_image"])
                assert torch.equal(dual_same["grounding_map"], expected["grounding_map"])
                actual = matcher.match_prepared(image, matcher.queries[0], True, True)
                assert torch.equal(actual["logits"], expected["logits_label_to_image"].squeeze(-1))
                assert torch.equal(actual["grounding_map"], expected["grounding_map"])
                assert torch.equal(actual["grounding_per_head"], expected["grounding_per_head"])
                assert actual["similarity_scores"].shape == (batch, classes, 3, 3)
                torch.testing.assert_close(actual["class_local"].norm(dim=-1),
                                           torch.ones(batch, classes))
                assert not any(parameter.requires_grad for parameter in model.parameters())
                checks.append([batch, classes, attend_cls, aggregation])
    support = F.normalize(torch.randn(6, 3, 12), dim=-1)
    labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0],
                           [0, 0, 1], [1, 0, 0], [0, 1, 1]])
    original = torch.randn(3, 12)
    fitted = fit_query_directions(support, labels, original)
    np.testing.assert_array_equal(fitted["queries"][0], original.numpy())
    np.testing.assert_allclose(np.linalg.norm(fitted["queries"][1], axis=-1), 1, atol=1e-6)
    uncertain = fit_uncertainty_query_directions(support, labels, original)
    np.testing.assert_array_equal(uncertain["queries"][0], original.numpy())
    np.testing.assert_allclose(np.linalg.norm(uncertain["queries"][1], axis=-1), 1, atol=1e-6)
    assert np.isfinite(uncertain["support_reliability"]).all()
    assert ((uncertain["support_reliability"] >= 0) &
            (uncertain["support_reliability"] <= 1)).all()
    head_support = F.normalize(torch.randn(6, 3, 3, 4), dim=-1)
    head_original = torch.randn(3, 12)
    tdqa = fit_token_discriminative_query_directions(
        head_support, labels, head_original, num_heads=3)
    np.testing.assert_array_equal(tdqa["queries"][0], head_original.numpy())
    np.testing.assert_allclose(
        np.linalg.norm(tdqa["queries"][1].reshape(3, 3, 4), axis=-1),
        1.0, atol=1e-5)
    np.testing.assert_allclose(tdqa["head_gates"].mean(-1), 1.0, atol=1e-6)
    fixed_model = SyntheticModel()
    fixed_model.radzero_logit_scale = nn.Parameter(torch.tensor(1.3))
    fixed_model.enable_multihead_radzero_forward_pass = True
    fixed_model.num_radzero_heads = 3
    fixed_model.use_learnable_head_weights = False
    fixed_model.radzero_head_weights = nn.Parameter(torch.ones(3))
    fixed_model.head_index = None
    fixed_model.radzero_head_enable_cross_attend_cls_token = False
    fixed_model.radzero_head_enable_bce_loss = True
    fixed_model.radzero_head_enable_softmax_loss = False
    fixed_matcher = QueryRefinementMatcher(fixed_model, original[None], (0.,))
    support_vision = [{"image_cls_raw_features": [torch.randn(6, 12)],
                       "image_patch_raw_features": [torch.randn(6, 9, 12)]}]
    reaggregated = fit_reaggregation_query_directions(
        support_vision, labels, original, fixed_matcher, eta=.3, max_iter=2)
    assert reaggregated["queries"].shape == (2, 3, 12)
    assert reaggregated["reaggregation_iterations"] == 2
    np.testing.assert_allclose(np.linalg.norm(reaggregated["queries"][1], axis=-1), 1, atol=1e-6)
    assert reaggregated["query_trajectory"].shape == (3, 3, 12)
    assert len(reaggregated["reaggregation_history"]) == 2
    assert all("support_utility" in step and "local_lipschitz_mean" in step
               for step in reaggregated["reaggregation_history"])
    gated = fit_support_only_gate(
        support, support_vision, labels, original, fixed_matcher,
        eta=.3, gamma_margin=1e9, gamma_kappa=1.0, max_iter=2)
    assert gated["gate_selected_full"] is False
    assert gated["gate_test_labels_used"] is False
    assert gated["gate_probe_native_passes"] == 3
    selected = fit_support_only_gate(
        support, support_vision, labels, original, fixed_matcher,
        eta=.3, gamma_margin=-1e9, gamma_kappa=1e9, max_iter=2)
    assert selected["gate_selected_full"] is True
    assert selected["gate_total_native_passes"] >= 3
    print(json.dumps({"status": "passed", "device": "cpu", "branch_cases": len(checks),
                      "zero_eta_logits_bitwise_equal": True,
                      "zero_eta_maps_bitwise_equal": True,
                      "zero_eta_query_bitwise_equal": True,
                      "singleton_batch_and_class": True,
                      "model_parameters_frozen": True,
                      "reaggregation_fixed_point_cases": 1}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        _self_test()
    else:
        parser.error("Use --self-test for a CPU check; import this module for inference")
