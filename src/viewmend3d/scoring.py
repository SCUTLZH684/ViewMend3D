"""Torch-only scoring kernels, independent of ActiveGS and its CUDA renderer.

The defect is a current-map consistency proxy, not a ground-truth error.
These functions accept CPU or CUDA tensors and never access a simulator.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryThresholds:
    opacity: float = 0.7
    normal_norm: float = 0.5
    residual_deadzone: float = 0.05
    depth_jump_absolute: float = 0.05
    depth_jump_relative: float = 0.05
    min_valid_fraction: float = 0.01
    minimum_peak: float = 1e-4

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.opacity > 1 or self.min_valid_fraction > 1:
            raise ValueError("opacity and min_valid_fraction must be at most one")
        if self.residual_deadzone >= 1:
            raise ValueError("residual_deadzone must be smaller than one")


def _image_batch(value, batch, height, width, name):
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if tuple(value.shape) != (batch, height, width):
        raise ValueError(f"{name} must have shape [H,W], [N,H,W], or [N,1,H,W]")
    return value


def nonnegative_finite(value):
    """Drop invalid/negative utilities without turning infinity into a reward."""
    return torch.where(torch.isfinite(value) & (value >= 0), value,
                       torch.zeros_like(value))


def normalize_sum(value):
    value = nonnegative_finite(value)
    total = value.sum()
    return value / total.clamp_min(torch.finfo(value.dtype).tiny)


def combine_utilities(base, defect, weight=0.5, minimum_peak=1e-4):
    """N(base) + weight*N(defect), with a group-level weak-signal fallback."""
    if base.ndim != 1 or defect.shape != base.shape or base.numel() == 0:
        raise ValueError("base and defect must be equally sized nonempty vectors")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if not math.isfinite(minimum_peak) or minimum_peak < 0:
        raise ValueError("minimum_peak must be finite and nonnegative")
    clean_defect = nonnegative_finite(defect)
    active = (clean_defect.max() >= minimum_peak) & (clean_defect.sum() > 0)
    geometry_component = normalize_sum(clean_defect) * active.to(base.dtype)
    return normalize_sum(base) + weight * geometry_component


def geometry_defect(depth, normal, depth_normal, opacity, confidence,
                    depth_range, thresholds=None, use_depth_gate=True):
    """Compute a supported, depth-continuous directed normal inconsistency.

    Normals have shape [3,H,W] or [N,3,H,W]. Other images are single-channel
    and match that batch. Results use a scalar/[H,W] for a single view and
    [N]/[N,H,W] for a batch. The image border is always invalid, including
    in the no-depth-gate ablation.
    """
    thresholds = thresholds or GeometryThresholds()
    single = normal.ndim == 3
    if single:
        normal = normal.unsqueeze(0)
        depth_normal = depth_normal.unsqueeze(0)
    if normal.ndim != 4 or normal.shape[1] != 3 or depth_normal.shape != normal.shape:
        raise ValueError("normal and depth_normal must have matching [N,3,H,W] shapes")
    if not normal.is_floating_point():
        raise ValueError("normal tensors must be floating point")
    batch, _, height, width = normal.shape
    if batch < 1 or height < 1 or width < 1:
        raise ValueError("geometry images cannot be empty")
    values = {}
    for name, value in (("depth", depth), ("opacity", opacity), ("confidence", confidence)):
        values[name] = _image_batch(value, batch, height, width, name).to(normal)
    depth, opacity, confidence = (values[n] for n in ("depth", "opacity", "confidence"))
    depth_normal = depth_normal.to(normal)
    # Camera bounds are metadata. Validate them on the host, so a planner
    # that passes its shared CPU bounds does not synchronize CUDA twice for
    # every candidate. Casting first preserves the image dtype's bounds.
    bounds = torch.as_tensor(depth_range, device="cpu", dtype=depth.dtype).reshape(-1)
    if bounds.numel() != 2:
        raise ValueError("depth_range must contain finite near < far")
    near, far = bounds.tolist()
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("depth_range must contain finite near < far")

    finite_normals = torch.isfinite(normal).all(dim=1) & torch.isfinite(depth_normal).all(dim=1)
    clean_normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    clean_depth_normal = torch.nan_to_num(depth_normal, nan=0.0, posinf=0.0, neginf=0.0)
    normal_length = clean_normal.norm(dim=1)
    depth_normal_length = clean_depth_normal.norm(dim=1)
    valid = (finite_normals & torch.isfinite(depth) & torch.isfinite(opacity)
             & torch.isfinite(confidence) & (depth >= near) & (depth <= far)
             & (depth > 0) & (opacity >= thresholds.opacity)
             & (normal_length > thresholds.normal_norm)
             & (depth_normal_length > thresholds.normal_norm))
    # Zero padding rejects the image boundary as well as invalid neighbors.
    neighbor_count = F.conv2d(valid.to(depth.dtype).unsqueeze(1),
                              torch.ones(1, 1, 3, 3, device=depth.device,
                                         dtype=depth.dtype), padding=1)[:, 0]
    stable = neighbor_count >= 9
    if use_depth_gate:
        clean_depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)
        largest = F.max_pool2d(clean_depth, 3, stride=1, padding=1)[:, 0]
        smallest = -F.max_pool2d(-clean_depth, 3, stride=1, padding=1)[:, 0]
        stable &= ((largest - smallest) <= thresholds.depth_jump_absolute
                   + thresholds.depth_jump_relative * depth)

    unit_normal = clean_normal / normal_length.unsqueeze(1).clamp_min(1e-8)
    unit_depth_normal = clean_depth_normal / depth_normal_length.unsqueeze(1).clamp_min(1e-8)
    cosine = (unit_normal * unit_depth_normal).sum(dim=1).clamp(-1, 1)
    residual = (1 - cosine) * 0.5
    positive_residual = (residual - thresholds.residual_deadzone).clamp_min(0) / (1 - thresholds.residual_deadzone)
    uncertainty = (1 - torch.nan_to_num(confidence, nan=1.0,
                                      posinf=1.0, neginf=1.0)).clamp(0, 1)
    valid_fraction = stable.to(depth.dtype).mean(dim=(-2, -1))
    area_pass = valid_fraction >= thresholds.min_valid_fraction
    stable &= area_pass[:, None, None]
    heatmap = stable.to(depth.dtype) * uncertainty * positive_residual
    result = {
        "score": heatmap.mean(dim=(-2, -1)),
        "valid_fraction": valid_fraction,
        "positive_fraction": (heatmap > 0).to(depth.dtype).mean(dim=(-2, -1)),
        "mask": stable,
        "residual": residual,
        "heatmap": heatmap,
    }
    return {name: value[0] for name, value in result.items()} if single else result


def view_scores(utilities, path_lengths, path_length_factor=0.5):
    """ActiveGS sum-normalized path penalty, with unreachable views excluded.

    All-zero utilities use the torch planning RNG only on reachable views.
    No reachable view is an explicit failure rather than a debugger prompt.
    """
    if utilities.ndim != 1 or utilities.numel() == 0:
        raise ValueError("candidate utilities must be a nonempty vector")
    if not math.isfinite(path_length_factor) or path_length_factor < 0:
        raise ValueError("path_length_factor must be finite and nonnegative")
    lengths = torch.as_tensor(path_lengths, dtype=utilities.dtype, device=utilities.device)
    if lengths.shape != utilities.shape:
        raise ValueError("path lengths and utilities must have matching shapes")
    reachable = torch.isfinite(lengths) & (lengths >= 0)
    if not bool(reachable.any()):
        raise RuntimeError("no reachable view candidates")
    clean = nonnegative_finite(utilities)
    clean = torch.where(reachable, clean, torch.zeros_like(clean))
    length_sum = lengths[reachable].sum()
    costs = torch.zeros_like(lengths)
    costs[reachable] = lengths[reachable] / length_sum.clamp_min(torch.finfo(lengths.dtype).tiny)
    if bool(clean.sum() > 0):
        scores = normalize_sum(clean) - path_length_factor * costs
    else:
        scores = torch.zeros_like(clean)
        scores[reachable] = torch.rand(int(reachable.sum()), device=clean.device,
                                       dtype=clean.dtype)
    return torch.where(reachable, scores, torch.full_like(scores, -torch.inf))


def guarded_view_scores(base, defect, path_lengths, path_length_factor=0.5,
                        beta=0.1, minimum_peak=1e-4, baseline_utilities=None):
    """Add a bounded geometry bonus to one unchanged baseline score draw.

    The bonus is at most ``beta / n_reachable`` and is peak-normalized only
    on reachable candidates. There is no second score normalization. The
    bound limits regret in the baseline proxy score, not reconstruction
    error. Weak, constant, disabled, and all-zero-base cases return the very
    same baseline tensor, including its single planning RNG draw.

    An adapter may supply its existing pre-normalized baseline utilities to
    preserve that adapter's floating-point evaluation order exactly. Raw
    ``base`` remains the basis of zero-signal checks and utility diagnostics.
    """
    if not isinstance(base, torch.Tensor) or not base.is_floating_point():
        raise ValueError("base utilities must be a floating-point tensor")
    if base.ndim != 1 or base.numel() == 0 or defect.shape != base.shape:
        raise ValueError("base and defect must be equally sized nonempty vectors")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not math.isfinite(minimum_peak) or minimum_peak < 0:
        raise ValueError("minimum_peak must be finite and nonnegative")
    lengths = torch.as_tensor(path_lengths, dtype=base.dtype, device=base.device)
    if baseline_utilities is not None and baseline_utilities.shape != base.shape:
        raise ValueError("baseline utilities must match base utilities")
    # This is deliberately the only view_scores call: all-zero utility
    # fallback must consume exactly the baseline's RNG draw once.
    baseline = view_scores(base if baseline_utilities is None else baseline_utilities,
                           lengths, path_length_factor)
    reachable = torch.isfinite(lengths) & (lengths >= 0)
    n_reachable = int(reachable.sum())
    clean_base = torch.where(reachable, nonnegative_finite(base), torch.zeros_like(base))
    clean_defect = nonnegative_finite(defect.to(base))
    clean_defect = torch.where(reachable, clean_defect, torch.zeros_like(base))
    peak = clean_defect[reachable].max()
    geometry_range = peak - clean_defect[reachable].min()
    signal_active = bool((peak >= minimum_peak) & (peak > 0))
    discriminative = bool(geometry_range > 0)
    cap = beta / n_reachable
    bonus = torch.zeros_like(base)
    peak_normalized = torch.zeros_like(base)
    if beta == 0:
        fallback = "weight_zero"
    elif not bool(clean_base.sum() > 0):
        fallback = "base_all_zero"
    elif n_reachable == 1:
        fallback = "single_reachable"
    elif not signal_active:
        fallback = "weak_geometry"
    elif not discriminative:
        fallback = "constant_geometry"
    else:
        fallback = "none"
        peak_normalized = (clean_defect / peak).clamp(0, 1)
        bonus = cap * peak_normalized
    applied = fallback == "none"
    scores = baseline + bonus if applied else baseline
    baseline_index = int(torch.argmax(baseline))
    selected = int(torch.argmax(scores))
    regret = baseline[baseline_index] - baseline[selected]
    # Permit ordinary addition roundoff without broadening the mathematical
    # bonus rule or excluding candidates via an extra eligibility guard.
    tolerance = 4 * torch.finfo(base.dtype).eps * (
        float(baseline[reachable].abs().max()) + cap)
    if float(regret) > cap + tolerance:
        raise RuntimeError("bounded geometry score exceeds baseline regret cap")
    costs = torch.zeros_like(lengths)
    costs[reachable] = lengths[reachable] / lengths[reachable].sum().clamp_min(
        torch.finfo(lengths.dtype).tiny)
    normalized_base = normalize_sum(clean_base)
    diagnostics = {
        "scoring_version": "bounded_geometry_v2",
        "geometry_beta": beta,
        "geometry_bonus_cap": cap,
        "n_reachable": n_reachable,
        "baseline_scores": baseline,
        "baseline_selected_index": baseline_index,
        "selected_index": selected,
        "selected_baseline_score": baseline[selected],
        "baseline_regret": regret,
        "regret_tolerance": tolerance,
        "regret_bound_satisfied": True,
        "geometry_peak": peak,
        "geometry_range": geometry_range,
        "defect_peak_normalized": peak_normalized,
        "geometry_bonus": bonus,
        "geometry_signal_active": signal_active,
        "geometry_discriminative": discriminative,
        "geometry_active": applied,
        "geometry_fallback_reason": fallback,
        "selection_changed": selected != baseline_index,
        "base_delta_vs_baseline": clean_base[selected] - clean_base[baseline_index],
        "path_delta_vs_baseline": lengths[selected] - lengths[baseline_index],
        "base_normalized_delta_vs_baseline": normalized_base[selected] - normalized_base[baseline_index],
        "path_cost_delta_vs_baseline": costs[selected] - costs[baseline_index],
        "normalization_domain": "reachable_candidates",
        "geometry_normalization": "peak_reachable_candidates",
        "tie_policy": "first_argmax",
    }
    return scores, diagnostics
