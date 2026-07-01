"""Gradient monitoring utilities for training diagnostics."""

import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import numpy as np


def create_gradient_distribution_plot(
    model: nn.Module,
    gradient_clip_val: Optional[float] = None,
    step: Optional[int] = None,
    **kwargs,
) -> go.Figure:
    """Create compact Plotly visualization of gradient distributions.

    Samples a subset of parameters for efficient analysis.
    Computes histograms directly on GPU for maximum performance.

    Args:
        model: Model with gradients
        gradient_clip_val: Gradient clipping threshold
        step: Training step number

    Returns:
        Plotly figure
    """
    # Collect all parameters with gradients
    params_with_grad = [
        p for p in model.parameters()
        if p.grad is not None and p.grad.numel() > 0
    ]

    if not params_with_grad:
        fig = go.Figure()
        fig.update_layout(title="No gradients available")
        return fig

    # Sample subset of parameters (5% or max 50 parameters for speed)
    total_params = len(params_with_grad)
    max_params = min(50, max(int(total_params * 0.05), 5))

    if total_params > max_params:
        # Randomly sample parameters
        indices = torch.randperm(total_params)[:max_params].tolist()
        sampled_params = [params_with_grad[i] for i in indices]
    else:
        sampled_params = params_with_grad

    # Collect gradients on GPU from sampled parameters
    # Also limit total elements to max 10k for fast histogram computation
    grad_norms_list = []
    grad_values_list = []
    param_values_list = []
    max_total_elements = 10_000

    # First pass: collect norms and estimate total size
    total_size = sum(p.numel() for p in sampled_params)
    sample_ratio = min(1.0, max_total_elements / max(total_size, 1))

    for param in sampled_params:
        grad = param.grad.detach()
        grad_norms_list.append(grad.norm(2))

        # Sample values if total would exceed max_total_elements
        if sample_ratio < 1.0:
            n_samples = max(1, int(param.numel() * sample_ratio))
            indices = torch.randint(0, param.numel(), (n_samples,), device=param.device)
            grad_values_list.append(grad.flatten()[indices])
            param_values_list.append(param.detach().flatten()[indices])
        else:
            grad_values_list.append(grad.flatten())
            param_values_list.append(param.detach().flatten())

    # Concatenate on GPU
    grad_norms_tensor = torch.stack(grad_norms_list)
    grad_values_tensor = torch.cat(grad_values_list)
    param_values_tensor = torch.cat(param_values_list)
    total_elements = grad_values_tensor.numel()

    sample_info = f" (sampled {len(sampled_params)}/{total_params} params"
    if sample_ratio < 1.0:
        sample_info += f", {total_elements:,}/{total_size:,} values"
    sample_info += ")"
    sample_info = sample_info if len(sampled_params) < total_params or sample_ratio < 1.0 else ""

    # Move to CPU for histogram (torch.histogram not available on CUDA)
    # Since we sampled only 10% of params, CPU transfer is fast
    grad_norms_cpu = grad_norms_tensor.cpu()
    grad_values_cpu = grad_values_tensor.cpu()
    param_values_cpu = param_values_tensor.cpu()

    # Compute histograms on CPU
    bins = 50
    grad_norm_hist_tensor, grad_norm_bins_tensor = torch.histogram(
        grad_norms_cpu.float(), bins=bins
    )
    grad_val_hist_tensor, grad_val_bins_tensor = torch.histogram(
        grad_values_cpu.float(), bins=bins
    )
    param_hist_tensor, param_bins_tensor = torch.histogram(
        param_values_cpu.float(), bins=bins
    )

    # Convert to numpy
    grad_norm_hist = grad_norm_hist_tensor.numpy()
    grad_norm_bins = grad_norm_bins_tensor.numpy()
    grad_val_hist = grad_val_hist_tensor.numpy()
    grad_val_bins = grad_val_bins_tensor.numpy()
    param_hist = param_hist_tensor.numpy()
    param_bins = param_bins_tensor.numpy()

    # Keep grad_norms as tensor for later statistics
    grad_norms = grad_norms_tensor

    # Compute bin centers
    grad_norm_centers = (grad_norm_bins[:-1] + grad_norm_bins[1:]) / 2
    grad_val_centers = (grad_val_bins[:-1] + grad_val_bins[1:]) / 2
    param_centers = (param_bins[:-1] + param_bins[1:]) / 2

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Gradient Norm Distribution",
            "Gradient Value Distribution",
            "Parameter Distribution",
            "Gradient Norm CDF",
        ),
    )

    # 1. Gradient norm histogram (use Bar instead of Histogram for smaller size)
    fig.add_trace(
        go.Bar(
            x=grad_norm_centers,
            y=grad_norm_hist,
            name="Grad Norm",
            marker_color="blue",
        ),
        row=1,
        col=1,
    )

    # Add clipping threshold line
    if gradient_clip_val:
        fig.add_vline(
            x=gradient_clip_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Clip: {gradient_clip_val}",
            row=1,
            col=1,
        )
        clipped_pct = ((grad_norms > gradient_clip_val).sum().item() / len(grad_norms) * 100)
        status = "✓" if clipped_pct < 10 else "⚠"
        fig.add_annotation(
            text=f"{status} {clipped_pct:.1f}% clipped",
            x=0.95,
            y=0.95,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            row=1,
            col=1,
        )

    # 2. Gradient value histogram
    fig.add_trace(
        go.Bar(
            x=grad_val_centers,
            y=grad_val_hist,
            name="Grad Values",
            marker_color="green",
        ),
        row=1,
        col=2,
    )

    # 3. Parameter histogram
    fig.add_trace(
        go.Bar(
            x=param_centers,
            y=param_hist,
            name="Params",
            marker_color="orange",
        ),
        row=2,
        col=1,
    )

    # 4. Gradient norm CDF
    sorted_norms, _ = torch.sort(grad_norms)
    sorted_norms_np = sorted_norms.cpu().numpy()
    cdf = np.arange(1, len(sorted_norms_np) + 1) / len(sorted_norms_np) * 100

    fig.add_trace(
        go.Scatter(
            x=sorted_norms_np,
            y=cdf,
            mode="lines",
            name="CDF",
            line=dict(color="purple"),
        ),
        row=2,
        col=2,
    )

    # Add P95 marker
    p95_idx = int(len(sorted_norms_np) * 0.95)
    p95_value = sorted_norms_np[p95_idx]
    fig.add_trace(
        go.Scatter(
            x=[p95_value],
            y=[95],
            mode="markers",
            marker=dict(size=10, color="red"),
            name=f"P95: {p95_value:.3f}",
        ),
        row=2,
        col=2,
    )

    # Update layout
    title = "Gradient Distribution"
    if step is not None:
        title += f" (Step {step})"
    title += sample_info

    fig.update_layout(
        title=title,
        showlegend=True,
        height=800,
        width=1600,
        template="plotly_white",
    )

    # Add statistics annotation
    stats_text = (
        f"Grad Norm:<br>"
        f"  Mean: {grad_norms.mean().item():.3f}<br>"
        f"  Std: {grad_norms.std().item():.3f}<br>"
        f"  P95: {p95_value:.3f}<br>"
        f"  Max: {grad_norms.max().item():.3f}<br>"
        f"<br>Recommended clip: {p95_value * 2:.3f}<br>"
        f"<br>Elements: {total_elements:,}{sample_info}"
    )

    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.01,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10, family="monospace"),
    )

    return fig
