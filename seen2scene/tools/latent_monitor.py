"""Semantic class distribution monitoring for tracking voxel class percentages."""

import torch
import numpy as np
import plotly.graph_objects as go
from typing import List
from collections import defaultdict


class SemanticDistributionTracker:
    """Tracks accumulated semantic class distribution over training."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.class_counts = defaultdict(int)

    def update(
        self,
        ins_mask: torch.Tensor,
        object_names: List[List[str]],
        batch_indices: torch.Tensor,
    ):
        """Update accumulated class counts from current batch.

        Args:
            ins_mask: Instance mask [N, M] where N is voxels, M is objects.
                      Each row is a binary vector indicating object membership.
                      All-zero rows indicate empty space.
            object_names: List of object name lists per batch item
            batch_indices: Batch index for each voxel [N]
        """
        ins_mask_np = ins_mask.cpu().numpy()
        batch_indices_np = batch_indices.cpu().numpy()

        # Compute column ranges for each batch
        # ins_mask columns are ordered: [batch0_obj0, ..., batch0_objM0, batch1_obj0, ...]
        col_offset = 0
        col_ranges = []
        for names in object_names:
            n_objs = len(names)
            col_ranges.append((col_offset, col_offset + n_objs))
            col_offset += n_objs

        # Process each batch separately
        for batch_idx in range(len(object_names)):
            batch_mask = batch_indices_np == batch_idx
            batch_ins_mask = ins_mask_np[batch_mask]  # [N_batch, M_total]

            # Extract only columns for this batch
            col_start, col_end = col_ranges[batch_idx]
            batch_ins_mask_local = batch_ins_mask[:, col_start:col_end]  # [N_batch, M_batch]

            # Check for empty voxels (all-zero rows in local columns)
            row_sums = batch_ins_mask_local.sum(axis=1)  # [N_batch]
            empty_mask = row_sums == 0
            n_empty = empty_mask.sum()

            if n_empty > 0:
                self.class_counts["empty"] += int(n_empty)

            # Process non-empty voxels
            non_empty_mask = ~empty_mask
            if non_empty_mask.any():
                non_empty_ins = batch_ins_mask_local[non_empty_mask]  # [N_non_empty, M_batch]

                # Get LOCAL object indices for each voxel (within this batch's objects)
                obj_indices_local = non_empty_ins.argmax(axis=1)  # [N_non_empty], range [0, M_batch)

                # Count voxels per object
                unique_objs, counts = np.unique(obj_indices_local, return_counts=True)

                for obj_idx_local, count in zip(unique_objs, counts):
                    obj_idx_local = int(obj_idx_local)
                    if 0 <= obj_idx_local < len(object_names[batch_idx]):
                        class_name = object_names[batch_idx][obj_idx_local]
                        self.class_counts[class_name] += int(count)

    def create_plot(self, step: int) -> go.Figure:
        """Create Plotly bar chart of accumulated class distribution."""
        if not self.class_counts:
            fig = go.Figure()
            fig.update_layout(title="No data yet")
            return fig

        # Compute percentages
        total = sum(self.class_counts.values())
        percentages = {
            cls: (count / total) * 100 for cls, count in self.class_counts.items()
        }

        # Sort and take top-k
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        top_classes = sorted_items[: self.top_k]

        # Group remaining as "Other"
        if len(sorted_items) > self.top_k:
            other_pct = sum(pct for _, pct in sorted_items[self.top_k :])
            top_classes.append(("Other", other_pct))

        classes, values = zip(*top_classes) if top_classes else ([], [])

        # Create bar chart
        fig = go.Figure(
            go.Bar(
                x=classes,
                y=values,
                marker=dict(color=values, colorscale="Viridis"),
                hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Semantic Class Distribution (Step {step})",
            xaxis_title="Class",
            yaxis_title="Percentage (%)",
            template="plotly_white",
            width=1000,
            height=600,
            showlegend=False,
            xaxis=dict(tickangle=-45),
        )

        return fig
