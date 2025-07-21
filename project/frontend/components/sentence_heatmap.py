import flet as ft
import numpy as np
from typing import List


class HorizontalSentenceHeatmap(ft.Container):
    """
    Horizontal heatmap for sentence importance under the snippet
    """

    def __init__(self, sentence_scores: list[float], width: int = 260, height: int = 20):
        """
        Args:
            sentence_scores: List of importance values between 0.0 and 1.0
            width: Width of the heatmap (should match ResultCard width minus padding)
            height: Height of the heatmap
        """
        sentence_scores = np.array(sentence_scores)
        n_bins = 26
        bin_max = self.resize_with_max(sentence_scores, n_bins)

        # Calculate segment width based on available space
        segment_width = max(1, width // n_bins)

        # Create horizontal segments
        segments = []
        for i, score in enumerate(bin_max):
            # Color mapping consistent with vertical heatmap
            if score < 0.7:
                color = ft.Colors.GREY_200
            elif score < 0.8:
                color = ft.Colors.ORANGE_400
            elif score < 0.9:
                color = ft.Colors.DEEP_ORANGE_500
            else:
                color = ft.Colors.RED_500

            segments.append(
                ft.Container(
                    width=segment_width,
                    height=height,
                    bgcolor=color,
                    border_radius=ft.border_radius.only(
                        top_left=2 if i == 0 else 0,
                        bottom_left=2 if i == 0 else 0,
                        top_right=2 if i == len(sentence_scores) - 1 else 0,
                        bottom_right=2 if i == len(sentence_scores) - 1 else 0
                    ),
                    tooltip=f"Segment {i+1}: {score:.2f}"
                )
            )

        super().__init__(
            content=ft.Row(
                controls=segments,
                spacing=0.5,
                tight=True
            ),
            width=min(width, len(sentence_scores) * segment_width + len(sentence_scores)),
            height=height,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=4,
            tooltip=f"Sentence Importance Heatmap ({len(sentence_scores)} sentences)",
            padding=ft.padding.all(2)
        )

    def resize_with_max(self, data, target_length):
        input_length = len(data)

        if input_length == target_length:
            return data

        elif input_length > target_length:
            # Compression: take max in each segment
            result = []
            bin_edges = np.linspace(0, input_length, target_length + 1, dtype=int)
            for i in range(target_length):
                segment = data[bin_edges[i]:bin_edges[i+1]]
                if len(segment) > 0:
                    result.append(np.max(segment))
                else:
                    result.append(0)
            return np.array(result)

        else:
            x_old = np.linspace(0, 1, input_length)
            x_new = np.linspace(0, 1, target_length)
            return np.interp(x_new, x_old, data)
