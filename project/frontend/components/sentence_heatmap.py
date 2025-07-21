import flet as ft
from typing import List


class HorizontalSentenceHeatmap(ft.Container):
    """
    Horizontal heatmap for sentence importance under the snippet
    """

    def __init__(self, sentence_scores: List[float], width: int = 280, height: int = 20):
        """
        Args:
            sentence_scores: List of importance values between 0.0 and 1.0
            width: Width of the heatmap (should match ResultCard width minus padding)
            height: Height of the heatmap
        """
        self.sentence_scores = sentence_scores

        # Calculate segment width based on available space
        segment_width = max(3, width // len(sentence_scores))

        # Create horizontal segments
        segments = []
        for i, score in enumerate(sentence_scores):
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
                    tooltip=f"Sentence {i+1}: {score:.2f}"
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
