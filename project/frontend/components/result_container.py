import flet as ft
from .sentence_heatmap import HorizontalSentenceHeatmap
from typing import List, Optional

class ResultTitle(ft.Text):
    def __init__(self, text):
        super().__init__(
            text,
            size=18,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_700,
            expand=True
        )

class ResultContainer(ft.Container):
    """Individual result card component"""

    def __init__(self, title: str, text: str, source: str, metadata: list[str], button: ft.Button, on_click, width=None, max_hight=None, sentence_scores: list[float] = None):
        metadata_control = ft.Column([ft.Text(source, color=ft.Colors.GREEN_600, size=12, weight=ft.FontWeight.BOLD)], spacing=5)
        for md in metadata:
            metadata_control.controls.append(ft.Text(md, color=ft.Colors.GREY_600, size=12))

        # Content controls building
        content_controls = [
            # Title and favorite button row
            ft.Row([
                ResultTitle(title),
                #ft.Column([], expand=True),
                button
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.START),

            # Description/snippet
            ft.Text(text, size=14, color=ft.Colors.GREY_700, max_lines=3),

            ft.Row([], expand=True),  # Spacer before heatmap
            # Add heatmap
            HorizontalSentenceHeatmap(
                sentence_scores=sentence_scores,
                width=(width - 40) if width else 260,
                height=20
            ) if sentence_scores else ft.Row([]),
            # Metadata row
            metadata_control
        ]

        super().__init__(
            content=ft.Column(content_controls, spacing=8),
            padding=20,
            margin=ft.margin.only(bottom=15),
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.GREY_200),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=4,
                color=ft.Colors.GREY_200,
                offset=ft.Offset(0, 2)
            ),
            width=width,
            height=max_hight,
            expand=True,
            on_click=on_click
        )