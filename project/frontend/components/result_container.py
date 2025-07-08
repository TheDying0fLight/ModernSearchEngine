import flet as ft

class ResultContainer(ft.Container):
    """Individual result card component"""

    def __init__(self, title: str, text: str, source: str, metadata: list[str], button: ft.Button, on_click):
        metadata_row = ft.Row([ft.Text(source, color=ft.Colors.GREEN_600, size=12, weight=ft.FontWeight.BOLD)], spacing=5)
        for md in metadata:
            metadata_row.controls.append(ft.Text("•", color=ft.Colors.GREY_400, size=12))
            metadata_row.controls.append(ft.Text(md, color=ft.Colors.GREY_600, size=12))
        super().__init__(
            content=ft.Column([
                # Title and favorite button row
                ft.Row([
                    ft.Text(
                        title,
                        size=18,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_700,
                        expand=True
                    ),
                    button
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                # Description/snippet
                ft.Text(text, size=14, color=ft.Colors.GREY_700, max_lines=3),

                # Metadata row
                metadata_row
            ], spacing=8),
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
            on_click=on_click
        )