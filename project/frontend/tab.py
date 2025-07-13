import flet as ft

class Tab(ft.Tab):
    def __init__(self, text, icon, controls, horizontal_alignment=None):
        super().__init__(
            text=text,
            icon=icon,
            content=ft.Container(
                ft.Column(controls=controls, expand=True, scroll=ft.ScrollMode.AUTO, horizontal_alignment=horizontal_alignment),
                padding=20,
                alignment=ft.alignment.top_center,
                expand=True)
        )

class TabTitle(ft.Container):
    def __init__(self, title, button=None):
        row = [ft.Text(title, size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800)]
        if button:
            row.append(button)
        super().__init__(
            content=ft.Row(row, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            margin=ft.margin.only(bottom=20)
        ),