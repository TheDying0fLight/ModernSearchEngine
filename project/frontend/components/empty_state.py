import flet as ft

class EmptyState(ft.Container):
    def __init__(self, icon, title, text, button_icon, button_text, on_button_click):
        super().__init__(
            content=ft.Column([
                ft.Icon(icon, size=64, color=ft.Colors.GREY_400),
                ft.Text(
                    title,
                    size=18,
                    color=ft.Colors.GREY_600,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Text(
                    text,
                    size=14,
                    color=ft.Colors.GREY_500,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.ElevatedButton(
                    text=button_text,
                    icon=button_icon,
                    on_click=on_button_click,
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.BLUE_600,
                        color=ft.Colors.WHITE
                    )
                )
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15),
            padding=50,
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=4,
                color=ft.Colors.GREY_200,
                offset=ft.Offset(0, 2)
            ),
            expand=True
        )