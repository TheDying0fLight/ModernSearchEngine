import flet as ft

colors = [
        ft.Colors.RED,
        ft.Colors.BLUE,
        ft.Colors.YELLOW,
        ft.Colors.PURPLE,
        ft.Colors.LIME,
    ]
def get_options():
        options = []
        for color in colors:
            options.append(
                ft.DropdownOption(
                    key=color.value,
                    content=ft.Text(
                        value=color.value,
                        color=color,
                    ),
                )
            )
        return options

def main(page: ft.Page):

    dd = ft.Dropdown(
        editable=True,
        label="Color",
        options=get_options(),
    )
    page.add(ft.Row([
            ft.Container(width=50, height=50, bgcolor=ft.Colors.BLUE_800),
            ft.Column([
                ft.Text(
                    "🔍 Tübingen Search",
                    size=32,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_800
                ),
                ft.Text(
                    "Discover Tübingen's history, culture, and attractions",
                    size=16,
                    color=ft.Colors.GREY_600,
                    text_align=ft.TextAlign.CENTER
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
            ft.Container(width=50, height=50, bgcolor=ft.Colors.BLUE_800)]
            ))


ft.app(main)