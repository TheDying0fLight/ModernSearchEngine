import flet as ft
from .tab import Tab, TabTitle

INFORMATION = {
    "About Tübingen Search Engine": "This search engine helps you find information about Tübingen's attractions, food, drinks, and local culture.",
    "How to Use:": "• Enter search terms in the search bar\n• Use the suggestions for popular searches\n• View your search history in the History tab\n• Save pages to favorites for quick access",
    "Project Team:": "Jan-Malte Giannikos • Simon Döhl • Carina Straub • Martin Eichler • Kilian Hunter",
    "Modern Search Engines Course": "University of Tübingen • 2025"
}

class HelpTab(Tab):
    """Help tab component with team information and usage instructions"""

    def __init__(self):
        infos = []
        for i, (title, text) in enumerate(INFORMATION.items()):
            if i != 0:
                infos.append(ft.Divider(height=20, color=ft.Colors.GREY_300))
            infos.append(ft.Text(title, size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700))
            infos.append(ft.Text(text, size=14, color=ft.Colors.GREY_700))
        super().__init__(
            text="Help",
            icon=ft.Icons.HELP,
            controls=[
                TabTitle("❓ Help"),
                ft.Container(
                    content=ft.Column(infos, spacing=10),
                    padding=30,
                    bgcolor=ft.Colors.WHITE,
                    border_radius=10,
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=4,
                        color=ft.Colors.GREY_200,
                        offset=ft.Offset(0, 2)
                    )
                )
            ]
        )