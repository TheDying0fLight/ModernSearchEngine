import flet as ft

from .search_bar import SearchBarComponent

class SearchEnginePage:
    def __init__(self):
        self.page = None

    def search(self, key_words: str):
        self.page.add(ft.Text(f"Search Results for {key_words}"))


    def main(self, page: ft.Page):
        self.page = page
        page.title = "Search Engine"
        page.vertical_alignment = ft.MainAxisAlignment.START

        page.add(
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    SearchBarComponent(search_func=self.search)
                ],
            ),
    )
    def run(self, host: str, port: int):
        ft.app(self.main, view=ft.AppView.WEB_BROWSER, host=host, port=port)