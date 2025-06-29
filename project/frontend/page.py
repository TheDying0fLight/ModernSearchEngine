import flet as ft

from .search_bar import SearchBarComponent

class SearchEnginePage:
    def __init__(self, page: ft.Page):
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

    def search(self, key_words: str):
        self.page.add(ft.Text(f"Search Results for {key_words}"))


class PageFactory:
    def __init__(self):
        pass

    def create_page(self, page: ft.Page):
        search_engine_page = SearchEnginePage(page)

    def run(self, host: str, port: int):
        ft.app(self.create_page, view=ft.AppView.WEB_BROWSER, host=host, port=port)