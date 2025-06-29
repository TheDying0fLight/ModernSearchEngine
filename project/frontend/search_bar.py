import flet as ft

class SearchBarComponent(ft.SearchBar):
    """Search bar with suggestions"""

    def __init__(self, search_func):
        self.search_func = search_func
        self.dummy_words = ["hello", "tubingen", "tübingen", "search", "engine", "class", "historic"] # TODO: Actual suggestions
        self.suggestions = ft.ListView([
                ft.ListTile(title=ft.Text(word), on_click=self.handle_suggestion_submit, data=word)
                for word in self.dummy_words
            ])
        super().__init__(
            view_elevation=4,
            divider_color=ft.Colors.BLUE,
            bar_hint_text="Search in Tübingen ...",
            view_hint_text="Search in Tübingen ...",
            on_change=self.handle_change,
            on_submit=self.handle_submit,
            on_tap=self.handle_tap,
            controls=[self.suggestions],
        )

    def handle_change(self, e: ft.ControlEvent):
        self.suggestions.controls = [
                ft.ListTile(title=ft.Text(word), on_click=self.handle_suggestion_submit, data=word)
                for word in self.dummy_words if e.data in word # TODO: Find actual suggestions
            ]
        self.update()

    def handle_submit(self, e: ft.ControlEvent):
        self.submit(e.data)

    def handle_suggestion_submit(self, e: ft.ControlEvent):
        self.submit(e.control.data)

    def submit(self, data):
        self.close_view()
        self.search_func(data)

    def handle_tap(self, e):
        self.open_view()