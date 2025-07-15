import flet as ft
from .components import LoadingIndicator, EmptyState, SearchBar
from .results_view import ResultsView, Result
from.tab import Tab

class SearchTab(Tab):
    """Main search tab component containing search interface and results"""

    def __init__(self, page: ft.Page, on_favorite_toggle=None):
        self.page = page
        self.on_favorite_toggle = on_favorite_toggle

        # Initialize components
        self.search_bar = SearchBar(search_func=lambda query: self.page.go(f'/search?q={query}'))
        self.loading_indicator = LoadingIndicator("Searching documents...")
        self.results_view = ResultsView(self.handle_favorite_toggle, self.handle_result_click)
        self.header = ft.Column([
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
                self.search_bar,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)

        self.advanced_options = AdvancedSearchOptions()

        # Create the tab container
        super().__init__(
            text="Search",
            icon=ft.Icons.SEARCH,
            controls=[
                self.header,
                self.advanced_options,
                self.loading_indicator,
                self.results_view
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def handle_result_click(self, result_data: Result):
        """Handle result click"""
        self.page.launch_url(result_data.url)

    def handle_favorite_toggle(self, result_data: Result, is_favorited):
        """Handle favorite toggle"""
        if self.on_favorite_toggle:
            success = self.on_favorite_toggle(result_data, is_favorited)
            if success:
                action = "Added to" if is_favorited else "Removed from"
                print(f"{action} favorites: {result_data.title}")
        else:
            print(f"Favorite toggle for: {result_data.title}")

    def start_loading(self, query):
        self.loading_indicator.show(f"Searching for '{query}'...")

    def display_results(self, query, results):
        """Display search results"""
        self.loading_indicator.hide()
        self.results_view.show_results(query, results)

    def clear_results(self):
        """Clear current search results"""
        self.results_view.hide_results()

class AdvancedSearchOptions(ft.ExpansionTile):
    def __init__(self):
        super().__init__(
            title=ft.Text("Advanced Search Options"),
            subtitle=ft.Text("Filters and settings"),
            collapsed_text_color=ft.Colors.BLUE_600,
            text_color=ft.Colors.BLUE_800,
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Checkbox(label="Search titles only", value=False),
                            ft.Checkbox(label="Exact phrase search", value=False)
                        ]),
                        ft.Row([
                            ft.Checkbox(label="Include historical documents", value=True),
                            ft.Checkbox(label="Include recent articles", value=True)
                        ]),
                        ft.Divider(),
                        ft.Row([
                            ft.Text("Document type:", size=14, color=ft.Colors.GREY_700),
                            ft.Dropdown(
                                width=150,
                                value="all",
                                options=[
                                    ft.dropdown.Option("all", "All types"),
                                    ft.dropdown.Option("articles", "Articles"),
                                    ft.dropdown.Option("guides", "Guides"),
                                    ft.dropdown.Option("historical", "Historical")
                                ]
                            )
                        ], spacing=10)
                    ], spacing=10),
                    padding=10
                )
            ]
        )