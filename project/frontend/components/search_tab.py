import flet as ft
from ..enhanced_search_bar import EnhancedSearchBar
from .loading_indicator import LoadingIndicator
from .results_view import ResultsView


class SearchTab:
    """Main search tab component containing search interface and results"""
    
    def __init__(self, search_func, on_favorite_toggle=None):
        self.search_func = search_func
        self.on_favorite_toggle = on_favorite_toggle
        
        # Initialize components
        self.search_bar = EnhancedSearchBar(search_func=self.handle_search)
        self.loading_indicator = LoadingIndicator("Searching documents...")
        self.results_view = ResultsView(
            on_result_click=self.handle_result_click,
            on_favorite_toggle=self.handle_favorite_toggle
        )
        
        # Create the tab container
        self.container = self.create_search_tab()
    
    def create_search_tab(self):
        """Create the main search tab content"""
        # Header with logo/title
        header = ft.Container(
            content=ft.Column([
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
                )
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            margin=ft.margin.only(bottom=30)
        )
        
        # Advanced search options
        advanced_options = ft.ExpansionTile(
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
        
        return ft.Container(
            content=ft.Column([
                header,
                self.search_bar,
                advanced_options,
                self.loading_indicator.get_container(),
                self.results_view.get_container()
            ], 
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
            spacing=10,
            scroll=ft.ScrollMode.AUTO),
            padding=20,
            alignment=ft.alignment.top_center,
            expand=True
        )
    
    def handle_search(self, query):
        """Handle search request"""
        if not query.strip():
            return
        
        # Show loading
        self.loading_indicator.show(f"Searching for '{query}'...")
        self.results_view.hide_results()
        
        # Call the main search function
        if self.search_func:
            self.search_func(query)
    
    def handle_result_click(self, result_data):
        """Handle result click"""
        print(f"Opening result: {result_data.get('title')}")
        # This would typically open the document or navigate to detail view
    
    def handle_favorite_toggle(self, result_data, is_favorited):
        """Handle favorite toggle"""
        if self.on_favorite_toggle:
            success = self.on_favorite_toggle(result_data, is_favorited)
            if success:
                action = "Added to" if is_favorited else "Removed from"
                print(f"{action} favorites: {result_data.get('title')}")
        else:
            print(f"Favorite toggle for: {result_data.get('title')}")
    
    def display_results(self, query, results):
        """Display search results"""
        self.loading_indicator.hide()
        self.results_view.display_results(query, results)
    
    def show_loading(self, show=True, message=None):
        """Show/hide loading indicator"""
        if show:
            self.loading_indicator.show(message)
        else:
            self.loading_indicator.hide()
    
    def get_container(self):
        """Get the search tab container"""
        return self.container
    
    def clear_results(self):
        """Clear current search results"""
        self.results_view.hide_results()
        self.loading_indicator.hide()
