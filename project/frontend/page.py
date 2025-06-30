import flet as ft
import time
from datetime import datetime

from .search_bar import SearchBarComponent

class SearchEnginePage:
    """Main page of the search engine with enhanced features"""
    def __init__(self, page: ft.Page):
        self.page = page
        self.search_history = []
        self.is_searching = False
        self.current_results = []
        
        # Page setup with modern styling
        page.title = "Tübingen Search Engine"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.bgcolor = ft.Colors.GREY_50
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.padding = 0  # Remove padding for full width tabs
        
        # Initialize UI components
        self.init_components()
        self.build_layout()

    def init_components(self):
        """Initialize all UI components"""
        # Header with logo/title
        self.header = ft.Container(
            content=ft.Column([
                ft.Text(
                    "🔍 Tübingen Search", 
                    size=32, 
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_800
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            margin=ft.margin.only(bottom=30)
        )
        
        # Search bar
        self.search_bar = SearchBarComponent(search_func=self.search)
        
        # Loading indicator
        self.loading_indicator = ft.Container(
            content=ft.Row([
                ft.ProgressRing(width=20, height=20),
                ft.Text("Searching...", color=ft.Colors.BLUE_600)
            ], alignment=ft.MainAxisAlignment.CENTER),
            visible=False,
            margin=ft.margin.only(top=20)
        )
        
        # Search results container
        self.results_container = ft.Container(
            content=ft.Column([]),
            margin=ft.margin.only(top=20),
            visible=False
        )
        
        # Search history
        self.history_container = ft.Container(
            content=ft.Column([
                ft.Text("Search History", size=18, weight=ft.FontWeight.BOLD),
                ft.Column([], scroll=ft.ScrollMode.AUTO, height=200)
            ]),
            margin=ft.margin.only(top=30),
            padding=20,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            visible=False
        )
        
        # Advanced search options
        self.advanced_options = ft.ExpansionTile(
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
                    ]),
                    padding=10
                )
            ]
        )

    def create_search_tab(self):
        """Create the main search tab content"""
        return ft.Container(
            content=ft.Column([
                self.header,
                self.search_bar,  # Direkt die SearchBar ohne Container
                self.advanced_options,
                self.loading_indicator,
                self.results_container,
                self.history_container
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20,
            alignment=ft.alignment.top_center
        )

    def create_favorites_tab(self):
        """Create the favorites/bookmarks tab content"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Text(
                        "⭐ Favorites", 
                        size=24, 
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_800
                    ),
                    margin=ft.margin.only(bottom=20)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.BOOKMARK_BORDER, size=64, color=ft.Colors.GREY_400),
                        ft.Text(
                            "No favorites yet",
                            size=18,
                            color=ft.Colors.GREY_600,
                            weight=ft.FontWeight.BOLD
                        ),
                        ft.Text(
                            "Add documents to your favorites to see them here",
                            size=14,
                            color=ft.Colors.GREY_500,
                            text_align=ft.TextAlign.CENTER
                        )
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                    padding=50,
                    alignment=ft.alignment.center,
                    bgcolor=ft.Colors.WHITE,
                    border_radius=10,
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=4,
                        color=ft.Colors.GREY_200,
                        offset=ft.Offset(0, 2)
                    )
                )
            ]),
            padding=20,
            alignment=ft.alignment.top_center
        )

    def create_help_tab(self):
        """Create the help/about tab content"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Text(
                        "❓ Help", 
                        size=24, 
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_800
                    ),
                    margin=ft.margin.only(bottom=20)
                ),
                
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "This search engine helps you find information about Tübingen's attractions, food, drinks, and local culture.",
                            size=16,
                            color=ft.Colors.GREY_700,
                            text_align=ft.TextAlign.CENTER
                        ),
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        ft.Text(
                            "Project Team:",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        ft.Text(
                            "Jan-Malte Giannikos • Simon Döhl • Carina Straub\nMartin Eichler • Kilian Hunter",
                            size=14,
                            color=ft.Colors.GREY_600,
                            text_align=ft.TextAlign.CENTER
                        )
                    ], spacing=10),
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
            ], scroll=ft.ScrollMode.AUTO),
            padding=20,
            alignment=ft.alignment.top_center
        )

    def build_layout(self):
        """Build the main page layout with tabs"""
        # Create tabs
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tab_alignment=ft.TabAlignment.CENTER,
            expand=1,
            tabs=[
                ft.Tab(
                    text="Search",
                    icon=ft.Icons.SEARCH,
                    content=self.create_search_tab()
                ),
                ft.Tab(
                    text="Favorites",
                    icon=ft.Icons.BOOKMARK,
                    content=self.create_favorites_tab()
                ),
                ft.Tab(
                    text="Help",
                    icon=ft.Icons.HELP,
                    content=self.create_help_tab()
                )
            ]
        )
        
        self.page.add(tabs)

    def search(self, key_words: str):
        """Enhanced search function with loading state"""
        if not key_words.strip():
            return
            
        # Add to search history
        self.add_to_history(key_words)
        
        # Show loading
        self.show_loading(True)
        self.results_container.visible = False
        self.page.update()
        
        # Simulate search delay (replace with actual search logic)
        time.sleep(1)
        
        # Hide loading and show results
        self.show_loading(False)
        self.display_results(key_words)
        
    def show_loading(self, show: bool):
        """Show/hide loading indicator"""
        self.loading_indicator.visible = show
        self.is_searching = show
        self.page.update()
        
    def add_to_history(self, query: str):
        """Add search query to history"""
        timestamp = datetime.now().strftime("%H:%M")
        self.search_history.insert(0, {"query": query, "time": timestamp})
        
        # Keep only last 10 searches
        if len(self.search_history) > 10:
            self.search_history = self.search_history[:10]
            
        self.update_history_display()
        
    def update_history_display(self):
        """Update the search history display"""
        history_column = self.history_container.content.controls[1]
        history_column.controls.clear()
        
        for item in self.search_history:
            history_item = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.HISTORY, size=16, color=ft.Colors.GREY_600),
                    ft.Text(item["query"], expand=True),
                    ft.Text(item["time"], color=ft.Colors.GREY_500, size=12)
                ]),
                padding=8,
                border_radius=5,
                bgcolor=ft.Colors.GREY_100,
                margin=ft.margin.only(bottom=5),
                on_click=lambda e, query=item["query"]: self.search(query)
            )
            history_column.controls.append(history_item)
            
        self.history_container.visible = len(self.search_history) > 0
        self.page.update()
        
    def create_simple_results_view(self, results: list):
        """Create a simple results view without document viewer"""
        if not results:
            return ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.SEARCH_OFF, size=64, color=ft.Colors.GREY_400),
                    ft.Text(
                        "No results found",
                        size=18,
                        color=ft.Colors.GREY_600,
                        weight=ft.FontWeight.BOLD
                    ),
                    ft.Text(
                        "Try different search terms",
                        size=14,
                        color=ft.Colors.GREY_500
                    )
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                padding=50,
                alignment=ft.alignment.center
            )
        
        # Create simple result cards
        result_cards = []
        for result in results:
            result_card = ft.Container(
                content=ft.Column([
                    ft.Text(
                        result.get("title", "No Title"),
                        size=18,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_700
                    ),
                    ft.Text(
                        result.get("snippet", "No description available."),
                        size=14,
                        color=ft.Colors.GREY_700,
                        max_lines=3
                    ),
                    ft.Row([
                        ft.Text(result.get("source", "Unknown"), color=ft.Colors.GREEN_600, size=12),
                        ft.Text("•", color=ft.Colors.GREY_400),
                        ft.Text(result.get("date", "Unknown"), color=ft.Colors.GREY_600, size=12)
                    ])
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
                )
            )
            result_cards.append(result_card)
        
        return ft.Column(
            controls=result_cards,
            scroll=ft.ScrollMode.AUTO,
            spacing=0
        )
        
    def display_results(self, query: str):
        """Display search results using DocumentViewer"""
        # Mock results based on actual queries
        mock_results = [
            {
                "id": "doc_003",
                "title": "Traditional Swabian Cuisine and Local Beverages [EXAMPLE]",
                "snippet": "A historical account of traditional food and drinks in Tübingen. Documents local specialties, taverns, and the brewing tradition of the region.",
                "full_text": "Historical overview of Swabian cuisine including Maultaschen, Spätzle, and regional wines. The document also covers traditional beer gardens and historic taverns...",
                "date": "03.03.2025",
                "source": "Regional Food Heritage Archive",
                "pages": 18
            },
            {
                "id": "doc_004",
                "title": "Historic Restaurants and Gastronomy [ex2]",
                "snippet": "Chronicle of Tübingen's culinary history, featuring historic restaurants, traditional recipes, and local food markets that have served the community for centuries.",
                "full_text": "Comprehensive record of Tübingen's gastronomic heritage, including the famous Wurstküche, traditional bakeries, and the historic market square food vendors...",
                "date": "08.26.2001",
                "source": "City Historical Society",
                "pages": 12
            }
        ]
        
        self.current_results = mock_results
        
        # Create simple results display
        results_view = self.create_simple_results_view(mock_results)
        
        results_header = ft.Container(
            content=ft.Text(
                f"Search results for '{query}' ({len(mock_results)} results)",
                size=20,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLUE_800
            ),
            margin=ft.margin.only(bottom=20)
        )
        
        # Combine header and results
        self.results_container.content = ft.Column([
            results_header,
            results_view
        ])
        
        self.results_container.visible = True
        self.page.update()


class PageFactory:
    """Creates new pages of type `SearchEnginePage`"""
    def __init__(self):
        pass

    def create_page(self, page: ft.Page):
        search_engine_page = SearchEnginePage(page)

    def run(self, host: str, port: int):
        ft.app(self.create_page, view=ft.AppView.WEB_BROWSER, host=host, port=port)