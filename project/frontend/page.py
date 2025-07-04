import flet as ft
import time

from .components import HelpTab, FavoritesTab, SearchTab, HistoryTab

class SearchEnginePage:
    """Main page of the search engine with component-based architecture"""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.is_searching = False
        self.current_results = []
        
        # Page setup with modern styling
        page.title = "Tübingen Search Engine"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.bgcolor = ft.Colors.GREY_50
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.padding = 0  # Remove padding for full width tabs
        
        # Initialize tab components
        self.help_tab = HelpTab()
        self.favorites_tab = FavoritesTab(on_navigate_to_search=self.navigate_to_search_tab)
        self.search_tab = SearchTab(search_func=self.search, on_favorite_toggle=self.handle_favorite_toggle)
        self.history_tab = HistoryTab()
        
        # Set up callbacks
        self.history_tab.set_callbacks(
            on_search=self.search_from_history,
            on_navigate=self.navigate_to_tab
        )
        
        # Build the layout
        self.build_layout()

    def build_layout(self):
        """Build the main page layout with tabs"""
        # Create tabs using the component containers
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tab_alignment=ft.TabAlignment.CENTER,
            expand=1,
            tabs=[
                ft.Tab(
                    text="Search",
                    icon=ft.Icons.SEARCH,
                    content=self.search_tab.get_container()
                ),
                ft.Tab(
                    text="History",
                    icon=ft.Icons.HISTORY,
                    content=self.history_tab.get_container()
                ),
                ft.Tab(
                    text="Favorites",
                    icon=ft.Icons.BOOKMARK,
                    content=self.favorites_tab.get_container()
                ),
                ft.Tab(
                    text="Help",
                    icon=ft.Icons.HELP,
                    content=self.help_tab.get_container()
                )
            ]
        )
        
        self.page.add(self.tabs)

    def navigate_to_search_tab(self):
        """Navigate to the search tab"""
        self.tabs.selected_index = 0
        self.page.update()

    def search(self, query: str):
        """Enhanced search function with loading state"""
        if not query.strip():
            return
            
        # Show loading in search tab
        self.search_tab.show_loading(True, f"Searching for '{query}'...")
        
        # Simulate search delay (replace with actual search logic)
        time.sleep(1)
        
        # Generate mock results
        mock_results = self.generate_mock_results(query)
        
        # Display results in search tab
        self.search_tab.display_results(query, mock_results)
        
        # Add to history
        self.history_tab.add_to_history(query, len(mock_results))
        
    def search_from_history(self, query):
        """Search triggered from history tab"""
        # Navigate to search tab and perform search
        self.tabs.selected_index = 0
        self.tabs.update()
        self.search(query)
        
    def navigate_to_tab(self, tab_index):
        """Navigate to a specific tab"""
        self.tabs.selected_index = tab_index
        self.tabs.update()
        
    def navigate_to_search_tab(self):
        """Navigate to search tab"""
        self.navigate_to_tab(0)
        
    def add_to_favorites(self, document):
        """Add a document to favorites"""
        return self.favorites_tab.add_favorite(document)
    
    def handle_favorite_toggle(self, result_data, is_favorited):
        """Handle favorite toggle from search results"""
        if is_favorited:
            return self.favorites_tab.add_favorite(result_data)
        else:
            self.favorites_tab.remove_favorite(result_data)
            return True
        
    def generate_mock_results(self, query):
        """Generate mock search results based on query"""
        # Enhanced mock results with more variety
        all_results = [
            {
                "id": "doc_001",
                "title": "Historical Overview of Tübingen's Old Town",
                "snippet": "Comprehensive guide to Tübingen's medieval architecture, historic buildings, and cultural landmarks in the old town area.",
                "source": "Tourism Board Archives",
                "date": "15.11.2024",
                "pages": 24
            },
            {
                "id": "doc_002", 
                "title": "Tübingen University: A 500-Year Legacy",
                "snippet": "The story of one of Germany's oldest universities, its famous alumni, and its impact on the city's development.",
                "source": "University Archives",
                "date": "03.09.2024",
                "pages": 45
            },
            {
                "id": "doc_003",
                "title": "Traditional Swabian Cuisine and Local Beverages",
                "snippet": "A culinary journey through Tübingen's traditional food scene, featuring local specialties, historic restaurants, and brewing traditions.",
                "source": "Regional Food Heritage Archive",
                "date": "22.08.2024",
                "pages": 18
            },
            {
                "id": "doc_004",
                "title": "Hohentübingen Castle: Fortress to Museum",
                "snippet": "The transformation of Tübingen's castle from medieval fortress to modern museum, including archaeological discoveries.",
                "source": "Museum Documentation",
                "date": "07.10.2024",
                "pages": 31
            },
            {
                "id": "doc_005",
                "title": "The Neckar River and Tübingen's Waterfront",
                "snippet": "How the Neckar River shaped Tübingen's history, from trade routes to modern recreational activities.",
                "source": "Environmental History Society",
                "date": "12.07.2024",
                "pages": 16
            },
            {
                "id": "doc_006",
                "title": "Festival Culture in Tübingen",
                "snippet": "Annual festivals, cultural events, and celebrations that define Tübingen's vibrant community life.",
                "source": "Cultural Events Archive",
                "date": "28.06.2024",
                "pages": 22
            }
        ]
        
        # Filter results based on query relevance
        query_lower = query.lower()
        relevant_results = []
        
        for result in all_results:
            title_lower = result["title"].lower()
            snippet_lower = result["snippet"].lower()
            
            # Simple relevance scoring
            if any(word in title_lower or word in snippet_lower for word in query_lower.split()):
                relevant_results.append(result)
        
        # Return relevant results or all results if no specific matches
        return relevant_results if relevant_results else all_results[:3]


class PageFactory:
    """Creates new pages of type `SearchEnginePage`"""
    def __init__(self):
        pass

    def create_page(self, page: ft.Page):
            search_engine_page = SearchEnginePage(page)
            def on_connect(e):
                page.clean()
                search_engine_page.build_layout()
            page.on_connect = on_connect

    def run(self, host: str, port: int):
        ft.app(self.create_page, view=ft.AppView.WEB_BROWSER, host=host, port=port)