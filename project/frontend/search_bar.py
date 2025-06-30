import flet as ft

class SearchBarComponent(ft.SearchBar):
    """Enhanced search bar with better suggestions and styling"""

    def __init__(self, search_func):
        self.search_func = search_func
        
        # Enhanced dummy suggestions with categories
        self.suggestion_categories = {
            "Popular Searches": ["tübingen attractions", "food and drinks", "historic sites", "restaurants"],
            "Attractions": ["Castle", "Old Town"],
            "Food & Drinks": ["restaurants", "cafes"],
        }
        
        # Flatten all suggestions for search
        self.all_suggestions = []
        for category, words in self.suggestion_categories.items():
            self.all_suggestions.extend(words)
        
        self.suggestions_view = self.create_suggestions_view()
        
        super().__init__(
            width=500,
            view_elevation=8,
            divider_color=ft.Colors.BLUE_300,
            bar_hint_text="Search documents about Tübingen...",
            view_hint_text="Search suggestions",
            bar_leading=ft.Icon(ft.Icons.SEARCH, color=ft.Colors.BLUE_600),
            view_leading=ft.Icon(ft.Icons.SEARCH, color=ft.Colors.BLUE_600),
            on_change=self.handle_change,
            on_submit=self.handle_submit,
            on_tap=self.handle_tap,
            controls=[self.suggestions_view],
        )

    def create_suggestions_view(self):
        """Create enhanced suggestions view with categories"""
        suggestions_column = ft.Column([], scroll=ft.ScrollMode.AUTO, spacing=5)
        
        for category, words in self.suggestion_categories.items():
            # Category header
            category_header = ft.Container(
                content=ft.Text(
                    category, 
                    size=14, 
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_700
                ),
                padding=ft.padding.only(left=16, top=8, bottom=4)
            )
            suggestions_column.controls.append(category_header)
            
            # Category items
            for word in words:
                suggestion_tile = ft.ListTile(
                    leading=ft.Icon(ft.Icons.HISTORY, size=16, color=ft.Colors.GREY_500),
                    title=ft.Text(word, size=14),
                    subtitle=ft.Text(f"Search for '{word}'", size=12, color=ft.Colors.GREY_600),
                    on_click=self.handle_suggestion_submit,
                    data=word,
                    hover_color=ft.Colors.BLUE_50
                )
                suggestions_column.controls.append(suggestion_tile)
        
        return suggestions_column

    def handle_change(self, e: ft.ControlEvent):
        """Handle search input changes with improved filtering"""
        query = e.data.lower().strip()
        
        if not query:
            # Show all categories when empty
            self.suggestions_view = self.create_suggestions_view()
        else:
            # Filter suggestions based on query
            filtered_suggestions = ft.Column([], scroll=ft.ScrollMode.AUTO, spacing=2)
            
            matching_suggestions = [
                word for word in self.all_suggestions 
                if query in word.lower()
            ]
            
            if matching_suggestions:
                # Add header for filtered results
                header = ft.Container(
                    content=ft.Text(
                        f"Suggestions for '{query}'", 
                        size=14, 
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_700
                    ),
                    padding=ft.padding.only(left=16, top=8, bottom=4),
                    bgcolor=ft.Colors.BLUE_50
                )
                filtered_suggestions.controls.append(header)
                
                # Add matching suggestions
                for word in matching_suggestions[:8]:  # Limit to 8 results
                    suggestion_tile = ft.ListTile(
                        leading=ft.Icon(ft.Icons.SEARCH, size=16, color=ft.Colors.BLUE_500),
                        title=ft.Text(
                            self.highlight_match(word, query), 
                            size=14
                        ),
                        on_click=self.handle_suggestion_submit,
                        data=word,
                        hover_color=ft.Colors.BLUE_50
                    )
                    filtered_suggestions.controls.append(suggestion_tile)
            else:
                # No matches found
                no_results = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.SEARCH_OFF, color=ft.Colors.GREY_400),
                        ft.Text("No suggestions found", color=ft.Colors.GREY_600)
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=20
                )
                filtered_suggestions.controls.append(no_results)
            
            self.suggestions_view = filtered_suggestions
        
        self.controls = [self.suggestions_view]
        self.update()

    def highlight_match(self, text: str, query: str):
        """Highlight matching parts in suggestions (simplified version)"""
        # For now, just return the text as-is
        # In a more advanced version, you could return RichText with highlighting
        return text

    def handle_submit(self, e: ft.ControlEvent):
        """Handle search submission"""
        if e.data and e.data.strip():
            self.submit(e.data.strip())

    def handle_suggestion_submit(self, e: ft.ControlEvent):
        """Handle suggestion selection"""
        self.submit(e.control.data)

    def submit(self, data):
        """Submit search query"""
        self.close_view()
        self.value = data  # Set the search bar value
        self.update()
        self.search_func(data)

    def handle_tap(self, e):
        """Handle search bar tap to open suggestions"""
        self.open_view()