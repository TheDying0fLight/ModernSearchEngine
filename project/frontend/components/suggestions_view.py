import flet as ft


class SuggestionsView:
    """Enhanced suggestions view component for the search bar"""

    def __init__(self, on_suggestion_click):
        self.on_suggestion_click = on_suggestion_click

        self.suggestion_categories = {
            "Popular Searches": [
                "tübingen attractions",
                "food and drinks",
                "historic sites",
                "restaurants",
                "university campus",
                "local events"
            ],
            "Attractions": [
                "Hohentübingen Castle",
                "Old Town",
                "Neckar River",
                "Botanical Garden",
                "Market Square",
                "Stiftskirche"
            ],
            "Food & Drinks": [
                "restaurants",
                "cafes",
                "beer gardens",
                "traditional cuisine",
                "local specialties",
                "wine bars"
            ],
            "Culture & Events": [
                "festivals",
                "museums",
                "theaters",
                "concerts",
                "art galleries",
                "cultural events"
            ]
        }
        # Flatten all suggestions for search
        self.all_suggestions = []
        for category, words in self.suggestion_categories.items():
            self.all_suggestions.extend(words)

        self.suggestions_container = self.create_suggestions_view()

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
                padding=ft.padding.only(left=16, top=8, bottom=4, right=16),
                bgcolor=ft.Colors.BLUE_50,
                border_radius=ft.border_radius.only(top_left=5, top_right=5)
            )
            suggestions_column.controls.append(category_header)

            # Category items
            for word in words:
                suggestion_tile = ft.ListTile(
                    leading=ft.Icon(
                        self.get_category_icon(category),
                        size=16,
                        color=ft.Colors.GREY_500
                    ),
                    title=ft.Text(word, size=14),
                    subtitle=ft.Text(
                        f"Search for '{word}'",
                        size=12,
                        color=ft.Colors.GREY_600
                    ),
                    on_click=lambda e, suggestion=word: self.handle_suggestion_click(suggestion),
                    hover_color=ft.Colors.BLUE_50,
                    selected_color=ft.Colors.BLUE_100
                )
                suggestions_column.controls.append(suggestion_tile)

        return suggestions_column

    def get_category_icon(self, category):
        """Get icon for category"""
        icons = {
            "Popular Searches": ft.Icons.TRENDING_UP,
            "Attractions": ft.Icons.PLACE,
            "Food & Drinks": ft.Icons.RESTAURANT,
            "Culture & Events": ft.Icons.EVENT
        }
        return icons.get(category, ft.Icons.SEARCH)

    def create_filtered_suggestions(self, query):
        """Create filtered suggestions based on query"""
        if not query:
            return self.create_suggestions_view()

        filtered_suggestions = ft.Column([], scroll=ft.ScrollMode.AUTO, spacing=2)

        # Find matching suggestions
        matching_suggestions = [
            word for word in self.all_suggestions
            if query.lower() in word.lower()
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
                padding=ft.padding.only(left=16, top=8, bottom=4, right=16),
                bgcolor=ft.Colors.BLUE_50,
                border_radius=ft.border_radius.only(top_left=5, top_right=5)
            )
            filtered_suggestions.controls.append(header)

            # Add matching suggestions (limit to 8 results)
            for word in matching_suggestions[:8]:
                suggestion_tile = ft.ListTile(
                    leading=ft.Icon(ft.Icons.SEARCH, size=16, color=ft.Colors.BLUE_500),
                    title=ft.Text(
                        self.highlight_match(word, query),
                        size=14
                    ),
                    subtitle=ft.Text(
                        f"Search for '{word}'",
                        size=12,
                        color=ft.Colors.GREY_600
                    ),
                    on_click=lambda e, suggestion=word: self.handle_suggestion_click(suggestion),
                    hover_color=ft.Colors.BLUE_50
                )
                filtered_suggestions.controls.append(suggestion_tile)
        else:
            # No matches found
            no_results = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.SEARCH_OFF, color=ft.Colors.GREY_400),
                        ft.Text("No suggestions found", color=ft.Colors.GREY_600)
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                    ft.Text(
                        "Try a different search term",
                        size=12,
                        color=ft.Colors.GREY_500,
                        text_align=ft.TextAlign.CENTER
                    )
                ], spacing=5),
                padding=20
            )
            filtered_suggestions.controls.append(no_results)

        return filtered_suggestions

    def highlight_match(self, text, query):
        """Highlight matching parts in suggestions"""
        return text

    def handle_suggestion_click(self, suggestion):
        """Handle suggestion click"""
        if self.on_suggestion_click:
            self.on_suggestion_click(suggestion)

    def update_suggestions(self, query=""):
        """Update suggestions based on query"""
        # Create new suggestions
        if query:
            new_suggestions = self.create_filtered_suggestions(query)
        else:
            new_suggestions = self.create_suggestions_view()

        self.suggestions_container.controls.clear()
        self.suggestions_container.controls.extend(new_suggestions.controls)

        # Trigger update
        self.suggestions_container.update()

    def get_container(self):
        """Get the suggestions container"""
        return self.suggestions_container

    def get_all_suggestions(self):
        """Get all available suggestions"""
        return self.all_suggestions

    def add_recent_search(self, query): #TODO
        """Add a recent search to suggestions"""
        pass