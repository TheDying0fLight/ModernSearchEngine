import flet as ft
from .autocomplete_engine import get_autocomplete_engine


class SuggestionsView:

    def __init__(self, on_suggestion_click):
        self.on_suggestion_click = on_suggestion_click
        self.autocomplete_engine = get_autocomplete_engine()
        self.suggestions_container = ft.Column([], scroll=ft.ScrollMode.AUTO, spacing=0)
        self.loading_indicator = ft.Container(
            content=ft.Row([
                ft.ProgressRing(width=16, height=16, color=ft.Colors.BLUE_500),
                ft.Text("Loading suggestions...", size=12, color=ft.Colors.GREY_600)
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            padding=20,
            visible=False
        )
        
        # Initialize with loading state
        self.suggestions_container.controls.append(self.loading_indicator)
        
        # Check if engine is ready
        if self.autocomplete_engine.is_ready():
            self._show_default_suggestions()
        else:
            self.loading_indicator.visible = True
            # Check readiness periodically
            self._check_engine_readiness()

    def _check_engine_readiness(self):
        import threading
        import time
        
        def check_ready():
            while not self.autocomplete_engine.is_ready():
                time.sleep(0.5)
            # Engine is ready, update UI
            self.loading_indicator.visible = False
            self._show_default_suggestions()
            self.suggestions_container.update()
        
        thread = threading.Thread(target=check_ready, daemon=True)
        thread.start()

    def _show_default_suggestions(self):
        suggestions = self.autocomplete_engine.get_suggestions("", max_results=8)
        self._render_suggestions(suggestions, is_default=True)

    def _render_suggestions(self, suggestions, is_default=False):
        self.suggestions_container.controls.clear()
        
        if not suggestions:
            # Show no suggestions message
            self.suggestions_container.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SEARCH_OFF, color=ft.Colors.GREY_400, size=20),
                            ft.Text("No suggestions available", color=ft.Colors.GREY_600, size=14)
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
            )
            return
        
        # Group suggestions by type for better organization
        suggestion_groups = {}
        for suggestion in suggestions:
            suggestion_type = suggestion['type']
            if suggestion_type not in suggestion_groups:
                suggestion_groups[suggestion_type] = []
            suggestion_groups[suggestion_type].append(suggestion)
        
        # Render groups in priority order
        type_order = ['recent', 'dictionary', 'popular', 'pattern', 'question', 'fuzzy']
        type_headers = {
            'recent': 'Recent searches',
            'dictionary': 'Suggestions',
            'popular': 'Popular searches',
            'pattern': 'Related searches',
            'question': 'Questions',
            'fuzzy': 'Did you mean?'
        }
        
        for suggestion_type in type_order:
            if suggestion_type in suggestion_groups:
                group_suggestions = suggestion_groups[suggestion_type]
                
                # Add group header (except for default view)
                if not is_default and len(suggestion_groups) > 1:
                    header = ft.Container(
                        content=ft.Text(
                            type_headers.get(suggestion_type, suggestion_type.title()),
                            size=12,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        padding=ft.padding.only(left=16, top=8, bottom=4, right=16),
                        bgcolor=ft.Colors.BLUE_50,
                    )
                    self.suggestions_container.controls.append(header)
                
                # Add suggestions
                for suggestion in group_suggestions:
                    self._create_suggestion_tile(suggestion)

    def _create_suggestion_tile(self, suggestion):
        """Create a suggestion tile with Google-style design"""
        suggestion_text = suggestion['text']
        suggestion_type = suggestion['type']
        icon = suggestion.get('icon', '🔍')
        
        # Create icon based on type
        if suggestion_type == 'recent':
            leading_icon = ft.Icon(ft.Icons.HISTORY, size=18, color=ft.Colors.GREY_500)
        elif suggestion_type == 'popular':
            leading_icon = ft.Icon(ft.Icons.TRENDING_UP, size=18, color=ft.Colors.ORANGE_500)
        elif suggestion_type == 'fuzzy':
            leading_icon = ft.Icon(ft.Icons.SPELLCHECK, size=18, color=ft.Colors.PURPLE_500)
        elif suggestion_type == 'question':
            leading_icon = ft.Icon(ft.Icons.HELP_OUTLINE, size=18, color=ft.Colors.GREEN_500)
        else:
            leading_icon = ft.Icon(ft.Icons.SEARCH, size=18, color=ft.Colors.BLUE_500)
        
        # Create subtitle based on type
        subtitles = {
            'recent': 'Recent search',
            'popular': 'Popular search',
            'fuzzy': 'Suggested correction',
            'question': 'Common question',
            'pattern': 'Related search',
            'dictionary': 'Search suggestion'
        }
        
        subtitle_text = subtitles.get(suggestion_type, 'Search suggestion')
        
        # Create the suggestion tile
        tile = ft.ListTile(
            leading=leading_icon,
            title=ft.Text(
                suggestion_text,
                size=14,
                weight=ft.FontWeight.NORMAL,
                color=ft.Colors.GREY_800
            ),
            subtitle=ft.Text(
                subtitle_text,
                size=12,
                color=ft.Colors.GREY_600
            ),
            trailing=ft.Icon(
                ft.Icons.NORTH_WEST,
                size=14,
                color=ft.Colors.GREY_400
            ),
            on_click=lambda e, suggestion=suggestion_text: self.handle_suggestion_click(suggestion),
            hover_color=ft.Colors.GREY_50,
            selected_color=ft.Colors.BLUE_50,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=4)
        )
        
        self.suggestions_container.controls.append(tile)

    def handle_suggestion_click(self, suggestion):
        """Handle suggestion click"""
        if self.on_suggestion_click:
            self.on_suggestion_click(suggestion)

    def update_suggestions(self, query=""):
        """Update suggestions based on query"""
        if not self.autocomplete_engine.is_ready():
            self.loading_indicator.visible = True
            self.suggestions_container.controls.clear()
            self.suggestions_container.controls.append(self.loading_indicator)
            self.suggestions_container.update()
            return
        
        # Hide loading indicator
        self.loading_indicator.visible = False
        
        # Get suggestions from engine
        suggestions = self.autocomplete_engine.get_suggestions(query, max_results=8)
        
        # Render suggestions
        self._render_suggestions(suggestions, is_default=(not query))
        
        # Update UI
        self.suggestions_container.update()

    def get_container(self):
        """Get the suggestions container"""
        return self.suggestions_container

    def get_all_suggestions(self):
        """Get all available suggestions"""
        if self.autocomplete_engine.is_ready():
            return self.autocomplete_engine.get_suggestions("", max_results=20)
        return []

    def add_recent_search(self, query):
        """Add a recent search to suggestions"""
        if self.autocomplete_engine.is_ready():
            self.autocomplete_engine.add_recent_search(query)