import flet as ft


class ResultCard:
    """Individual result card component"""
    
    def __init__(self, result_data, on_click_callback=None, on_favorite_callback=None):
        self.result_data = result_data
        self.on_click_callback = on_click_callback
        self.on_favorite_callback = on_favorite_callback
        self.is_favorited = False
        self.container = self.create_result_card()
    
    def create_result_card(self):
        """Create a result card container"""
        return ft.Container(
            content=ft.Column([
                # Title and favorite button row
                ft.Row([
                    ft.Text(
                        self.result_data.get("title", "No Title"),
                        size=18,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_700,
                        expand=True
                    ),
                    ft.IconButton(
                        icon=ft.Icons.BOOKMARK_BORDER if not self.is_favorited else ft.Icons.BOOKMARK,
                        icon_color=ft.Colors.GREY_600 if not self.is_favorited else ft.Colors.ORANGE_600,
                        tooltip="Add to favorites" if not self.is_favorited else "Remove from favorites",
                        on_click=self.toggle_favorite
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                # Description/snippet
                ft.Text(
                    self.result_data.get("snippet", "No description available."),
                    size=14,
                    color=ft.Colors.GREY_700,
                    max_lines=3
                ),
                
                # Metadata row
                ft.Row([
                    ft.Text(
                        self.result_data.get("source", "Unknown"),
                        color=ft.Colors.GREEN_600,
                        size=12,
                        weight=ft.FontWeight.BOLD
                    ),
                    ft.Text("•", color=ft.Colors.GREY_400, size=12),
                    ft.Text(
                        self.result_data.get("date", "Unknown"),
                        color=ft.Colors.GREY_600,
                        size=12
                    ),
                    ft.Text("•", color=ft.Colors.GREY_400, size=12),
                    ft.Text(
                        f"{self.result_data.get('pages', 'N/A')} pages",
                        color=ft.Colors.GREY_600,
                        size=12
                    )
                ], spacing=5)
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
            ),
            on_click=lambda e: self.handle_click()
        )
    
    def toggle_favorite(self, e):
        """Toggle favorite status"""
        self.is_favorited = not self.is_favorited
        
        # Update the icon
        favorite_button = self.container.content.controls[0].controls[1]
        favorite_button.icon = ft.Icons.BOOKMARK if self.is_favorited else ft.Icons.BOOKMARK_BORDER
        favorite_button.icon_color = ft.Colors.ORANGE_600 if self.is_favorited else ft.Colors.GREY_600
        favorite_button.tooltip = "Remove from favorites" if self.is_favorited else "Add to favorites"
        
        # Call callback if provided
        if self.on_favorite_callback:
            self.on_favorite_callback(self.result_data, self.is_favorited)
        
        self.container.update()
    
    def handle_click(self):
        """Handle result card click"""
        if self.on_click_callback:
            self.on_click_callback(self.result_data)
    
    def get_container(self):
        """Get the result card container"""
        return self.container


class ResultsView:
    """Results view component for displaying search results"""
    
    def __init__(self, on_result_click=None, on_favorite_toggle=None):
        self.on_result_click = on_result_click
        self.on_favorite_toggle = on_favorite_toggle
        self.results = []
        self.container = self.create_results_container()
    
    def create_results_container(self):
        """Create the results container"""
        return ft.Container(
            content=ft.Column([]),
            margin=ft.margin.only(top=20),
            visible=False
        )
    
    def display_results(self, query, results):
        """Display search results"""
        self.results = results
        
        if not results:
            self.show_no_results(query)
        else:
            self.show_results(query, results)
        
        self.container.visible = True
        self.container.update()
    
    def show_no_results(self, query):
        """Show no results state"""
        no_results_content = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.SEARCH_OFF, size=64, color=ft.Colors.GREY_400),
                ft.Text(
                    "No results found",
                    size=18,
                    color=ft.Colors.GREY_600,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Text(
                    f"No documents found for '{query}'",
                    size=14,
                    color=ft.Colors.GREY_500,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.Text(
                    "Try different search terms or check your spelling",
                    size=14,
                    color=ft.Colors.GREY_500,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.ElevatedButton(
                    text="Try Advanced Search",
                    icon=ft.Icons.TUNE,
                    on_click=self.show_advanced_search,
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.BLUE_600,
                        color=ft.Colors.WHITE
                    )
                )
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15),
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
        
        self.container.content = ft.Column([
            self.create_results_header(query, 0),
            no_results_content
        ])
    
    def show_results(self, query, results):
        """Show search results"""
        # Create result cards
        result_cards = []
        for result in results:
            result_card = ResultCard(
                result_data=result,
                on_click_callback=self.on_result_click,
                on_favorite_callback=self.on_favorite_toggle
            )
            result_cards.append(result_card.get_container())
        
        results_list = ft.Column(
            controls=result_cards,
            scroll=ft.ScrollMode.AUTO,
            spacing=0
        )
        
        self.container.content = ft.Column([
            self.create_results_header(query, len(results)),
            results_list
        ])
    
    def create_results_header(self, query, count):
        """Create the results header"""
        return ft.Container(
            content=ft.Text(
                f"Search results for '{query}' ({count} result{'s' if count != 1 else ''})",
                size=20,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLUE_800
            ),
            margin=ft.margin.only(bottom=20)
        )
    
    def show_advanced_search(self, e):
        """Show advanced search options"""
        print("Show advanced search")
    
    def hide_results(self):
        """Hide the results container"""
        self.container.visible = False
        self.container.update()
    
    def get_container(self):
        """Get the results container"""
        return self.container
    
    def get_results_count(self):
        """Get the number of current results"""
        return len(self.results)
