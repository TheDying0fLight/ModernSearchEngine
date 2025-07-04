import flet as ft


class FavoritesTab:
    """Favorites tab component for managing bookmarked documents"""
    
    def __init__(self, on_navigate_to_search=None):
        self.favorites = []
        self.on_navigate_to_search = on_navigate_to_search  # Callback for navigation
        self.container = self.create_favorites_content()
    
    def create_favorites_content(self):
        """Create the favorites tab content"""
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
                self.create_favorites_list()
            ], scroll=ft.ScrollMode.AUTO),
            padding=20,
            alignment=ft.alignment.top_center,
            expand=True
        )
    
    def create_favorites_list(self):
        """Create the favorites list or empty state"""
        if not self.favorites:
            return self.create_empty_state()
        
        # Create list of favorite items
        favorites_column = ft.Column([], spacing=10)
        for favorite in self.favorites:
            favorite_card = self.create_favorite_card(favorite)
            favorites_column.controls.append(favorite_card)
        
        return ft.Container(
            content=favorites_column,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            padding=20,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=4,
                color=ft.Colors.GREY_200,
                offset=ft.Offset(0, 2)
            )
        )
    
    def create_empty_state(self):
        """Create empty state when no favorites exist"""
        return ft.Container(
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
                ),
                ft.ElevatedButton(
                    text="Go to Search",
                    icon=ft.Icons.SEARCH,
                    on_click=self.navigate_to_search,
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
    
    def create_favorite_card(self, favorite):
        """Create a card for a favorite item"""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(
                        favorite.get("title", "Untitled"),
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_700,
                        expand=True
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_color=ft.Colors.RED_400,
                        tooltip="Remove from favorites",
                        on_click=lambda e, fav=favorite: self.remove_favorite(fav)
                    )
                ]),
                ft.Text(
                    favorite.get("snippet", "No description available."),
                    size=14,
                    color=ft.Colors.GREY_700,
                    max_lines=2
                ),
                ft.Row([
                    ft.Text(
                        favorite.get("source", "Unknown"),
                        color=ft.Colors.GREEN_600,
                        size=12
                    ),
                    ft.Text("•", color=ft.Colors.GREY_400),
                    ft.Text(
                        f"Added: {favorite.get('added_date', 'Unknown')}",
                        color=ft.Colors.GREY_600,
                        size=12
                    )
                ])
            ], spacing=8),
            padding=15,
            margin=ft.margin.only(bottom=10),
            bgcolor=ft.Colors.GREY_50,
            border_radius=8,
            border=ft.border.all(1, ft.Colors.GREY_200),
            on_click=lambda e, fav=favorite: self.open_favorite(fav)
        )
    
    def add_favorite(self, document):
        """Add a document to favorites"""
        from datetime import datetime
        
        # Check if already in favorites
        if any(fav.get("id") == document.get("id") for fav in self.favorites):
            return False
        
        # Add timestamp
        document["added_date"] = datetime.now().strftime("%m/%d/%Y")
        self.favorites.insert(0, document)
        self.refresh_display()
        return True
    
    def remove_favorite(self, favorite):
        """Remove a document from favorites"""
        self.favorites = [fav for fav in self.favorites if fav.get("id") != favorite.get("id")]
        self.refresh_display()
    
    def open_favorite(self, favorite):
        """Open a favorite document"""
        # This would typically navigate to the document or open it
        print(f"Opening favorite: {favorite.get('title')}")
    
    def navigate_to_search(self, e):
        """Navigate to search tab"""
        if self.on_navigate_to_search:
            self.on_navigate_to_search()
        else:
            print("Navigate to search tab - no callback provided")
    
    def refresh_display(self):
        """Refresh the favorites display"""
        # Update the content
        self.container.content.controls[1] = self.create_favorites_list()
        self.container.update()
    
    def get_container(self):
        """Get the favorites tab container"""
        return self.container
