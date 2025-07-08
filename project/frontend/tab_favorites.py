import flet as ft
from datetime import datetime

from .tab import Tab, TabTitle
from .components import EmptyState, ResultContainer

class FavoritesTab(Tab):
    """Favorites tab component for managing bookmarked documents"""

    def __init__(self, on_navigate_to_search):
        self.favorites = []
        self.empty_state = EmptyState(
            icon=ft.Icons.BOOKMARK_BORDER,
            title="No favorites yet",
            text="Add documents to your favorites to see them here",
            button_icon=ft.Icons.SEARCH,
            button_text="Start Searching",
            on_button_click=lambda e: on_navigate_to_search(),
        )
        self.favorites_list = ft.Container(self.empty_state)
        super().__init__(
            text="Favorites",
            icon=ft.Icons.BOOKMARK,
            controls=[
                TabTitle("⭐ Favorites"),
                self.favorites_list
            ]
        )

    def update_favorites_list(self):
        if not self.favorites:
            self.favorites_list.content = self.empty_state
        else:
            self.favorites_list.content = ft.Column([
                FavoriteCard(favorite, self.remove_favorite, self.open_favorite)
                for favorite in self.favorites
            ], expand=True)
        self.favorites_list.update()

    def add_favorite(self, document):
        """Add a document to favorites"""
        # Check if already in favorites
        if any(fav.get("id") == document.get("id") for fav in self.favorites):
            return False

        # Add timestamp
        document["added_date"] = datetime.now().strftime("%m/%d/%Y")
        self.favorites.insert(0, document)
        self.update_favorites_list()
        return True

    def remove_favorite(self, favorite):
        """Remove a document from favorites"""
        self.favorites = [fav for fav in self.favorites if fav.get("id") != favorite.get("id")]
        self.update_favorites_list()

    def open_favorite(self, favorite):
        """Open a favorite document"""
        # This would typically navigate to the document or open it
        print(f"Opening favorite: {favorite.get('title')}")



class FavoriteCard(ResultContainer):
    def __init__(self, favorite, on_remove_favorite, on_open_favorite):
        delete_button=ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_color=ft.Colors.RED_400,
            tooltip="Remove from favorites",
            on_click=lambda e, fav=favorite: on_remove_favorite(fav)
        )
        super().__init__(
            title=favorite.get("title", "Untitled"),
            text=favorite.get("snippet", "No description available."),
            source=favorite.get("source", "Unknown"),
            metadata=[f"Added: {favorite.get('added_date', 'Unknown')}"],
            button=delete_button,
            on_click=lambda e, fav=favorite: on_open_favorite(fav))
