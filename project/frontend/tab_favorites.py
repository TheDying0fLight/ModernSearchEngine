from dataclasses import dataclass, asdict
import flet as ft
from datetime import datetime

from .tab import Tab, TabTitle
from .components import EmptyState, ResultContainer
from .results_view import Result

class Favorite(Result):
    added_date: str

    def __init__(self, res, added_date):
        super().__init__(**asdict(res))
        self.added_date = added_date


class FavoritesTab(Tab):
    """Favorites tab component for managing bookmarked documents"""

    def __init__(self, page: ft.Page, on_navigate_to_search):
        self.page = page
        self.favorites: list[Favorite] = []
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

    def add_favorite(self, result: ResultContainer):
        """Add a document to favorites"""
        # Check if already in favorites
        print("here")
        if any(fav.url == result.url for fav in self.favorites):
            "remove"
            return False

        # Add timestamp
        self.favorites.insert(0, Favorite(result, datetime.now().strftime("%m/%d/%Y")))
        print(self.favorites)
        self.update_favorites_list()
        return True

    def remove_favorite(self, favorite: Favorite):
        """Remove a document from favorites"""
        self.favorites = [fav for fav in self.favorites if fav.url != favorite.url]
        self.update_favorites_list()

    def open_favorite(self, favorite: Favorite):
        """Open a favorite document"""
        self.page.launch_url(favorite.url)



class FavoriteCard(ResultContainer):
    def __init__(self, favorite: Favorite, on_remove_favorite, on_open_favorite):
        delete_button=ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_color=ft.Colors.RED_400,
            tooltip="Remove from favorites",
            on_click=lambda e, fav=favorite: on_remove_favorite(fav)
        )
        super().__init__(
            title=favorite.title,
            text=favorite.snippet,
            source=favorite.source,
            metadata=[f"Added: {favorite.added_date}"],
            button=delete_button,
            on_click=lambda e, fav=favorite: on_open_favorite(fav))
