from dataclasses import dataclass
import flet as ft
from .components import ResultContainer, EmptyState
from .tab import TabTitle

@dataclass
class Result:
    url: str
    title: str
    snippet: str
    source: str
    date: str
    pages: str


class ResultCard(ResultContainer):
    """Individual result card component"""

    def __init__(self, result_data: Result, on_click_callback=None, on_favorite_callback=None):
        self.result_data = result_data
        self.on_click_callback = on_click_callback
        self.on_favorite_callback = on_favorite_callback
        self.is_favorited = False
        self.favorite_button = ft.IconButton(
            icon=ft.Icons.BOOKMARK_BORDER if not self.is_favorited else ft.Icons.BOOKMARK,
            icon_color=ft.Colors.GREY_600 if not self.is_favorited else ft.Colors.ORANGE_600,
            tooltip="Add to favorites" if not self.is_favorited else "Remove from favorites",
            on_click=self.toggle_favorite
        )
        super().__init__(
            title=self.result_data.title,
            text=self.result_data.snippet,
            source=self.result_data.source,
            metadata=[self.result_data.date, f"{self.result_data.pages} pages"],
            button=self.favorite_button,
            on_click=lambda e, result_data=result_data: on_click_callback(result_data)
        )

    def toggle_favorite(self, e):
        """Toggle favorite status"""
        self.is_favorited = not self.is_favorited
        # Update the icon
        self.favorite_button.icon = ft.Icons.BOOKMARK if self.is_favorited else ft.Icons.BOOKMARK_BORDER
        self.favorite_button.icon_color = ft.Colors.ORANGE_600 if self.is_favorited else ft.Colors.GREY_600
        self.favorite_button.tooltip = "Remove from favorites" if self.is_favorited else "Add to favorites"
        # Call callback if provided
        if self.on_favorite_callback:
            self.on_favorite_callback(self.result_data, self.is_favorited)
        self.update()


class ResultsView(ft.Container):
    """Results view component for displaying search results"""

    def __init__(self, on_favorite_toggle=None, on_result_click=None):
        self.on_favorite_toggle = on_favorite_toggle
        self.on_result_click = on_result_click
        super().__init__(
            content= ft.Column([]),
            margin=ft.margin.only(top=20),
            visible=False
        )

    def show_results(self, query, results: list[Result]):
        if (not results) or len(results) == 0:
            self.content = results_component = EmptyState(
                icon=ft.Icons.SEARCH_OFF,
                title="No results found",
                text=f"No documents found for '{query}' \n Try different search terms or check your spelling",
                button_icon=ft.Icons.SEARCH,
                button_text="Try Searching again",
                on_button_click= lambda e: None,
            )
        else:
            title = TabTitle(f"Search results for '{query}' ({len(results)} result{'s' if len(results) != 1 else ''})")
            results_component = ft.Column([
                ResultCard(
                    result_data=result,
                    on_click_callback=self.on_result_click,
                    on_favorite_callback=self.on_favorite_toggle
                ) for result in results],
                scroll=ft.ScrollMode.AUTO,
                spacing=0)
            self.content = ft.Column([title, results_component])
        self.visible = True
        self.update()

    def hide_results(self):
        self.visible = False
        self.update()