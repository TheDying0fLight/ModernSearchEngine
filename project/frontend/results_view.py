from dataclasses import dataclass
from typing import List, Optional
import flet as ft
import time
from .components import ResultContainer, EmptyState, ResultTitle
from .tab import TabTitle

@dataclass
class Result:
    url: str
    title: str
    snippet: str
    source: str
    date: str
    words: str
    sentence_scores:list[float]


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
            metadata=[self.result_data.date, f"{self.result_data.words} words"],
            button=self.favorite_button,
            on_click=lambda e, result_data=result_data: on_click_callback(result_data) if on_click_callback else None,
            width=300,
            max_hight=400,
            sentence_scores=self.result_data.sentence_scores
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

    def show_results(self, query, results: list[list[Result]], time):
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
            title = TabTitle(f"Best 100 results for '{query}' ({round(time, 2)}s)")
            result_column = []
            for result_row in results:
                result_column.append(ft.Container(
                    ft.Column([
                        ResultTitle(f"Results similar to '{result_row[0].title}'"),
                        ft.Row([
                            ResultCard(
                                result_data=result,
                                on_click_callback=self.on_result_click,
                                on_favorite_callback=self.on_favorite_toggle
                            ) for result in result_row],
                            scroll=ft.ScrollMode.ALWAYS,
                            spacing=10,
                            expand=True)
                        ], spacing=10),
                    padding=20)
                )
            results_component = ft.Column(result_column, scroll=ft.ScrollMode.ALWAYS, spacing=20)
            self.content = ft.Column([title, results_component])
        self.visible = True
        self.update()

    def hide_results(self):
        self.visible = False
        self.update()