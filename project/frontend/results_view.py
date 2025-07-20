from dataclasses import dataclass
from typing import List, Optional
import flet as ft
from .components import ResultContainer, EmptyState, ResultTitle
from .tab import TabTitle

@dataclass
class Result:
    url: str
    title: str
    snippet: str
    source: str
    date: str
    pages: str
    example_sentence_scores: Optional[List[float]] = None  # Wichtigkeitswerte für dieses Result


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

    def show_results(self, query, results: list[list[Result]]):
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
            num_results = sum([len(res) for res in results])
            title = TabTitle(f"Search results for '{query}' ({num_results} result{'s' if num_results != 1 else ''})")
            result_column = []
            for result_row in results:
                result_column.append(ft.Container(
                    ft.Column([
                        ResultTitle(f"Results similar to '{result_row[0].title}'"),
                        ft.Row([
                            ResultContainer(
                                title=result.title,
                                text=result.snippet,
                                source=result.source,
                                metadata=[result.date, f"{result.pages} pages"],
                                button=ft.IconButton(
                                    icon=ft.Icons.BOOKMARK_BORDER,
                                    icon_color=ft.Colors.GREY_600,
                                    tooltip="Add to favorites",
                                    on_click=lambda e, r=result: self.on_favorite_toggle(r, True) if self.on_favorite_toggle else None
                                ),
                                on_click=lambda e, r=result: self.on_result_click(r) if self.on_result_click else None,
                                width=300,
                                sentence_scores=result.example_sentence_scores
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