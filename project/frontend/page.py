import flet as ft
import time
from datetime import datetime
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
import urllib.parse

from .tab_help import HelpTab
from .tab_history import HistoryTab
from .tab_search import SearchTab
from .tab_favorites import FavoritesTab
from .results_view import Result

from project import SearchEngine, Document

class SearchEnginePage:
    """Main page of the search engine with component-based architecture"""

    def __init__(self, page: ft.Page, search_engine: SearchEngine):
        self.page = page
        self.search_engine = search_engine
        self.page.on_route_change = self.route_change
        self.is_searching = False
        self.current_results = []
        self.possible_clustering_algos = {
            'Affinity Propagation': AffinityPropagation(),
            'Agglomerative Clustering': AgglomerativeClustering(),
            'KMeans': KMeans()
        }

        # Page setup with modern styling
        page.title = "Tübingen Search Engine"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.bgcolor = ft.Colors.GREY_50
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.padding = 0  # Remove padding for full width tabs

        # Initialize tab components
        self.help_tab = HelpTab()
        self.favorites_tab = FavoritesTab(page=self.page, on_navigate_to_search=self.navigate_to_search_tab)
        self.search_tab = SearchTab(page=self.page, clustering_options=list(self.possible_clustering_algos.keys()), on_favorite_toggle=self.handle_favorite_toggle)
        self.history_tab = HistoryTab(page=self.page, on_navigate_to_search=self.navigate_to_search_tab)

        # Build the layout
        self.build_layout()

    def build_layout(self):
        """Build the main page layout with tabs"""
        # Create tabs using the component containers
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tab_alignment=ft.TabAlignment.CENTER,
            expand=1,
            tabs=[
                self.search_tab,
                self.history_tab,
                self.favorites_tab,
                self.help_tab
            ]
        )
        self.page.add(self.tabs)

    def route_change(self, route):
        template_route = ft.TemplateRoute(self.page.route)
        if template_route.match('/search?:q&:c'):
            query = urllib.parse.unquote_plus(template_route.q.split('=')[1])
            cluster_option = urllib.parse.unquote_plus(template_route.c.split('=')[1])
            if not query.strip() or cluster_option not in self.possible_clustering_algos:
                self.page.go('/')
                return
            self.navigate_to_search_tab()
            self.search_tab.start_loading(query)
            self.search(query, cluster_option)
        else:
            self.page.go('/')

    def navigate_to_search_tab(self):
        """Navigate to the search tab"""
        self.tabs.selected_index = 0
        self.tabs.update()

    def search(self, query: str, cluster_option: str):
        """Enhanced search function with loading state"""
        clustering_algo = self.possible_clustering_algos[cluster_option]
        time1 = time.time()
        results, sentence_wise_similarities = self.search_engine.search_and_cluster(query, clustering_algo)
        results = [[self.convert_doc(res, sentence_wise_similarities) for res in topic] for topic in results]
        time2 = time.time()
        self.search_tab.display_results(query, results, time=time2-time1)
        self.history_tab.add_to_history(query)

    def search_from_history(self, query):
        """Search triggered from history tab"""
        # Navigate to search tab and perform search
        self.tabs.selected_index = 0
        self.tabs.update()
        self.search(query)

    def handle_favorite_toggle(self, result_data: Result, is_favorited):
        """Handle favorite toggle from search results"""
        if is_favorited:
            return self.favorites_tab.add_favorite(result_data)
        else:
            self.favorites_tab.remove_favorite(result_data)
            return True

    def convert_doc(self, doc: Document, sentence_wise_similarities: dict[str, list[float]]):
        return Result(
            url=doc.url,
            title=doc.title.strip('\n'),
            snippet=doc.description.strip("\n"), #doc['snippet'],
            source=doc.domain,
            date=datetime.fromtimestamp(doc.last_crawl_timestamp),
            words=doc.word_count,
            sentence_scores=sentence_wise_similarities[doc.url] if type(sentence_wise_similarities[doc.url]) is list else [sentence_wise_similarities[doc.url]]
        )


class PageFactory():
    """Creates new pages of type `SearchEnginePage`"""
    def __init__(self):
        self.search_engine = SearchEngine("data")

    def create_page(self, page: ft.Page):
        search_engine_page = SearchEnginePage(page, self.search_engine)
        def on_connect(e):
            page.clean()
            search_engine_page.build_layout()
        page.on_connect = on_connect

    def run(self, host: str, port: int):
        ft.app(self.create_page, view=ft.AppView.WEB_BROWSER, host=host, port=port)
