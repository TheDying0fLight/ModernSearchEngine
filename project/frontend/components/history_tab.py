import flet as ft
from datetime import datetime, timedelta


class HistoryTab:
    """History tab component for managing search history"""
    
    def __init__(self):
        self.history_items = []
        self.max_items = 50
        self.container = self.create_history_content()
    
    def create_history_content(self):
        """Create the history tab content"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Text(
                            "🕒 Search History", 
                            size=24, 
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_800
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CLEAR_ALL,
                            icon_color=ft.Colors.RED_400,
                            tooltip="Clear all history",
                            on_click=self.clear_all_history
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    margin=ft.margin.only(bottom=20)
                ),
                self.create_history_list()
            ], expand=True, scroll=ft.ScrollMode.AUTO),
            padding=20,
            alignment=ft.alignment.top_center,
            expand=True
        )
    
    def create_history_list(self):
        """Create the history list or empty state"""
        if not self.history_items:
            return self.create_empty_state()
        
        # Create list of history items grouped by date
        history_groups = self.group_history_by_date()
        history_column = ft.Column([], spacing=15, expand=True)
        
        for date_group, items in history_groups.items():
            # Date header
            date_header = ft.Container(
                content=ft.Text(
                    date_group,
                    size=16,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_700
                ),
                padding=ft.padding.only(left=10, top=10, bottom=5),
                border_radius=5
            )
            history_column.controls.append(date_header)
            
            # History items for this date
            for item in items:
                history_item = self.create_history_item(item)
                history_column.controls.append(history_item)
        
        return ft.Container(
            content=history_column,
            expand=True
        )
    
    def create_empty_state(self):
        """Create empty state when no history exists"""
        return ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.HISTORY, size=64, color=ft.Colors.GREY_400),
                ft.Text(
                    "No search history yet",
                    size=18,
                    color=ft.Colors.GREY_600,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Text(
                    "Your search history will appear here",
                    size=14,
                    color=ft.Colors.GREY_500,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.ElevatedButton(
                    text="Start Searching",
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
            ),
            expand=True
        )
    
    def create_history_item(self, item):
        """Create a history item card"""
        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.SEARCH, size=20, color=ft.Colors.BLUE_600),
                ft.Column([
                    ft.Text(
                        item['query'],
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.GREY_800
                    ),
                    ft.Text(
                        f"Searched at {item['time']} • {item['results_count']} results",
                        size=12,
                        color=ft.Colors.GREY_600
                    )
                ], expand=True, spacing=2),
                ft.Row([
                    ft.IconButton(
                        icon=ft.Icons.SEARCH,
                        icon_color=ft.Colors.BLUE_600,
                        tooltip="Search again",
                        on_click=lambda e, query=item['query']: self.search_again(query)
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_color=ft.Colors.RED_400,
                        tooltip="Remove from history",
                        on_click=lambda e, item=item: self.remove_history_item(item)
                    )
                ])
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=15,
            margin=ft.margin.only(bottom=10),
            bgcolor=ft.Colors.WHITE,
            border_radius=8,
            border=ft.border.all(1, ft.Colors.GREY_200),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=2,
                color=ft.Colors.GREY_100,
                offset=ft.Offset(0, 1)
            )
        )
    
    def group_history_by_date(self):
        """Group history items by date"""
        groups = {}
        for item in self.history_items:
            date_key = item['date']
            if date_key not in groups:
                groups[date_key] = []
            groups[date_key].append(item)
        return groups
    
    def add_to_history(self, query, results_count=0):
        """Add a search query to history"""
        if not query.strip():
            return
        
        now = datetime.now()
        timestamp = now.strftime("%H:%M")
        date_str = now.strftime("%B %d, %Y")
        today = datetime.now().date()
        yesterday = (datetime.now() - timedelta(days=1)).date()
        
        # Format date nicely
        if now.date() == today:
            date_key = "Today"
        elif now.date() == yesterday:
            date_key = "Yesterday"
        else:
            date_key = date_str
        
        # Remove duplicate if exists
        self.history_items = [
            item for item in self.history_items 
            if item['query'].lower() != query.lower()
        ]
        
        # Add new item at the beginning
        self.history_items.insert(0, {
            'query': query,
            'time': timestamp,
            'date': date_key,
            'results_count': results_count,
            'timestamp': now
        })
        
        # Keep only max_items
        if len(self.history_items) > self.max_items:
            self.history_items = self.history_items[:self.max_items]
        
        self.refresh_display()
    
    def remove_history_item(self, item):
        """Remove a specific history item"""
        self.history_items = [h for h in self.history_items if h != item]
        self.refresh_display()
    
    def clear_all_history(self, e):
        """Clear all search history"""
        self.history_items.clear()
        self.refresh_display()
    
    def search_again(self, query):
        """Search again with the same query"""
        if self.on_search_callback:
            self.on_search_callback(query)
    
    def navigate_to_search(self, e):
        """Navigate to search tab"""
        if self.on_navigate_callback:
            self.on_navigate_callback(0)  # Search tab index
    
    def refresh_display(self):
        """Refresh the history display"""
        # Update the content
        self.container.content.controls[1] = self.create_history_list()
        self.container.update()
    
    def set_callbacks(self, on_search=None, on_navigate=None):
        """Set callback functions"""
        self.on_search_callback = on_search
        self.on_navigate_callback = on_navigate
    
    def get_container(self):
        """Get the history tab container"""
        return self.container
    
    def get_history_count(self):
        """Get the number of history items"""
        return len(self.history_items)
    
    def get_recent_searches(self, count=5):
        """Get recent searches"""
        return [item['query'] for item in self.history_items[:count]]
