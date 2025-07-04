import flet as ft
from datetime import datetime


class HistoryItem:
    """Individual history item component"""
    
    def __init__(self, query, timestamp, on_click_callback):
        self.query = query
        self.timestamp = timestamp
        self.on_click_callback = on_click_callback
        self.container = self.create_history_item()
    
    def create_history_item(self):
        """Create a history item container"""
        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.HISTORY, size=16, color=ft.Colors.GREY_600),
                ft.Text(
                    self.query, 
                    expand=True,
                    size=14,
                    color=ft.Colors.GREY_800
                ),
                ft.Text(
                    self.timestamp, 
                    color=ft.Colors.GREY_500, 
                    size=12
                )
            ], spacing=10),
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            border_radius=5,
            bgcolor=ft.Colors.GREY_100,
            margin=ft.margin.only(bottom=5),
            on_click=lambda e: self.on_click_callback(self.query),
            tooltip=f"Search again for '{self.query}'"
        )
    
    def get_container(self):
        """Get the history item container"""
        return self.container


class HistoryContainer:
    """History container component for managing search history"""
    
    def __init__(self, on_history_click):
        self.on_history_click = on_history_click
        self.history_items = []
        self.max_items = 10
        self.container = self.create_history_container()
    
    def create_history_container(self):
        """Create the history container"""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(
                        "Search History", 
                        size=18, 
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_800
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CLEAR_ALL,
                        icon_color=ft.Colors.GREY_600,
                        tooltip="Clear all history",
                        on_click=self.clear_history
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(height=10, color=ft.Colors.GREY_300),
                ft.Column(
                    [], 
                    scroll=ft.ScrollMode.AUTO, 
                    height=200,
                    spacing=0
                )
            ], spacing=5),
            margin=ft.margin.only(top=30),
            padding=20,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            visible=False,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=4,
                color=ft.Colors.GREY_200,
                offset=ft.Offset(0, 2)
            )
        )
    
    def add_to_history(self, query):
        """Add a search query to history"""
        if not query.strip():
            return
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Remove duplicate if exists
        self.history_items = [
            item for item in self.history_items 
            if item['query'].lower() != query.lower()
        ]
        
        self.history_items.insert(0, {
            'query': query,
            'timestamp': timestamp
        })
        
        # Keep only max_items
        if len(self.history_items) > self.max_items:
            self.history_items = self.history_items[:self.max_items]
        
        self.update_display()
    
    def update_display(self):
        """Update the history display"""
        history_column = self.container.content.controls[2]
        history_column.controls.clear()
        
        if not self.history_items:
            # Show empty state
            empty_state = ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.HISTORY, size=32, color=ft.Colors.GREY_400),
                    ft.Text(
                        "No search history yet",
                        color=ft.Colors.GREY_500,
                        size=14
                    )
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                padding=30,
                alignment=ft.alignment.center
            )
            history_column.controls.append(empty_state)
        else:
            # Add history items
            for item in self.history_items:
                history_item = HistoryItem(
                    query=item['query'],
                    timestamp=item['timestamp'],
                    on_click_callback=self.on_history_click
                )
                history_column.controls.append(history_item.get_container())
        
        # Show/hide container based on whether there are items
        self.container.visible = len(self.history_items) > 0
        self.container.update()
    
    def clear_history(self, e):
        """Clear all search history"""
        self.history_items.clear()
        self.update_display()
    
    def get_container(self):
        """Get the history container"""
        return self.container
    
    def get_history_count(self):
        """Get the number of history items"""
        return len(self.history_items)
    
    def get_recent_searches(self, count=5):
        """Get recent searches"""
        return [item['query'] for item in self.history_items[:count]]
