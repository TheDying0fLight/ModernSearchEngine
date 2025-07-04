import flet as ft


class LoadingIndicator:
    """Loading indicator component for search operations"""
    
    def __init__(self, message="Searching..."):
        self.message = message
        self.container = self.create_loading_indicator()
    
    def create_loading_indicator(self):
        """Create the loading indicator"""
        return ft.Container(
            content=ft.Row([
                ft.ProgressRing(width=20, height=20, color=ft.Colors.BLUE_600),
                ft.Text(
                    self.message, 
                    color=ft.Colors.BLUE_600,
                    size=16
                )
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            visible=False,
            margin=ft.margin.only(top=20),
            padding=20
        )
    
    def show(self, message=None):
        """Show the loading indicator"""
        if message:
            self.message = message
            # Update the text in the indicator
            self.container.content.controls[1].value = message
        self.container.visible = True
        self.container.update()
    
    def hide(self):
        """Hide the loading indicator"""
        self.container.visible = False
        self.container.update()
    
    def set_message(self, message):
        """Update the loading message"""
        self.message = message
        self.container.content.controls[1].value = message
        self.container.update()
    
    def get_container(self):
        """Get the loading indicator container"""
        return self.container
