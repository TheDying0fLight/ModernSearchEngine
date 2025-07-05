import flet as ft


class LoadingIndicator(ft.Container):
    """Loading indicator component for search operations"""

    def __init__(self, message="Searching..."):
        self.message = message
        super().__init__(
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
            padding=20)

    def show(self, message=None):
        """Show the loading indicator"""
        if message:
            self.message = message
            # Update the text in the indicator
            self.content.controls[1].value = message
        self.visible = True
        self.update()

    def hide(self):
        """Hide the loading indicator"""
        self.visible = False
        self.update()

    def set_message(self, message):
        """Update the loading message"""
        self.message = message
        self.content.controls[1].value = message
        self.update()
