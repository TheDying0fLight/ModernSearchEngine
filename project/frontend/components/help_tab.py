import flet as ft


class HelpTab:
    """Help tab component with team information and usage instructions"""
    
    def __init__(self):
        self.container = self.create_help_content()
    
    def create_help_content(self):
        """Create the help tab content"""
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Text(
                        "❓ Help", 
                        size=24, 
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_800
                    ),
                    margin=ft.margin.only(bottom=20)
                ),
                
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "About Tübingen Search Engine",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        ft.Text(
                            "This search engine helps you find information about Tübingen's attractions, food, drinks, and local culture.",
                            size=16,
                            color=ft.Colors.GREY_700,
                            text_align=ft.TextAlign.CENTER
                        ),
                        
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        
                        ft.Text(
                            "How to Use:",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        ft.Column([
                            ft.Text("• Enter search terms in the search bar", size=14, color=ft.Colors.GREY_600),
                            ft.Text("• Use the suggestions for popular searches", size=14, color=ft.Colors.GREY_600),
                            ft.Text("• View your search history in the History tab", size=14, color=ft.Colors.GREY_600),
                            ft.Text("• Save pages to favorites for quick access", size=14, color=ft.Colors.GREY_600),
                        ], spacing=5),
                        
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        
                        ft.Text(
                            "Project Team:",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        ft.Text(
                            "Jan-Malte Giannikos • Simon Döhl • Carina Straub\nMartin Eichler • Kilian Hunter",
                            size=14,
                            color=ft.Colors.GREY_600,
                            text_align=ft.TextAlign.CENTER
                        ),
                        
                        ft.Divider(height=20, color=ft.Colors.GREY_300),
                        
                        ft.Text(
                            "Modern Search Engines Course",
                            size=14,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        ft.Text(
                            "University of Tübingen • 2025",
                            size=12,
                            color=ft.Colors.GREY_500,
                            text_align=ft.TextAlign.CENTER
                        )
                    ], spacing=10),
                    padding=30,
                    bgcolor=ft.Colors.WHITE,
                    border_radius=10,
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=4,
                        color=ft.Colors.GREY_200,
                        offset=ft.Offset(0, 2)
                    )
                )
            ], scroll=ft.ScrollMode.AUTO),
            padding=20,
            alignment=ft.alignment.top_center,
            expand=True
        )
    
    def get_container(self):
        """Get the help tab container"""
        return self.container
