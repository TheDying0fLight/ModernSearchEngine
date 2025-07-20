import flet as ft
from typing import List


class SentenceImportanceHeatmap(ft.Container):
    """
    Compact vertical heatmap bar that visualizes the importance of sentences on a webpage.
    Each sentence is displayed as a horizontal stripe whose color indicates importance.
    """
    
    def __init__(self, 
                 sentence_scores: List[float],
                 width: int = 8,
                 max_height: int = 400,
                 title: str = "Sentence Importance"):
        """
        Args:
            sentence_scores: List of importance values between 0.0 and 1.0
            width: Width of the heatmap bar in pixels
            max_height: Maximum height of the bar
            title: Title of the heatmap
        """
        self.sentence_scores = sentence_scores
        self.heatmap_width = width
        self.max_height = max_height
        self.title = title
        
        # Calculate height per sentence
        self.stripe_height = max(1, min(8, max_height // len(sentence_scores))) if sentence_scores else 4
        self.total_height = len(sentence_scores) * self.stripe_height
        
        # Create the heatmap
        heatmap_content = self._create_heatmap()
        
        # Container with title and heatmap
        super().__init__(
            content=ft.Column([
                ft.Text(
                    title,
                    size=10,
                    color=ft.Colors.GREY_600,
                    text_align=ft.TextAlign.CENTER,
                    width=width + 20
                ),
                ft.Container(height=4),  # Spacer
                heatmap_content,
                ft.Container(height=8),  # Spacer
                self._create_legend()
            ], 
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0),
            width=width + 20,
            padding=ft.padding.all(8),
            border_radius=4,
            bgcolor=ft.Colors.GREY_50
        )
    
    def _create_heatmap(self) -> ft.Container:
        """Creates the actual heatmap from horizontal stripes"""
        stripes = []
        
        for score in self.sentence_scores:
            # Normalize the score
            normalized_score = max(0.0, min(1.0, score))
            
            # Determine color and opacity based on importance
            color, opacity = self._get_importance_color_and_opacity(normalized_score)
            
            # Create a horizontal stripe
            stripe = ft.Container(
                height=self.stripe_height,
                width=self.heatmap_width,
                bgcolor=color,
                opacity=opacity,
                border_radius=1
            )
            stripes.append(stripe)
        
        return ft.Container(
            content=ft.Column(
                controls=stripes,
                spacing=0,
                tight=True
            ),
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=2,
            padding=1
        )
    
    def _get_importance_color_and_opacity(self, score: float) -> tuple[str, float]:
        """
        Converts an importance value to color and opacity.
        
        Args:
            score: Importance value between 0.0 and 1.0
            
        Returns:
            Tuple of (color string, opacity value)
        """
        if score < 0.7:
            # Low importance - gray 
            return ft.Colors.GREY_400, 0.4
        elif score < 0.8:
            # Medium importance - orange  
            return ft.Colors.ORANGE_500, 0.9
        elif score < 0.9:
            # High importance - red-orange
            return ft.Colors.DEEP_ORANGE_600, 1.0
        else:
            # Maximum importance - red
            return ft.Colors.RED_600, 1.0
    
    def _create_legend(self) -> ft.Column:
        """Creates a small legend for color coding"""
        legend_items = [
            (ft.Colors.GREY_400, "Low"),
            (ft.Colors.YELLOW_400, "Med"),
            (ft.Colors.ORANGE_500, "High"),
            (ft.Colors.RED_600, "Max")
        ]
        
        legend_controls = []
        for color, label in legend_items:
            legend_controls.append(
                ft.Row([
                    ft.Container(
                        width=8,
                        height=4,
                        bgcolor=color,
                        border_radius=1
                    ),
                    ft.Text(label, size=8, color=ft.Colors.GREY_600)
                ], 
                spacing=2,
                alignment=ft.MainAxisAlignment.START)
            )
        
        return ft.Column(
            controls=legend_controls,
            spacing=1,
            tight=True
        )
    
    def update_scores(self, new_scores: List[float]):
        """
        Updates the importance values and regenerates the heatmap.
        
        Args:
            new_scores: New list of importance values
        """
        self.sentence_scores = new_scores
        
        # Calculate new dimensions
        self.stripe_height = max(1, min(8, self.max_height // len(new_scores))) if new_scores else 4
        self.total_height = len(new_scores) * self.stripe_height
        
        # Regenerate the heatmap
        heatmap_content = self._create_heatmap()
        
        # Update the content (simplified - in a real application you would only update the heatmap)
        self.content.controls[2] = heatmap_content
        
        # Only update if the control has already been added to the page
        if hasattr(self, 'page') and self.page is not None:
            self.update()


class HorizontalSentenceHeatmap(ft.Container):
    """
    Horizontal heatmap for sentence importance under the snippet
    """
    
    def __init__(self, sentence_scores: List[float], width: int = 280, height: int = 20):
        """
        Args:
            sentence_scores: List of importance values between 0.0 and 1.0
            width: Width of the heatmap (should match ResultCard width minus padding)
            height: Height of the heatmap
        """
        self.sentence_scores = sentence_scores
        
        # Calculate segment width based on available space
        segment_width = max(3, width // len(sentence_scores)) if sentence_scores else 4
        
        # Create horizontal segments
        segments = []
        for i, score in enumerate(sentence_scores):
            normalized_score = max(0.0, min(1.0, score))
            
            # Color mapping consistent with vertical heatmap - only ~20% colored
            if normalized_score < 0.7:
                color = ft.Colors.GREY_200
            elif normalized_score < 0.8:
                color = ft.Colors.ORANGE_400
            elif normalized_score < 0.9:
                color = ft.Colors.DEEP_ORANGE_500
            else:
                color = ft.Colors.RED_500
            
            segments.append(
                ft.Container(
                    width=segment_width,
                    height=height,
                    bgcolor=color,
                    border_radius=ft.border_radius.only(
                        top_left=2 if i == 0 else 0,
                        bottom_left=2 if i == 0 else 0,
                        top_right=2 if i == len(sentence_scores) - 1 else 0,
                        bottom_right=2 if i == len(sentence_scores) - 1 else 0
                    ),
                    tooltip=f"Sentence {i+1}: {score:.2f}"
                )
            )
        
        super().__init__(
            content=ft.Row(
                controls=segments,
                spacing=0.5,
                tight=True
            ),
            width=min(width, len(sentence_scores) * segment_width + len(sentence_scores)),
            height=height,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=4,
            tooltip=f"Sentence Importance Heatmap ({len(sentence_scores)} sentences)",
            padding=ft.padding.all(2)
        )


class CompactSentenceHeatmap(ft.Container):
    """
    Compact version of the heatmap for individual search results
    """
    
    def __init__(self, sentence_scores: List[float], width: int = 6, height: int = 120):
        """
        Args:
            sentence_scores: List of importance values between 0.0 and 1.0
            width: Width of the heatmap
            height: Height of the heatmap
        """
        self.sentence_scores = sentence_scores
        
        # Calculate stripe height based on available space
        stripe_height = max(2, height // len(sentence_scores)) if sentence_scores else 4
        
        # Create stripes
        stripes = []
        for score in sentence_scores:
            normalized_score = max(0.0, min(1.0, score))
            
            # Extended color mapping with more nuances
            if normalized_score < 0.1:
                color = ft.Colors.GREY_200
            elif normalized_score < 0.3:
                color = ft.Colors.BLUE_200
            elif normalized_score < 0.5:
                color = ft.Colors.YELLOW_400
            elif normalized_score < 0.7:
                color = ft.Colors.ORANGE_500
            elif normalized_score < 0.9:
                color = ft.Colors.DEEP_ORANGE_600
            else:
                color = ft.Colors.RED_600
            
            stripes.append(
                ft.Container(
                    height=stripe_height,
                    width=width,
                    bgcolor=color,
                    border_radius=0
                )
            )
        
        super().__init__(
            content=ft.Column(
                controls=stripes,
                spacing=0,
                tight=True
            ),
            width=width,
            height=min(height, len(sentence_scores) * stripe_height),
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=2,
            tooltip=f"Sentence Importance ({len(sentence_scores)} sentences)"
        )
