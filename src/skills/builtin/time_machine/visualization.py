"""
Visualization Engine for Context Time Machine
Author: Drmusab
Last Modified: 2025-01-08

Creates visual representations of behavioral trends and patterns:
- Time series charts for behavioral metrics
- Trend visualization
- Comparison charts
- Progress indicators
- Exportable chart formats
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import Language

from .reflection_analyzer import BehavioralAnalysis, BehavioralTrend, BehavioralMetric, TrendDirection


class ChartType(Enum):
    """Types of charts that can be generated."""
    LINE_CHART = "line_chart"          # Time series line chart
    BAR_CHART = "bar_chart"            # Bar chart for comparisons
    AREA_CHART = "area_chart"          # Area chart for trends
    SCATTER_PLOT = "scatter_plot"      # Scatter plot for correlations
    HEATMAP = "heatmap"               # Heatmap for patterns
    RADAR_CHART = "radar_chart"        # Radar chart for multi-metric view
    PROGRESS_BAR = "progress_bar"      # Progress indicator


class ExportFormat(Enum):
    """Export formats for visualizations."""
    JSON = "json"                      # Chart.js compatible JSON
    CSV = "csv"                        # CSV data
    SVG = "svg"                        # SVG format (simplified)
    ASCII = "ascii"                    # ASCII art charts


@dataclass
class ChartDataPoint:
    """A single data point in a chart."""
    x: Any  # X-axis value (usually timestamp)
    y: float  # Y-axis value
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartDataset:
    """A dataset for chart visualization."""
    label: str
    data: List[ChartDataPoint]
    color: str = "#3498db"
    type: ChartType = ChartType.LINE_CHART
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartConfiguration:
    """Configuration for chart generation."""
    title: str
    subtitle: Optional[str] = None
    x_axis_label: str = ""
    y_axis_label: str = ""
    chart_type: ChartType = ChartType.LINE_CHART
    width: int = 800
    height: int = 400
    show_legend: bool = True
    show_grid: bool = True
    language: Language = Language.ARABIC


@dataclass
class VisualizationResult:
    """Result of visualization generation."""
    chart_data: Dict[str, Any]
    export_format: ExportFormat
    chart_config: ChartConfiguration
    datasets: List[ChartDataset]
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualizationEngine:
    """
    Creates visual representations of behavioral analysis data.
    
    Generates charts and visualizations to help users understand
    their behavioral patterns and trends over time.
    """
    
    def __init__(self, container: Container):
        """Initialize the visualization engine."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Chart colors for different metrics
        self.metric_colors = {
            BehavioralMetric.TONE: "#e74c3c",
            BehavioralMetric.MOOD: "#f39c12",
            BehavioralMetric.ENERGY: "#e67e22",
            BehavioralMetric.CONFIDENCE: "#27ae60",
            BehavioralMetric.ENGAGEMENT: "#3498db",
            BehavioralMetric.FORMALITY: "#9b59b6",
            BehavioralMetric.COMPLEXITY: "#34495e",
            BehavioralMetric.RESPONSIVENESS: "#1abc9c"
        }
        
        # Arabic labels for metrics
        self.arabic_metric_labels = {
            BehavioralMetric.TONE: "النبرة",
            BehavioralMetric.MOOD: "المزاج",
            BehavioralMetric.ENERGY: "الطاقة",
            BehavioralMetric.CONFIDENCE: "الثقة",
            BehavioralMetric.ENGAGEMENT: "التفاعل",
            BehavioralMetric.FORMALITY: "الرسمية",
            BehavioralMetric.COMPLEXITY: "التعقيد",
            BehavioralMetric.RESPONSIVENESS: "الاستجابة"
        }
        
        self.logger.info("VisualizationEngine initialized")
    
    async def create_behavioral_trends_chart(
        self,
        analysis: BehavioralAnalysis,
        chart_type: ChartType = ChartType.LINE_CHART,
        export_format: ExportFormat = ExportFormat.JSON
    ) -> VisualizationResult:
        """
        Create a chart showing behavioral trends over time.
        
        Args:
            analysis: Behavioral analysis results
            chart_type: Type of chart to create
            export_format: Export format for the chart
            
        Returns:
            VisualizationResult with chart data
        """
        try:
            # Create datasets for each behavioral metric
            datasets = []
            
            for trend in analysis.trends:
                dataset = await self._create_trend_dataset(trend, analysis.language)
                datasets.append(dataset)
            
            # Configure chart
            chart_config = ChartConfiguration(
                title="الاتجاهات السلوكية" if analysis.language == Language.ARABIC else "Behavioral Trends",
                subtitle=f"من {analysis.analysis_period[0].strftime('%Y-%m-%d')} إلى {analysis.analysis_period[1].strftime('%Y-%m-%d')}",
                x_axis_label="الوقت" if analysis.language == Language.ARABIC else "Time",
                y_axis_label="النتيجة" if analysis.language == Language.ARABIC else "Score",
                chart_type=chart_type,
                language=analysis.language
            )
            
            # Generate chart data based on export format
            if export_format == ExportFormat.JSON:
                chart_data = await self._generate_chartjs_config(datasets, chart_config)
            elif export_format == ExportFormat.CSV:
                chart_data = await self._generate_csv_data(datasets)
            elif export_format == ExportFormat.ASCII:
                chart_data = await self._generate_ascii_chart(datasets, chart_config)
            else:
                chart_data = {"error": f"Unsupported export format: {export_format.value}"}
            
            # Generate insights
            insights = await self._generate_chart_insights(analysis, analysis.language)
            
            result = VisualizationResult(
                chart_data=chart_data,
                export_format=export_format,
                chart_config=chart_config,
                datasets=datasets,
                insights=insights,
                metadata={
                    "metrics_count": len(datasets),
                    "data_points_total": sum(len(ds.data) for ds in datasets),
                    "confidence_score": analysis.confidence_score
                }
            )
            
            self.logger.debug(f"Created behavioral trends chart with {len(datasets)} datasets")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create behavioral trends chart: {str(e)}")
            return self._create_error_visualization(str(e), export_format)
    
    async def create_metric_comparison_chart(
        self,
        analysis: BehavioralAnalysis,
        export_format: ExportFormat = ExportFormat.JSON
    ) -> VisualizationResult:
        """Create a chart comparing different behavioral metrics."""
        try:
            # Create bar chart dataset with average scores for each metric
            data_points = []
            labels = []
            colors = []
            
            for trend in analysis.trends:
                if trend.data_points:
                    avg_score = sum(dp.value for dp in trend.data_points) / len(trend.data_points)
                    label = self._get_metric_label(trend.metric, analysis.language)
                    
                    data_points.append(ChartDataPoint(
                        x=label,
                        y=avg_score,
                        metadata={"trend_direction": trend.direction.value}
                    ))
                    labels.append(label)
                    colors.append(self.metric_colors.get(trend.metric, "#95a5a6"))
            
            dataset = ChartDataset(
                label="متوسط النتائج" if analysis.language == Language.ARABIC else "Average Scores",
                data=data_points,
                color=colors[0] if colors else "#3498db",
                type=ChartType.BAR_CHART,
                metadata={"colors": colors}
            )
            
            datasets = [dataset]
            
            # Configure chart
            chart_config = ChartConfiguration(
                title="مقارنة المؤشرات السلوكية" if analysis.language == Language.ARABIC else "Behavioral Metrics Comparison",
                x_axis_label="المؤشرات" if analysis.language == Language.ARABIC else "Metrics",
                y_axis_label="متوسط النتيجة" if analysis.language == Language.ARABIC else "Average Score",
                chart_type=ChartType.BAR_CHART,
                language=analysis.language
            )
            
            # Generate chart data
            if export_format == ExportFormat.JSON:
                chart_data = await self._generate_chartjs_config(datasets, chart_config)
            elif export_format == ExportFormat.CSV:
                chart_data = await self._generate_csv_data(datasets)
            elif export_format == ExportFormat.ASCII:
                chart_data = await self._generate_ascii_chart(datasets, chart_config)
            else:
                chart_data = {"error": f"Unsupported export format: {export_format.value}"}
            
            # Generate insights
            insights = []
            if analysis.language == Language.ARABIC:
                insights.append(f"تم مقارنة {len(datasets[0].data)} مؤشرات سلوكية")
            else:
                insights.append(f"Compared {len(datasets[0].data)} behavioral metrics")
            
            result = VisualizationResult(
                chart_data=chart_data,
                export_format=export_format,
                chart_config=chart_config,
                datasets=datasets,
                insights=insights
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create metric comparison chart: {str(e)}")
            return self._create_error_visualization(str(e), export_format)
    
    async def create_progress_visualization(
        self,
        analysis: BehavioralAnalysis,
        export_format: ExportFormat = ExportFormat.JSON
    ) -> VisualizationResult:
        """Create a progress visualization showing improvements."""
        try:
            datasets = []
            
            for trend in analysis.trends:
                if trend.direction == TrendDirection.IMPROVING:
                    # Create progress bar for improving metrics
                    progress_value = min(1.0, trend.change_magnitude * 2)  # Scale to 0-1
                    
                    dataset = ChartDataset(
                        label=self._get_metric_label(trend.metric, analysis.language),
                        data=[ChartDataPoint(x=0, y=progress_value)],
                        color=self.metric_colors.get(trend.metric, "#27ae60"),
                        type=ChartType.PROGRESS_BAR,
                        metadata={
                            "change_magnitude": trend.change_magnitude,
                            "confidence": trend.confidence
                        }
                    )
                    datasets.append(dataset)
            
            # Configure chart
            chart_config = ChartConfiguration(
                title="التقدم في المؤشرات السلوكية" if analysis.language == Language.ARABIC else "Behavioral Progress",
                y_axis_label="التحسن" if analysis.language == Language.ARABIC else "Improvement",
                chart_type=ChartType.PROGRESS_BAR,
                language=analysis.language
            )
            
            # Generate chart data
            if export_format == ExportFormat.JSON:
                chart_data = await self._generate_progress_chart_data(datasets, chart_config)
            elif export_format == ExportFormat.ASCII:
                chart_data = await self._generate_ascii_progress_bars(datasets, chart_config)
            else:
                chart_data = {"error": f"Unsupported export format for progress chart: {export_format.value}"}
            
            # Generate insights
            insights = []
            improving_count = len(datasets)
            if analysis.language == Language.ARABIC:
                insights.append(f"يتحسن {improving_count} مؤشر سلوكي")
            else:
                insights.append(f"{improving_count} behavioral metrics are improving")
            
            result = VisualizationResult(
                chart_data=chart_data,
                export_format=export_format,
                chart_config=chart_config,
                datasets=datasets,
                insights=insights
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create progress visualization: {str(e)}")
            return self._create_error_visualization(str(e), export_format)
    
    async def _create_trend_dataset(self, trend: BehavioralTrend, language: Language) -> ChartDataset:
        """Create a dataset for a behavioral trend."""
        data_points = []
        
        for dp in trend.data_points:
            data_points.append(ChartDataPoint(
                x=dp.timestamp.isoformat(),
                y=dp.value,
                metadata={
                    "confidence": dp.confidence,
                    "context": dp.context
                }
            ))
        
        label = self._get_metric_label(trend.metric, language)
        color = self.metric_colors.get(trend.metric, "#95a5a6")
        
        return ChartDataset(
            label=label,
            data=data_points,
            color=color,
            type=ChartType.LINE_CHART,
            metadata={
                "trend_direction": trend.direction.value,
                "change_magnitude": trend.change_magnitude,
                "confidence": trend.confidence
            }
        )
    
    def _get_metric_label(self, metric: BehavioralMetric, language: Language) -> str:
        """Get localized label for a behavioral metric."""
        if language == Language.ARABIC:
            return self.arabic_metric_labels.get(metric, metric.value)
        else:
            return metric.value.replace("_", " ").title()
    
    async def _generate_chartjs_config(
        self,
        datasets: List[ChartDataset],
        config: ChartConfiguration
    ) -> Dict[str, Any]:
        """Generate Chart.js compatible configuration."""
        chart_datasets = []
        
        for dataset in datasets:
            if config.chart_type == ChartType.LINE_CHART:
                chart_dataset = {
                    "label": dataset.label,
                    "data": [{"x": dp.x, "y": dp.y} for dp in dataset.data],
                    "borderColor": dataset.color,
                    "backgroundColor": dataset.color + "20",  # Add transparency
                    "fill": False,
                    "tension": 0.1
                }
            elif config.chart_type == ChartType.BAR_CHART:
                colors = dataset.metadata.get("colors", [dataset.color] * len(dataset.data))
                chart_dataset = {
                    "label": dataset.label,
                    "data": [dp.y for dp in dataset.data],
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1
                }
            else:
                chart_dataset = {
                    "label": dataset.label,
                    "data": [{"x": dp.x, "y": dp.y} for dp in dataset.data],
                    "backgroundColor": dataset.color
                }
            
            chart_datasets.append(chart_dataset)
        
        # Extract labels for bar charts
        labels = []
        if config.chart_type == ChartType.BAR_CHART and datasets:
            labels = [dp.x for dp in datasets[0].data]
        
        chart_config = {
            "type": "line" if config.chart_type == ChartType.LINE_CHART else "bar",
            "data": {
                "labels": labels,
                "datasets": chart_datasets
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": config.title
                    },
                    "legend": {
                        "display": config.show_legend
                    }
                },
                "scales": {
                    "x": {
                        "display": True,
                        "title": {
                            "display": True,
                            "text": config.x_axis_label
                        }
                    },
                    "y": {
                        "display": True,
                        "title": {
                            "display": True,
                            "text": config.y_axis_label
                        },
                        "min": 0,
                        "max": 1
                    }
                }
            }
        }
        
        return chart_config
    
    async def _generate_csv_data(self, datasets: List[ChartDataset]) -> Dict[str, Any]:
        """Generate CSV-formatted data."""
        csv_data = {
            "headers": ["Timestamp", "Metric", "Value", "Confidence"],
            "rows": []
        }
        
        for dataset in datasets:
            for dp in dataset.data:
                row = [
                    dp.x,
                    dataset.label,
                    dp.y,
                    dp.metadata.get("confidence", 1.0)
                ]
                csv_data["rows"].append(row)
        
        return csv_data
    
    async def _generate_ascii_chart(
        self,
        datasets: List[ChartDataset],
        config: ChartConfiguration
    ) -> Dict[str, Any]:
        """Generate ASCII art chart."""
        if not datasets or not datasets[0].data:
            return {"ascii": "No data available"}
        
        # Simple ASCII line chart
        width = 60
        height = 20
        
        # Get data range
        all_values = []
        for dataset in datasets:
            all_values.extend([dp.y for dp in dataset.data])
        
        if not all_values:
            return {"ascii": "No data to display"}
        
        min_val = min(all_values)
        max_val = max(all_values)
        value_range = max_val - min_val if max_val != min_val else 1
        
        # Create ASCII chart
        lines = []
        lines.append(f"{'=' * width}")
        lines.append(f"{config.title:^{width}}")
        lines.append(f"{'=' * width}")
        
        # For each dataset, create a simple representation
        for dataset in datasets:
            lines.append(f"\n{dataset.label}:")
            
            if dataset.data:
                # Create a simple bar representation
                for i, dp in enumerate(dataset.data[:10]):  # Limit to 10 points
                    normalized_val = (dp.y - min_val) / value_range
                    bar_length = int(normalized_val * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    lines.append(f"  {i+1:2d}: {bar} {dp.y:.2f}")
        
        lines.append(f"{'=' * width}")
        
        return {"ascii": "\n".join(lines)}
    
    async def _generate_progress_chart_data(
        self,
        datasets: List[ChartDataset],
        config: ChartConfiguration
    ) -> Dict[str, Any]:
        """Generate progress chart data."""
        progress_data = {
            "type": "progress",
            "title": config.title,
            "metrics": []
        }
        
        for dataset in datasets:
            if dataset.data:
                progress_value = dataset.data[0].y
                progress_data["metrics"].append({
                    "label": dataset.label,
                    "value": progress_value,
                    "percentage": progress_value * 100,
                    "color": dataset.color,
                    "change_magnitude": dataset.metadata.get("change_magnitude", 0),
                    "confidence": dataset.metadata.get("confidence", 0)
                })
        
        return progress_data
    
    async def _generate_ascii_progress_bars(
        self,
        datasets: List[ChartDataset],
        config: ChartConfiguration
    ) -> Dict[str, Any]:
        """Generate ASCII progress bars."""
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"{config.title:^60}")
        lines.append(f"{'=' * 60}")
        
        for dataset in datasets:
            if dataset.data:
                progress_value = dataset.data[0].y
                bar_length = int(progress_value * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                percentage = progress_value * 100
                
                lines.append(f"{dataset.label}:")
                lines.append(f"  {bar} {percentage:.1f}%")
                lines.append("")
        
        lines.append(f"{'=' * 60}")
        
        return {"ascii": "\n".join(lines)}
    
    async def _generate_chart_insights(
        self,
        analysis: BehavioralAnalysis,
        language: Language
    ) -> List[str]:
        """Generate insights from chart data."""
        insights = []
        
        if language == Language.ARABIC:
            improving_trends = [t for t in analysis.trends if t.direction == TrendDirection.IMPROVING]
            declining_trends = [t for t in analysis.trends if t.direction == TrendDirection.DECLINING]
            
            if improving_trends:
                insights.append(f"{len(improving_trends)} مؤشرات في تحسن")
            
            if declining_trends:
                insights.append(f"{len(declining_trends)} مؤشرات في تراجع")
            
            if analysis.confidence_score > 0.7:
                insights.append("النتائج موثوقة")
            elif analysis.confidence_score > 0.4:
                insights.append("النتائج متوسطة الموثوقية")
            else:
                insights.append("النتائج تحتاج المزيد من البيانات")
        else:
            improving_trends = [t for t in analysis.trends if t.direction == TrendDirection.IMPROVING]
            declining_trends = [t for t in analysis.trends if t.direction == TrendDirection.DECLINING]
            
            if improving_trends:
                insights.append(f"{len(improving_trends)} metrics improving")
            
            if declining_trends:
                insights.append(f"{len(declining_trends)} metrics declining")
            
            if analysis.confidence_score > 0.7:
                insights.append("Results are reliable")
            elif analysis.confidence_score > 0.4:
                insights.append("Results have moderate reliability")
            else:
                insights.append("Results need more data")
        
        return insights
    
    def _create_error_visualization(
        self,
        error_message: str,
        export_format: ExportFormat
    ) -> VisualizationResult:
        """Create an error visualization result."""
        chart_config = ChartConfiguration(
            title="خطأ في إنشاء الرسم البياني",
            chart_type=ChartType.LINE_CHART
        )
        
        if export_format == ExportFormat.ASCII:
            chart_data = {"ascii": f"Error: {error_message}"}
        else:
            chart_data = {"error": error_message}
        
        return VisualizationResult(
            chart_data=chart_data,
            export_format=export_format,
            chart_config=chart_config,
            datasets=[],
            insights=[f"Error: {error_message}"]
        )