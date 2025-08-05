"""
Export Manager for Note Taking
Author: Drmusab
Last Modified: 2025-01-08

Handles exporting notes to various formats including Text, JSON, Markdown, and PDF.
Provides integration hooks for external note-taking applications.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger

from .note_taker_skill import Note


class ExportManager:
    """
    Manages note export functionality with support for multiple formats.
    
    Supported formats:
    - Text (.txt)
    - JSON (.json)
    - Markdown (.md)
    - PDF (.pdf) - basic HTML to PDF conversion
    
    Integration support:
    - Logseq (markdown format)
    - AppFlowy (structured format)
    - Joplin (markdown with metadata)
    """
    
    def __init__(self, container: Container):
        """Initialize the export manager."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        
        # Setup export configuration
        self._setup_export_config()
        
        self.logger.info("ExportManager initialized")
    
    def _setup_export_config(self) -> None:
        """Setup export configuration."""
        export_config = self.config.get("note_taker", {}).get("export", {})
        
        self.default_format = export_config.get("default_format", "markdown")
        self.include_audio_links = export_config.get("include_audio_links", True)
        self.timestamp_format = export_config.get("timestamp_format", "iso")
        self.export_directory = Path(export_config.get("export_directory", "data/notes/exports"))
        
        # Integration settings
        integrations = export_config.get("integrations", {})
        self.logseq_enabled = integrations.get("logseq", {}).get("enabled", False)
        self.logseq_auto_export = integrations.get("logseq", {}).get("auto_export", False)
        self.appflowy_enabled = integrations.get("appflowy", {}).get("enabled", False)
        self.joplin_enabled = integrations.get("joplin", {}).get("enabled", False)
        
        # Ensure export directory exists
        self.export_directory.mkdir(parents=True, exist_ok=True)
    
    async def export_note(
        self,
        note: Note,
        format: str = "markdown",
        include_audio: bool = True,
        custom_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Export a note to the specified format.
        
        Args:
            note: Note to export
            format: Export format (text, json, markdown, pdf)
            include_audio: Whether to include audio file references
            custom_path: Custom export path (optional)
            
        Returns:
            Export result with file path and metadata
        """
        format = format.lower()
        
        # Validate format
        if format not in ["text", "json", "markdown", "pdf"]:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Generate export filename
        if custom_path:
            export_path = Path(custom_path)
        else:
            filename = self._generate_filename(note, format)
            export_path = self.export_directory / filename
        
        # Ensure export directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        export_methods = {
            "text": self._export_to_text,
            "json": self._export_to_json,
            "markdown": self._export_to_markdown,
            "pdf": self._export_to_pdf
        }
        
        content = export_methods[format](note, include_audio)
        
        # Write to file
        if format == "json":
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        self.logger.info(f"Exported note {note.metadata.id} to {export_path}")
        
        return {
            "note_id": note.metadata.id,
            "export_path": str(export_path),
            "format": format,
            "file_size": export_path.stat().st_size,
            "exported_at": datetime.now().isoformat(),
            "include_audio": include_audio
        }
    
    def _generate_filename(self, note: Note, format: str) -> str:
        """Generate filename for export."""
        # Clean title for filename
        safe_title = self._sanitize_filename(note.metadata.title)
        
        # Add timestamp if needed
        if self.timestamp_format == "iso":
            timestamp = note.metadata.created_at.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = note.metadata.created_at.strftime("%Y-%m-%d")
        
        # Format extension mapping
        extensions = {
            "text": "txt",
            "json": "json", 
            "markdown": "md",
            "pdf": "html"  # We generate HTML for PDF conversion
        }
        
        extension = extensions.get(format, "txt")
        
        return f"{safe_title}_{timestamp}.{extension}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove or replace invalid characters
        import re
        
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:50]
        
        # Ensure not empty
        if not filename:
            filename = "untitled"
        
        return filename
    
    def _export_to_text(self, note: Note, include_audio: bool) -> str:
        """Export note to plain text format."""
        lines = []
        
        # Title
        lines.append(note.metadata.title)
        lines.append("=" * len(note.metadata.title))
        lines.append("")
        
        # Metadata
        lines.append(f"Created: {self._format_timestamp(note.metadata.created_at)}")
        lines.append(f"Category: {note.metadata.category.value}")
        lines.append(f"Language: {note.metadata.language}")
        if note.metadata.tags:
            lines.append(f"Tags: {', '.join(note.metadata.tags)}")
        lines.append(f"Word Count: {note.metadata.word_count}")
        lines.append(f"Duration: {note.metadata.duration:.1f} seconds")
        lines.append("")
        
        # Audio reference
        if include_audio and note.original_audio_path:
            lines.append(f"Audio File: {note.original_audio_path}")
            lines.append("")
        
        # Summary
        if note.summary:
            lines.append("SUMMARY")
            lines.append("-" * 7)
            lines.append(note.summary)
            lines.append("")
        
        # Key Points
        if note.key_points:
            lines.append("KEY POINTS")
            lines.append("-" * 10)
            for i, point in enumerate(note.key_points, 1):
                lines.append(f"{i}. {point}")
            lines.append("")
        
        # Action Items
        if note.action_items:
            lines.append("ACTION ITEMS")
            lines.append("-" * 12)
            for i, item in enumerate(note.action_items, 1):
                lines.append(f"[ ] {item}")
            lines.append("")
        
        # Definitions
        if note.definitions:
            lines.append("DEFINITIONS")
            lines.append("-" * 11)
            for term, definition in note.definitions.items():
                lines.append(f"• {term}: {definition}")
            lines.append("")
        
        # Full Transcription
        lines.append("FULL TRANSCRIPTION")
        lines.append("-" * 18)
        lines.append(note.transcription)
        
        return "\n".join(lines)
    
    def _export_to_json(self, note: Note, include_audio: bool) -> Dict[str, Any]:
        """Export note to JSON format."""
        export_data = {
            "metadata": {
                "id": note.metadata.id,
                "title": note.metadata.title,
                "category": note.metadata.category.value,
                "language": note.metadata.language,
                "created_at": note.metadata.created_at.isoformat(),
                "updated_at": note.metadata.updated_at.isoformat(),
                "duration": note.metadata.duration,
                "word_count": note.metadata.word_count,
                "confidence": note.metadata.confidence,
                "tags": note.metadata.tags,
                "status": note.metadata.status.value
            },
            "content": {
                "transcription": note.transcription,
                "summary": note.summary,
                "key_points": note.key_points,
                "action_items": note.action_items,
                "definitions": note.definitions
            },
            "analysis": {
                "entities": note.entities,
                "sentiment": note.sentiment
            },
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "format": "json",
                "version": "1.0"
            }
        }
        
        if include_audio and note.original_audio_path:
            export_data["audio"] = {
                "file_path": note.original_audio_path,
                "duration": note.metadata.duration
            }
        
        return export_data
    
    def _export_to_markdown(self, note: Note, include_audio: bool) -> str:
        """Export note to Markdown format."""
        lines = []
        
        # Title
        lines.append(f"# {note.metadata.title}")
        lines.append("")
        
        # Metadata table
        lines.append("## Metadata")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Created | {self._format_timestamp(note.metadata.created_at)} |")
        lines.append(f"| Category | {note.metadata.category.value} |")
        lines.append(f"| Language | {note.metadata.language} |")
        lines.append(f"| Word Count | {note.metadata.word_count} |")
        lines.append(f"| Duration | {note.metadata.duration:.1f}s |")
        lines.append(f"| Confidence | {note.metadata.confidence:.2f} |")
        
        if note.metadata.tags:
            tags_str = " ".join(f"`{tag}`" for tag in note.metadata.tags)
            lines.append(f"| Tags | {tags_str} |")
        
        lines.append("")
        
        # Audio reference
        if include_audio and note.original_audio_path:
            lines.append("## Audio")
            lines.append("")
            lines.append(f"📎 **Audio File:** `{note.original_audio_path}`")
            lines.append("")
        
        # Summary
        if note.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(note.summary)
            lines.append("")
        
        # Key Points
        if note.key_points:
            lines.append("## Key Points")
            lines.append("")
            for point in note.key_points:
                lines.append(f"- {point}")
            lines.append("")
        
        # Action Items
        if note.action_items:
            lines.append("## Action Items")
            lines.append("")
            for item in note.action_items:
                lines.append(f"- [ ] {item}")
            lines.append("")
        
        # Definitions
        if note.definitions:
            lines.append("## Definitions")
            lines.append("")
            for term, definition in note.definitions.items():
                lines.append(f"**{term}:** {definition}")
                lines.append("")
        
        # Full Transcription
        lines.append("## Full Transcription")
        lines.append("")
        lines.append("```")
        lines.append(note.transcription)
        lines.append("```")
        
        return "\n".join(lines)
    
    def _export_to_pdf(self, note: Note, include_audio: bool) -> str:
        """Export note to HTML format (for PDF conversion)."""
        html_lines = []
        
        # HTML head
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html lang='en'>")
        html_lines.append("<head>")
        html_lines.append("<meta charset='UTF-8'>")
        html_lines.append(f"<title>{note.metadata.title}</title>")
        html_lines.append("<style>")
        html_lines.append(self._get_pdf_css())
        html_lines.append("</style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        
        # Title
        html_lines.append(f"<h1>{note.metadata.title}</h1>")
        
        # Metadata
        html_lines.append("<div class='metadata'>")
        html_lines.append(f"<p><strong>Created:</strong> {self._format_timestamp(note.metadata.created_at)}</p>")
        html_lines.append(f"<p><strong>Category:</strong> {note.metadata.category.value}</p>")
        html_lines.append(f"<p><strong>Language:</strong> {note.metadata.language}</p>")
        html_lines.append(f"<p><strong>Word Count:</strong> {note.metadata.word_count}</p>")
        html_lines.append(f"<p><strong>Duration:</strong> {note.metadata.duration:.1f} seconds</p>")
        
        if note.metadata.tags:
            tags_html = " ".join(f"<span class='tag'>{tag}</span>" for tag in note.metadata.tags)
            html_lines.append(f"<p><strong>Tags:</strong> {tags_html}</p>")
        
        html_lines.append("</div>")
        
        # Audio reference
        if include_audio and note.original_audio_path:
            html_lines.append("<div class='audio-section'>")
            html_lines.append("<h2>Audio</h2>")
            html_lines.append(f"<p>📎 <strong>Audio File:</strong> <code>{note.original_audio_path}</code></p>")
            html_lines.append("</div>")
        
        # Summary
        if note.summary:
            html_lines.append("<div class='summary-section'>")
            html_lines.append("<h2>Summary</h2>")
            html_lines.append(f"<p>{note.summary}</p>")
            html_lines.append("</div>")
        
        # Key Points
        if note.key_points:
            html_lines.append("<div class='keypoints-section'>")
            html_lines.append("<h2>Key Points</h2>")
            html_lines.append("<ul>")
            for point in note.key_points:
                html_lines.append(f"<li>{point}</li>")
            html_lines.append("</ul>")
            html_lines.append("</div>")
        
        # Action Items
        if note.action_items:
            html_lines.append("<div class='actions-section'>")
            html_lines.append("<h2>Action Items</h2>")
            html_lines.append("<ul class='action-list'>")
            for item in note.action_items:
                html_lines.append(f"<li>☐ {item}</li>")
            html_lines.append("</ul>")
            html_lines.append("</div>")
        
        # Definitions
        if note.definitions:
            html_lines.append("<div class='definitions-section'>")
            html_lines.append("<h2>Definitions</h2>")
            html_lines.append("<dl>")
            for term, definition in note.definitions.items():
                html_lines.append(f"<dt>{term}</dt>")
                html_lines.append(f"<dd>{definition}</dd>")
            html_lines.append("</dl>")
            html_lines.append("</div>")
        
        # Full Transcription
        html_lines.append("<div class='transcription-section'>")
        html_lines.append("<h2>Full Transcription</h2>")
        html_lines.append(f"<div class='transcription'>{note.transcription}</div>")
        html_lines.append("</div>")
        
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        return "\n".join(html_lines)
    
    def _get_pdf_css(self) -> str:
        """Get CSS styles for PDF export."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 40px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .metadata {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .tag {
            background-color: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
            margin-right: 5px;
        }
        .transcription {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #6c757d;
            white-space: pre-wrap;
            font-family: monospace;
        }
        ul.action-list {
            list-style-type: none;
        }
        ul.action-list li {
            margin: 5px 0;
        }
        dl dt {
            font-weight: bold;
            margin-top: 10px;
        }
        dl dd {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        code {
            background-color: #e9ecef;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        """
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp according to configuration."""
        if self.timestamp_format == "iso":
            return timestamp.isoformat()
        else:
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    async def export_for_logseq(self, note: Note) -> Dict[str, Any]:
        """Export note in Logseq-compatible format."""
        if not self.logseq_enabled:
            raise ValueError("Logseq integration is not enabled")
        
        # Logseq uses markdown with specific conventions
        lines = []
        
        # Properties block (Logseq-specific)
        lines.append("---")
        lines.append(f"title: {note.metadata.title}")
        lines.append(f"category: {note.metadata.category.value}")
        lines.append(f"language: {note.metadata.language}")
        lines.append(f"created: {note.metadata.created_at.isoformat()}")
        lines.append(f"tags: [{', '.join(note.metadata.tags)}]")
        lines.append("---")
        lines.append("")
        
        # Content in Logseq block format
        lines.append(f"# {note.metadata.title}")
        lines.append("")
        
        if note.summary:
            lines.append("- **Summary**")
            lines.append(f"  - {note.summary}")
            lines.append("")
        
        if note.key_points:
            lines.append("- **Key Points**")
            for point in note.key_points:
                lines.append(f"  - {point}")
            lines.append("")
        
        if note.action_items:
            lines.append("- **Action Items**")
            for item in note.action_items:
                lines.append(f"  - TODO {item}")
            lines.append("")
        
        # Full transcription as a block
        lines.append("- **Transcription**")
        for line in note.transcription.split('\n'):
            if line.strip():
                lines.append(f"  - {line.strip()}")
        
        content = "\n".join(lines)
        
        # Save to Logseq pages directory if configured
        filename = f"{self._sanitize_filename(note.metadata.title)}.md"
        export_path = self.export_directory / "logseq" / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "note_id": note.metadata.id,
            "export_path": str(export_path),
            "format": "logseq_markdown",
            "exported_at": datetime.now().isoformat()
        }
    
    async def export_for_appflowy(self, note: Note) -> Dict[str, Any]:
        """Export note in AppFlowy-compatible format."""
        if not self.appflowy_enabled:
            raise ValueError("AppFlowy integration is not enabled")
        
        # AppFlowy uses structured JSON format
        appflowy_data = {
            "type": "document",
            "title": note.metadata.title,
            "metadata": {
                "created_at": note.metadata.created_at.isoformat(),
                "category": note.metadata.category.value,
                "language": note.metadata.language,
                "tags": note.metadata.tags
            },
            "blocks": []
        }
        
        # Add summary block
        if note.summary:
            appflowy_data["blocks"].append({
                "type": "paragraph",
                "style": "heading2",
                "content": "Summary"
            })
            appflowy_data["blocks"].append({
                "type": "paragraph",
                "content": note.summary
            })
        
        # Add key points
        if note.key_points:
            appflowy_data["blocks"].append({
                "type": "paragraph",
                "style": "heading2",
                "content": "Key Points"
            })
            for point in note.key_points:
                appflowy_data["blocks"].append({
                    "type": "bullet_list_item",
                    "content": point
                })
        
        # Add action items
        if note.action_items:
            appflowy_data["blocks"].append({
                "type": "paragraph",
                "style": "heading2", 
                "content": "Action Items"
            })
            for item in note.action_items:
                appflowy_data["blocks"].append({
                    "type": "todo_list_item",
                    "content": item,
                    "checked": False
                })
        
        # Add transcription
        appflowy_data["blocks"].append({
            "type": "paragraph",
            "style": "heading2",
            "content": "Transcription"
        })
        appflowy_data["blocks"].append({
            "type": "code_block",
            "content": note.transcription,
            "language": "text"
        })
        
        # Save as JSON
        filename = f"{self._sanitize_filename(note.metadata.title)}.json"
        export_path = self.export_directory / "appflowy" / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(appflowy_data, f, indent=2, ensure_ascii=False)
        
        return {
            "note_id": note.metadata.id,
            "export_path": str(export_path),
            "format": "appflowy_json",
            "exported_at": datetime.now().isoformat()
        }
    
    async def export_for_joplin(self, note: Note) -> Dict[str, Any]:
        """Export note in Joplin-compatible format."""
        if not self.joplin_enabled:
            raise ValueError("Joplin integration is not enabled")
        
        # Joplin uses markdown with metadata
        lines = []
        
        # Title
        lines.append(f"# {note.metadata.title}")
        lines.append("")
        
        # Metadata as Joplin custom fields
        lines.append("<!-- Metadata -->")
        lines.append(f"<!-- category: {note.metadata.category.value} -->")
        lines.append(f"<!-- language: {note.metadata.language} -->")
        lines.append(f"<!-- created: {note.metadata.created_at.isoformat()} -->")
        lines.append(f"<!-- word_count: {note.metadata.word_count} -->")
        lines.append(f"<!-- duration: {note.metadata.duration} -->")
        lines.append("")
        
        # Tags in Joplin format
        if note.metadata.tags:
            for tag in note.metadata.tags:
                lines.append(f"#{tag}")
            lines.append("")
        
        # Content similar to markdown export
        if note.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(note.summary)
            lines.append("")
        
        if note.key_points:
            lines.append("## Key Points")
            lines.append("")
            for point in note.key_points:
                lines.append(f"- {point}")
            lines.append("")
        
        if note.action_items:
            lines.append("## Action Items")
            lines.append("")
            for item in note.action_items:
                lines.append(f"- [ ] {item}")
            lines.append("")
        
        lines.append("## Transcription")
        lines.append("")
        lines.append(note.transcription)
        
        content = "\n".join(lines)
        
        # Save as markdown
        filename = f"{self._sanitize_filename(note.metadata.title)}.md"
        export_path = self.export_directory / "joplin" / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "note_id": note.metadata.id,
            "export_path": str(export_path),
            "format": "joplin_markdown",
            "exported_at": datetime.now().isoformat()
        }
    
    async def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics and information."""
        stats = {
            "export_directory": str(self.export_directory),
            "default_format": self.default_format,
            "supported_formats": ["text", "json", "markdown", "pdf"],
            "integrations": {
                "logseq": {"enabled": self.logseq_enabled, "auto_export": self.logseq_auto_export},
                "appflowy": {"enabled": self.appflowy_enabled},
                "joplin": {"enabled": self.joplin_enabled}
            },
            "total_exports": 0,
            "exports_by_format": {}
        }
        
        # Count existing export files
        if self.export_directory.exists():
            for format_dir in self.export_directory.iterdir():
                if format_dir.is_dir():
                    count = len(list(format_dir.glob("*")))
                    stats["exports_by_format"][format_dir.name] = count
                    stats["total_exports"] += count
        
        return stats