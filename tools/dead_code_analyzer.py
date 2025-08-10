#!/usr/bin/env python3
"""
Dead Code Analysis Tool for AI Assistant Project

This script performs comprehensive dead code analysis including:
- Unused imports detection
- Dead functions and classes
- Unreferenced files
- Import graph analysis

Usage:
    python tools/dead_code_analyzer.py [options]

Options:
    --confidence LEVEL    Minimum confidence level (default: 80)
    --report-format FORMAT   Output format: text, json, html (default: text)
    --output FILE         Output file path (default: stdout)
    --exclude PATTERNS    Comma-separated patterns to exclude
    --include-tests       Include test files in analysis (default: False)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add src to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class DeadCodeAnalyzer:
    """Comprehensive dead code analyzer for the AI Assistant project."""
    
    def __init__(self, confidence: int = 80, include_tests: bool = False):
        self.confidence = confidence
        self.include_tests = include_tests
        self.project_root = Path(__file__).parent.parent
        self.src_path = self.project_root / "src"
        self.test_path = self.project_root / "tests"
        
    def run_vulture(self, paths: List[str], exclude_patterns: List[str] = None) -> str:
        """Run vulture on specified paths and return output."""
        cmd = [
            "vulture",
            *paths,
            f"--min-confidence={self.confidence}",
            "--config", str(self.project_root / "setup.cfg")
        ]
        
        if exclude_patterns:
            for pattern in exclude_patterns:
                cmd.extend(["--exclude", pattern])
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running vulture: {e}"
    
    def analyze_import_usage(self) -> Dict[str, List[str]]:
        """Analyze which files are never imported."""
        # This is a simplified version - a full implementation would parse AST
        python_files = list(self.src_path.rglob("*.py"))
        potentially_unused = []
        
        for file_path in python_files:
            # Skip __init__.py files and main modules
            if file_path.name in ["__init__.py", "main.py", "cli.py"]:
                continue
                
            relative_path = file_path.relative_to(self.project_root)
            # Convert file path to import path
            import_path = str(relative_path).replace("/", ".").replace(".py", "")
            
            # Check if this module is imported anywhere
            is_imported = False
            for check_file in python_files:
                if check_file == file_path:
                    continue
                try:
                    content = check_file.read_text(encoding='utf-8')
                    if import_path in content or file_path.stem in content:
                        is_imported = True
                        break
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not is_imported:
                potentially_unused.append(str(relative_path))
        
        return {"potentially_unused_files": potentially_unused}
    
    def generate_report(self, format_type: str = "text") -> str:
        """Generate comprehensive dead code report."""
        print("Running dead code analysis...", file=sys.stderr)
        
        # Run vulture analysis
        paths = [str(self.src_path)]
        if self.include_tests:
            paths.append(str(self.test_path))
        
        vulture_output = self.run_vulture(paths)
        
        # Analyze file usage
        file_analysis = self.analyze_import_usage()
        
        if format_type == "json":
            return self._generate_json_report(vulture_output, file_analysis)
        elif format_type == "html":
            return self._generate_html_report(vulture_output, file_analysis)
        else:
            return self._generate_text_report(vulture_output, file_analysis)
    
    def _generate_text_report(self, vulture_output: str, file_analysis: Dict) -> str:
        """Generate text format report."""
        report = []
        report.append("=" * 80)
        report.append("AI ASSISTANT - DEAD CODE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {self._get_timestamp()}")
        report.append(f"Confidence Level: {self.confidence}%")
        report.append(f"Include Tests: {self.include_tests}")
        report.append("")
        
        # Vulture findings
        report.append("1. VULTURE ANALYSIS (Unused Code Detection)")
        report.append("-" * 50)
        if vulture_output.strip():
            lines = vulture_output.strip().split('\n')
            report.append(f"Found {len(lines)} potential issues:")
            report.append("")
            
            # Categorize findings
            categories = {
                "unused import": [],
                "unused variable": [],
                "unused function": [],
                "unused class": [],
                "unused method": [],
                "unused attribute": [],
                "unused code": []
            }
            
            for line in lines:
                categorized = False
                for category in categories:
                    if category in line.lower():
                        categories[category].append(line)
                        categorized = True
                        break
                if not categorized:
                    categories["unused code"].append(line)
            
            for category, items in categories.items():
                if items:
                    report.append(f"{category.upper()}:")
                    for item in items:
                        report.append(f"  {item}")
                    report.append("")
        else:
            report.append("✅ No unused code detected!")
        
        report.append("")
        
        # File analysis
        report.append("2. FILE USAGE ANALYSIS")
        report.append("-" * 50)
        unused_files = file_analysis.get("potentially_unused_files", [])
        if unused_files:
            report.append(f"Found {len(unused_files)} potentially unused files:")
            report.append("")
            for file_path in unused_files:
                report.append(f"  {file_path}")
        else:
            report.append("✅ All files appear to be referenced!")
        
        report.append("")
        report.append("3. RECOMMENDATIONS")
        report.append("-" * 50)
        report.append("• Review each finding carefully before removing code")
        report.append("• Some findings may be false positives (e.g., dynamic imports)")
        report.append("• Consider adding vulture whitelist for legitimate unused code")
        report.append("• Run tests after removing any dead code")
        report.append("")
        
        return "\n".join(report)
    
    def _generate_json_report(self, vulture_output: str, file_analysis: Dict) -> str:
        """Generate JSON format report."""
        vulture_lines = vulture_output.strip().split('\n') if vulture_output.strip() else []
        
        data = {
            "analysis_info": {
                "timestamp": self._get_timestamp(),
                "confidence_level": self.confidence,
                "include_tests": self.include_tests,
                "total_findings": len(vulture_lines)
            },
            "vulture_findings": vulture_lines,
            "file_analysis": file_analysis,
            "summary": {
                "has_unused_code": len(vulture_lines) > 0,
                "has_unused_files": len(file_analysis.get("potentially_unused_files", [])) > 0
            }
        }
        
        return json.dumps(data, indent=2)
    
    def _generate_html_report(self, vulture_output: str, file_analysis: Dict) -> str:
        """Generate HTML format report."""
        vulture_lines = vulture_output.strip().split('\n') if vulture_output.strip() else []
        unused_files = file_analysis.get("potentially_unused_files", [])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Assistant - Dead Code Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .finding {{ margin: 10px 0; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; }}
        .success {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .stats {{ display: flex; gap: 20px; }}
        .stat {{ background: #e9ecef; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Assistant - Dead Code Analysis Report</h1>
        <p>Generated: {self._get_timestamp()}</p>
        <div class="stats">
            <div class="stat">
                <strong>Confidence Level:</strong> {self.confidence}%
            </div>
            <div class="stat">
                <strong>Vulture Findings:</strong> {len(vulture_lines)}
            </div>
            <div class="stat">
                <strong>Unused Files:</strong> {len(unused_files)}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Vulture Analysis</h2>
        """
        
        if vulture_lines:
            html += f'<p class="warning">Found {len(vulture_lines)} potential issues:</p>'
            for line in vulture_lines:
                html += f'<div class="finding">{line}</div>'
        else:
            html += '<p class="success">✅ No unused code detected!</p>'
        
        html += """
    </div>
    
    <div class="section">
        <h2>File Usage Analysis</h2>
        """
        
        if unused_files:
            html += f'<p class="warning">Found {len(unused_files)} potentially unused files:</p>'
            html += '<ul>'
            for file_path in unused_files:
                html += f'<li>{file_path}</li>'
            html += '</ul>'
        else:
            html += '<p class="success">✅ All files appear to be referenced!</p>'
        
        html += """
    </div>
</body>
</html>
        """
        
        return html
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dead Code Analysis Tool for AI Assistant Project"
    )
    parser.add_argument(
        "--confidence", 
        type=int, 
        default=80,
        help="Minimum confidence level (default: 80)"
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in analysis"
    )
    
    args = parser.parse_args()
    
    analyzer = DeadCodeAnalyzer(
        confidence=args.confidence,
        include_tests=args.include_tests
    )
    
    report = analyzer.generate_report(args.report_format)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()