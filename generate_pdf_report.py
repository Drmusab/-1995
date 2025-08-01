#!/usr/bin/env python3
"""
PDF Generator for Architecture Analysis Report
Converts the Markdown report to a properly formatted PDF
"""

import markdown
import os
from pathlib import Path

def generate_pdf_report():
    """Generate a PDF version of the architecture analysis report"""
    
    # Paths
    current_dir = Path(__file__).parent
    markdown_file = current_dir / "docs" / "architecture" / "ARCHITECTURE_ANALYSIS_REPORT.md"
    html_file = current_dir / "docs" / "architecture" / "report.html"
    
    # Read the markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert to HTML with extensions
    md = markdown.Markdown(extensions=['toc', 'tables', 'fenced_code', 'codehilite'])
    html_content = md.convert(markdown_content)
    
    # Create a complete HTML document with styling
    complete_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Assistant System Architecture Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            h1 {{
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 8px;
            }}
            h3 {{
                color: #34495e;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', 'Lucida Console', monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 5px;
                padding: 15px;
                overflow-x: auto;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding-left: 20px;
                color: #555;
            }}
            .toc {{
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
            }}
            .toc ul {{
                list-style-type: none;
                padding-left: 20px;
            }}
            .toc a {{
                text-decoration: none;
                color: #3498db;
            }}
            .toc a:hover {{
                text-decoration: underline;
            }}
            ul, ol {{
                padding-left: 30px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            .header-info {{
                background-color: #e8f4fd;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
            @media print {{
                body {{
                    margin: 0;
                    padding: 15px;
                }}
                h1, h2, h3 {{
                    page-break-after: avoid;
                }}
                pre, table {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header-info">
            <h1>AI Assistant System Architecture Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p><strong>Version:</strong> 1.0</p>
            <p><strong>Document Type:</strong> Comprehensive Architecture Analysis</p>
        </div>
        
        {html_content}
        
        <div class="footer">
            <p>This document was automatically generated from the AI Assistant system architecture analysis.</p>
            <p>For the latest version, please refer to the source repository.</p>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(complete_html)
    
    print(f"✓ HTML report generated: {html_file}")
    print("📄 HTML file can be:")
    print("   - Opened in a browser and printed to PDF")
    print("   - Converted using wkhtmltopdf if available")
    print("   - Shared as an HTML document")
    
    return html_file

if __name__ == "__main__":
    from datetime import datetime
    generate_pdf_report()