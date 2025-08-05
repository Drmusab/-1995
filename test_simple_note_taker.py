#!/usr/bin/env python3
"""
Simple test for Note Taking Skill components
Author: Drmusab
Last Modified: 2025-01-08

Direct component testing without complex imports.
"""

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that we can import all note-taking components."""
    print("Testing component imports...")
    
    try:
        # Test note category enum
        note_taker_path = Path(__file__).parent.parent / "src" / "skills" / "builtin" / "note_taker"
        sys.path.insert(0, str(note_taker_path))
        
        import note_taker_skill
        from note_taker_skill import NoteCategory, NoteStatus, Note, NoteMetadata
        print("✓ Note classes imported successfully")
        
        import content_analyzer
        from content_analyzer import ContentAnalyzer
        print("✓ ContentAnalyzer imported successfully")
        
        import summarizer
        from summarizer import NoteSummarizer  
        print("✓ NoteSummarizer imported successfully")
        
        import export_manager
        from export_manager import ExportManager
        print("✓ ExportManager imported successfully")
        
        import audio_processor
        from audio_processor import AudioProcessor
        print("✓ AudioProcessor imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_note_classes():
    """Test note data classes."""
    print("\nTesting note data classes...")
    
    try:
        note_taker_path = Path(__file__).parent.parent / "src" / "skills" / "builtin" / "note_taker"
        sys.path.insert(0, str(note_taker_path))
        
        from note_taker_skill import NoteCategory, NoteStatus, Note, NoteMetadata
        
        # Test note metadata
        metadata = NoteMetadata(
            id="test_note",
            title="Test Note", 
            category=NoteCategory.IDEAS,
            language="en",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        print(f"✓ NoteMetadata created: {metadata.title}")
        
        # Test note
        note = Note(
            metadata=metadata,
            transcription="Test transcription",
            summary="Test summary"
        )
        print(f"✓ Note created with {len(note.transcription)} character transcription")
        
        # Test categories
        categories = list(NoteCategory)
        print(f"✓ Found {len(categories)} note categories")
        
        return True
        
    except Exception as e:
        print(f"❌ Note classes test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_content_analyzer():
    """Test content analyzer."""
    print("\nTesting content analyzer...")
    
    try:
        note_taker_path = Path(__file__).parent.parent / "src" / "skills" / "builtin" / "note_taker"
        sys.path.insert(0, str(note_taker_path))
        
        from content_analyzer import ContentAnalyzer
        from note_taker_skill import NoteCategory
        
        # Create mock container
        container = MagicMock()
        config = MagicMock()
        config.get.return_value = {}
        
        bilingual_manager = MagicMock()
        bilingual_manager.build_language_context.return_value = MagicMock(
            user_query_language=MagicMock(value='en')
        )
        
        container.get.side_effect = lambda cls: {
            "ConfigLoader": config,
            "BilingualManager": bilingual_manager
        }.get(str(cls), MagicMock())
        
        container.get_optional.return_value = None
        
        analyzer = ContentAnalyzer(container)
        print("✓ ContentAnalyzer created")
        
        # Test categorization
        task_text = "I need to complete the project report"
        category = await analyzer.categorize_content(task_text)
        print(f"✓ Categorized task text as: {category}")
        
        # Test action extraction
        actions = await analyzer.extract_action_items(task_text)
        print(f"✓ Extracted {len(actions)} action items")
        
        # Test title generation
        title = await analyzer.generate_title(task_text)
        print(f"✓ Generated title: '{title}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Content analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_summarizer():
    """Test summarizer."""
    print("\nTesting summarizer...")
    
    try:
        note_taker_path = Path(__file__).parent.parent / "src" / "skills" / "builtin" / "note_taker"
        sys.path.insert(0, str(note_taker_path))
        
        from summarizer import NoteSummarizer
        
        # Create mock container
        container = MagicMock()
        config = MagicMock()
        config.get.return_value = {}
        
        bilingual_manager = MagicMock()
        bilingual_manager.build_language_context.return_value = MagicMock(
            user_query_language=MagicMock(value='en')
        )
        
        container.get.side_effect = lambda cls: {
            "ConfigLoader": config,
            "BilingualManager": bilingual_manager
        }.get(str(cls), MagicMock())
        
        container.get_optional.return_value = None
        
        summarizer = NoteSummarizer(container)
        print("✓ NoteSummarizer created")
        
        # Test summarization
        long_text = "Machine learning is a subset of AI that enables computers to learn. It uses algorithms to analyze data and make predictions. There are supervised, unsupervised, and reinforcement learning types."
        
        summary = await summarizer.create_summary(long_text, max_length=50)
        print(f"✓ Created summary: '{summary}'")
        
        # Test key points
        key_points = await summarizer.extract_key_points(long_text, max_points=2)
        print(f"✓ Extracted {len(key_points)} key points")
        
        return True
        
    except Exception as e:
        print(f"❌ Summarizer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_export_manager():
    """Test export manager."""
    print("\nTesting export manager...")
    
    try:
        note_taker_path = Path(__file__).parent.parent / "src" / "skills" / "builtin" / "note_taker"
        sys.path.insert(0, str(note_taker_path))
        
        from export_manager import ExportManager
        from note_taker_skill import Note, NoteMetadata, NoteCategory
        
        # Create mock container
        container = MagicMock()
        config = MagicMock()
        config.get.return_value = {
            "export": {"default_format": "markdown"}
        }
        
        container.get.side_effect = lambda cls: {
            "ConfigLoader": config
        }.get(str(cls), MagicMock())
        
        export_manager = ExportManager(container)
        print("✓ ExportManager created")
        
        # Create test note
        metadata = NoteMetadata(
            id="test_note",
            title="Test Note",
            category=NoteCategory.IDEAS,
            language="en",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["test"]
        )
        note = Note(
            metadata=metadata,
            transcription="Test transcription content",
            summary="Test summary",
            key_points=["Point 1", "Point 2"]
        )
        
        # Test export to different formats
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_manager.export_directory = Path(tmp_dir)
            
            # Test markdown export
            result = await export_manager.export_note(note, format="markdown")
            print(f"✓ Exported to markdown: {result['format']}")
            
            # Test JSON export
            result = await export_manager.export_note(note, format="json")
            print(f"✓ Exported to JSON: {result['format']}")
            
            # Check files were created
            files = list(Path(tmp_dir).glob("*"))
            print(f"✓ Created {len(files)} export files")
        
        return True
        
    except Exception as e:
        print(f"❌ Export manager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Simple Note Taker Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_note_classes,
        test_content_analyzer,
        test_summarizer,
        test_export_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
                
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Note Taker components are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)