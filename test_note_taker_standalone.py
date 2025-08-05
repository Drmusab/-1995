#!/usr/bin/env python3
"""
Standalone test for Note Taking Skill
Author: Drmusab
Last Modified: 2025-01-08

Simple standalone test to verify the note-taking skill functionality
without complex dependencies.
"""

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_note_taker_basic():
    """Basic test of note taker functionality."""
    print("Testing Note Taker Skill...")
    
    try:
        # Create minimal mock container
        container = MagicMock()
        
        # Mock config
        config = MagicMock()
        config.get.return_value = {
            "audio": {"format": "wav", "quality": "high"},
            "processing": {"auto_summarize": True, "detect_action_items": True},
            "export": {"default_format": "markdown"},
            "storage": {"notes_directory": "/tmp/test_notes"}
        }
        
        # Mock dependencies
        container.get.side_effect = lambda cls: {
            "ConfigLoader": config,
            "MemoryManager": AsyncMock(),
            "BilingualManager": MagicMock(),
            "EnhancedWhisperTranscriber": AsyncMock()
        }.get(str(cls), MagicMock())
        
        container.get_optional.return_value = None
        
        # Import and create note taker skill
        from src.skills.builtin.note_taker.note_taker_skill import (
            NoteTakerSkill, 
            NoteCategory, 
            NoteStatus
        )
        
        skill = NoteTakerSkill(container)
        print("✓ NoteTakerSkill created successfully")
        
        # Test start recording
        result = await skill.start_recording(user_id="test_user")
        note_id = result["note_id"]
        print(f"✓ Recording started with note ID: {note_id}")
        
        # Test note retrieval
        assert note_id in skill.notes
        assert skill.notes[note_id].metadata.status == NoteStatus.RECORDING
        print("✓ Note created and stored correctly")
        
        # Test categories
        categories = await skill.get_categories()
        assert len(categories) > 0
        print(f"✓ Categories retrieved: {len(categories)} categories")
        
        # Test statistics
        stats = await skill.get_statistics()
        assert "total_notes" in stats
        print(f"✓ Statistics retrieved: {stats['total_notes']} total notes")
        
        # Test search (empty results)
        results = await skill.search_notes(query="test")
        print(f"✓ Search functionality works: {len(results)} results")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_content_analyzer():
    """Test content analyzer functionality."""
    print("\nTesting Content Analyzer...")
    
    try:
        # Create minimal mock container
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
        
        from src.skills.builtin.note_taker.content_analyzer import ContentAnalyzer, NoteCategory
        
        analyzer = ContentAnalyzer(container)
        print("✓ ContentAnalyzer created successfully")
        
        # Test categorization
        task_text = "I need to complete the project report and submit it by Friday"
        category = await analyzer.categorize_content(task_text)
        print(f"✓ Task categorization: {category}")
        
        idea_text = "I have an innovative idea for improving the user interface"
        category = await analyzer.categorize_content(idea_text)
        print(f"✓ Idea categorization: {category}")
        
        # Test action item extraction
        action_text = "We need to review the code. I should call the client tomorrow."
        actions = await analyzer.extract_action_items(action_text)
        print(f"✓ Action items extracted: {len(actions)} items")
        
        # Test definition extraction
        def_text = "Machine Learning is a subset of artificial intelligence"
        definitions = await analyzer.extract_definitions(def_text)
        print(f"✓ Definitions extracted: {len(definitions)} definitions")
        
        # Test tag extraction
        tag_text = "This is about #machinelearning and neural networks"
        tags = await analyzer.extract_tags(tag_text)
        print(f"✓ Tags extracted: {len(tags)} tags")
        
        # Test title generation
        content = "Today we discussed the new project requirements and timeline"
        title = await analyzer.generate_title(content)
        print(f"✓ Title generated: '{title}'")
        
        print("🎉 Content analyzer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Content analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_summarizer():
    """Test summarizer functionality."""
    print("\nTesting Summarizer...")
    
    try:
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
        
        from src.skills.builtin.note_taker.summarizer import NoteSummarizer
        
        summarizer = NoteSummarizer(container)
        print("✓ NoteSummarizer created successfully")
        
        long_text = """
        Machine learning is a powerful subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. It uses 
        algorithms and statistical models to analyze data and make predictions or decisions. 
        There are three main types of machine learning: supervised learning, unsupervised 
        learning, and reinforcement learning. Each type has its own applications and methodologies.
        """
        
        # Test summarization
        summary = await summarizer.create_summary(long_text, max_length=100)
        print(f"✓ Summary created: '{summary[:50]}...'")
        assert len(summary) <= 120  # Allow some flexibility
        
        # Test key point extraction
        key_points = await summarizer.extract_key_points(long_text, max_points=3)
        print(f"✓ Key points extracted: {len(key_points)} points")
        
        # Test bullet points
        bullet_points = await summarizer.create_bullet_points(long_text)
        print(f"✓ Bullet points created: {len(bullet_points)} points")
        
        # Test structured outline
        outline = await summarizer.create_structured_outline(long_text)
        print(f"✓ Structured outline created with {len(outline['sections'])} sections")
        
        print("🎉 Summarizer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Summarizer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_export_manager():
    """Test export manager functionality."""
    print("\nTesting Export Manager...")
    
    try:
        container = MagicMock()
        config = MagicMock()
        config.get.return_value = {
            "export": {
                "default_format": "markdown",
                "export_directory": "/tmp/test_exports"
            }
        }
        
        container.get.side_effect = lambda cls: {
            "ConfigLoader": config
        }.get(str(cls), MagicMock())
        
        from src.skills.builtin.note_taker.export_manager import ExportManager
        from src.skills.builtin.note_taker.note_taker_skill import Note, NoteMetadata, NoteCategory
        
        export_manager = ExportManager(container)
        print("✓ ExportManager created successfully")
        
        # Create a test note
        note_metadata = NoteMetadata(
            id="test_note",
            title="Test Note",
            category=NoteCategory.IDEAS,
            language="en",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["test", "sample"]
        )
        note = Note(
            metadata=note_metadata,
            transcription="This is a test transcription.",
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            action_items=["Action 1", "Action 2"]
        )
        
        # Test different export formats
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_manager.export_directory = Path(tmp_dir)
            
            # Test markdown export
            result = await export_manager.export_note(note, format="markdown")
            print(f"✓ Markdown export: {result['format']}")
            
            # Test JSON export
            result = await export_manager.export_note(note, format="json")
            print(f"✓ JSON export: {result['format']}")
            
            # Test text export
            result = await export_manager.export_note(note, format="text")
            print(f"✓ Text export: {result['format']}")
            
            # Check files were created
            files = list(Path(tmp_dir).glob("*"))
            print(f"✓ Export files created: {len(files)} files")
        
        # Test filename sanitization
        sanitized = export_manager._sanitize_filename("Test/Note<>:File")
        print(f"✓ Filename sanitization: '{sanitized}'")
        
        print("🎉 Export manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Export manager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Note Taker Skill Tests")
    print("=" * 50)
    
    tests = [
        test_note_taker_basic,
        test_content_analyzer,
        test_summarizer,
        test_export_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
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
        print("🎉 All tests passed! Note Taker Skill is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)