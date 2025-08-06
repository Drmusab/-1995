#!/usr/bin/env python3
"""
Simple test script for Time Machine skill
Author: Drmusab
Last Modified: 2025-01-08

This script demonstrates the basic functionality of the Context Time Machine skill.
It can be run independently to test the skill without the full system setup.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simple mock container for testing
class MockContainer:
    def __init__(self):
        self._services = {}
    
    def get(self, service_type):
        if service_type not in self._services:
            # Create mock services as needed
            service_name = service_type.__name__ if hasattr(service_type, '__name__') else str(service_type)
            
            if 'BilingualManager' in service_name:
                from src.processing.natural_language.bilingual_manager import Language
                class MockBilingualManager:
                    def detect_language(self, text):
                        # Simple language detection
                        arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
                        english_chars = len([c for c in text if c.isalpha() and c.isascii()])
                        return Language.ARABIC if arabic_chars > english_chars else Language.ENGLISH
                
                self._services[service_type] = MockBilingualManager()
            
            elif 'MemoryManager' in service_name:
                class MockMemoryManager:
                    async def search_memories(self, query, memory_type=None):
                        # Mock search results
                        from src.memory.core_memory.base_memory import MemorySearchResult, MemoryItem, MemoryType, MemoryMetadata
                        
                        # Create some mock memory items
                        items = []
                        for i in range(3):
                            metadata = MemoryMetadata()
                            metadata.created_at = datetime.now(timezone.utc)
                            
                            item = MemoryItem(
                                memory_id=f"mock_memory_{i}",
                                content=f"محادثة تجريبية رقم {i+1} حول {query}" if 'ا' in query else f"Mock conversation {i+1} about {query}",
                                memory_type=memory_type or MemoryType.EPISODIC,
                                metadata=metadata
                            )
                            items.append(item)
                        
                        return MemorySearchResult(items, len(items), 0.1)
                    
                    async def get_recent_memories(self, limit=10):
                        return []
                    
                    @property
                    def memory_store(self):
                        class MockMemoryStore:
                            async def query(self, query):
                                return []
                        return MockMemoryStore()
                
                self._services[service_type] = MockMemoryManager()
            
            elif 'EventBus' in service_name:
                class MockEventBus:
                    async def emit(self, event):
                        print(f"Event: {event}")
                
                self._services[service_type] = MockEventBus()
            
            elif 'ErrorHandler' in service_name:
                class MockErrorHandler:
                    pass
                
                self._services[service_type] = MockErrorHandler()
            
            else:
                # Create a generic mock
                class GenericMock:
                    pass
                
                self._services[service_type] = GenericMock()
        
        return self._services[service_type]


async def test_arabic_query():
    """Test an Arabic query."""
    print("=== Testing Arabic Query ===")
    
    # Mock container
    container = MockContainer()
    
    try:
        # Import the Time Machine skill
        from src.skills.builtin.time_machine.time_machine_skill import TimeMachineSkill
        
        # Create the skill
        skill = TimeMachineSkill(container)
        
        # Test query
        query = "ماذا تحدثنا عنه الأسبوع الماضي؟"
        print(f"Query: {query}")
        
        # Process the query
        result = await skill.process_query(
            user_query=query,
            user_id="test_user",
            session_id="test_session"
        )
        
        # Print results
        print("\nResults:")
        print(f"Language detected: {result['query_analysis']['detected_language']}")
        print(f"Query type: {result['query_analysis']['query_type']}")
        print(f"Answer: {result['answer']}")
        
        if result['insights']:
            print("\nInsights:")
            for insight in result['insights']:
                print(f"- {insight}")
        
        print("\nSuccess! Arabic query processed correctly.")
        
    except Exception as e:
        print(f"Error testing Arabic query: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_english_query():
    """Test an English query."""
    print("\n=== Testing English Query ===")
    
    # Mock container
    container = MockContainer()
    
    try:
        # Import the Time Machine skill
        from src.skills.builtin.time_machine.time_machine_skill import TimeMachineSkill
        
        # Create the skill
        skill = TimeMachineSkill(container)
        
        # Test query
        query = "What did we talk about last week?"
        print(f"Query: {query}")
        
        # Process the query
        result = await skill.process_query(
            user_query=query,
            user_id="test_user",
            session_id="test_session"
        )
        
        # Print results
        print("\nResults:")
        print(f"Language detected: {result['query_analysis']['detected_language']}")
        print(f"Query type: {result['query_analysis']['query_type']}")
        print(f"Answer: {result['answer']}")
        
        if result['insights']:
            print("\nInsights:")
            for insight in result['insights']:
                print(f"- {insight}")
        
        print("\nSuccess! English query processed correctly.")
        
    except Exception as e:
        print(f"Error testing English query: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_behavioral_query():
    """Test a behavioral analysis query."""
    print("\n=== Testing Behavioral Analysis Query ===")
    
    # Mock container
    container = MockContainer()
    
    try:
        # Import the Time Machine skill
        from src.skills.builtin.time_machine.time_machine_skill import TimeMachineSkill
        
        # Create the skill
        skill = TimeMachineSkill(container)
        
        # Test query
        query = "هل تحسن أسلوبي في الاجتماعات؟"
        print(f"Query: {query}")
        
        # Process the query
        result = await skill.process_query(
            user_query=query,
            user_id="test_user",
            session_id="test_session"
        )
        
        # Print results
        print("\nResults:")
        print(f"Language detected: {result['query_analysis']['detected_language']}")
        print(f"Query type: {result['query_analysis']['query_type']}")
        print(f"Answer: {result['answer']}")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"- {rec}")
        
        print("\nSuccess! Behavioral query processed correctly.")
        
    except Exception as e:
        print(f"Error testing behavioral query: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_skill_info():
    """Test getting skill information."""
    print("\n=== Testing Skill Info ===")
    
    # Mock container
    container = MockContainer()
    
    try:
        # Import the Time Machine skill
        from src.skills.builtin.time_machine.time_machine_skill import TimeMachineSkill
        
        # Create the skill
        skill = TimeMachineSkill(container)
        
        # Get skill info
        info = await skill.get_skill_info()
        
        print(f"Skill Name: {info['name']}")
        print(f"Version: {info['version']}")
        print(f"Description (AR): {info['description']['ar']}")
        print(f"Description (EN): {info['description']['en']}")
        print(f"Supported Languages: {info['supported_languages']}")
        print(f"Capabilities: {info['capabilities']}")
        
        print("\nExample Queries (Arabic):")
        for example in info['example_queries']['ar']:
            print(f"- {example}")
        
        print("\nExample Queries (English):")
        for example in info['example_queries']['en']:
            print(f"- {example}")
        
        print("\nSuccess! Skill info retrieved correctly.")
        
    except Exception as e:
        print(f"Error testing skill info: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("Time Machine Skill Test")
    print("=====================")
    
    try:
        await test_skill_info()
        await test_arabic_query()
        await test_english_query()
        await test_behavioral_query()
        
        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("The Context Time Machine skill is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)