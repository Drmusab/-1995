"""
Integration Test for Bilingual AI Assistant
Author: AI Assistant  
Last Modified: 2025-01-01

This script provides comprehensive integration testing for the bilingual AI assistant,
demonstrating various language scenarios and validating the implementation.
"""

import asyncio
from unittest.mock import Mock, AsyncMock
from src.processing.natural_language.bilingual_manager import (
    BilingualManager, 
    Language, 
    ContentType
)
from src.core.dependency_injection import Container
from src.core.config.loader import ConfigLoader


class BilingualIntegrationTest:
    """Integration test suite for bilingual functionality."""
    
    def __init__(self):
        """Initialize the test environment."""
        print("🌍 Initializing Bilingual AI Assistant Integration Tests...")
        self.setup_mocks()
        self.manager = BilingualManager(self.container)
        print("✅ BilingualManager initialized successfully\n")
    
    def setup_mocks(self):
        """Setup mock dependencies."""
        self.container = Mock(spec=Container)
        self.config_loader = Mock(spec=ConfigLoader)
        self.config_loader.get.return_value = {
            "primary_language": "ar",
            "secondary_language": "en", 
            "enable_mixing": True,
            "technical_domain": "AI/ML systems",
            "codebase_language": "English"
        }
        self.container.get.return_value = self.config_loader
    
    def test_scenario_1_arabic_general_conversation(self):
        """Test Scenario 1: Arabic general conversation."""
        print("📝 Test Scenario 1: Arabic General Conversation")
        
        # Arabic query about AI
        user_input = "ما هو الذكاء الاصطناعي وكيف يعمل؟"
        print(f"👤 User Input: {user_input}")
        
        # Build context
        context = self.manager.build_language_context(user_input)
        print(f"🔍 Detected Language: {context.user_query_language.value}")
        print(f"📂 Content Type: {context.content_type.value}")
        
        # Determine response language
        response_lang = self.manager.determine_response_language(context)
        print(f"🗣️ Response Language: {response_lang.value}")
        
        # Create prompt
        prompt = self.manager.create_bilingual_prompt(user_input, context)
        print(f"💬 Prompt Contains Arabic Instructions: {'العربية' in prompt}")
        
        # Assertions
        assert context.user_query_language == Language.ARABIC
        assert context.content_type == ContentType.GENERAL_CONVERSATION
        assert response_lang == Language.ARABIC
        assert "العربية الفصحى الحديثة" in prompt
        
        print("✅ Scenario 1 PASSED: Arabic conversation handled correctly\n")
    
    def test_scenario_2_english_technical_query(self):
        """Test Scenario 2: English technical query."""
        print("📝 Test Scenario 2: English Technical Query")
        
        user_input = "How do I debug the FastAPI endpoint error?"
        print(f"👤 User Input: {user_input}")
        
        context = self.manager.build_language_context(user_input)
        print(f"🔍 Detected Language: {context.user_query_language.value}")
        print(f"📂 Content Type: {context.content_type.value}")
        print(f"🐛 Is Debugging: {context.is_debugging}")
        print(f"🔧 Has Technical Terms: {context.has_technical_terms}")
        
        response_lang = self.manager.determine_response_language(context)
        print(f"🗣️ Response Language: {response_lang.value}")
        
        prompt = self.manager.create_bilingual_prompt(user_input, context)
        print(f"💬 Prompt Contains English Instructions: {'Respond in English' in prompt}")
        
        # Assertions
        assert context.user_query_language == Language.ENGLISH
        assert context.is_debugging is True
        assert context.has_technical_terms is True
        assert response_lang == Language.ENGLISH
        assert "Respond in English" in prompt
        
        print("✅ Scenario 2 PASSED: English technical query handled correctly\n")
    
    def test_scenario_3_mixed_language_code_request(self):
        """Test Scenario 3: Arabic query with code/file references."""
        print("📝 Test Scenario 3: Mixed Language (Arabic + Code)")
        
        user_input = "كيف أصلح خطأ في ملف core_engine.py؟"
        print(f"👤 User Input: {user_input}")
        
        context = self.manager.build_language_context(user_input)
        print(f"🔍 Detected Language: {context.user_query_language.value}")
        print(f"📂 Content Type: {context.content_type.value}")
        print(f"📁 Has File Paths: {context.has_file_paths}")
        
        response_lang = self.manager.determine_response_language(context)
        print(f"🗣️ Response Language: {response_lang.value}")
        
        # Test response post-processing
        mock_response = "لإصلاح الخطأ، افتح ملف `core_engine.py` وتحقق من دالة `process_request`"
        processed = self.manager.process_response(mock_response, context)
        print(f"🔄 Processed Response: {processed}")
        
        # Assertions
        assert context.user_query_language == Language.ARABIC
        assert context.has_file_paths is True
        assert response_lang == Language.MIXED  # Should be MIXED for Arabic query with file paths
        assert "`core_engine.py`" in processed
        assert "`process_request`" in processed
        
        print("✅ Scenario 3 PASSED: Mixed language handling works correctly\n")
    
    def test_scenario_4_project_structure_discussion(self):
        """Test Scenario 4: Project structure discussion (should be English)."""
        print("📝 Test Scenario 4: Project Structure Discussion")
        
        user_input = "كيف هيكل المشروع في هذا repository؟"
        print(f"👤 User Input: {user_input}")
        
        context = self.manager.build_language_context(user_input)
        print(f"🔍 Detected Language: {context.user_query_language.value}")
        print(f"📂 Content Type: {context.content_type.value}")
        print(f"🏗️ Is Project Discussion: {context.is_project_discussion}")
        
        response_lang = self.manager.determine_response_language(context)
        print(f"🗣️ Response Language: {response_lang.value}")
        
        # Assertions
        assert context.user_query_language == Language.ARABIC
        assert context.is_project_discussion is True
        assert response_lang == Language.ENGLISH  # Should switch to English for project structure
        
        print("✅ Scenario 4 PASSED: Project structure forces English response\n")
    
    def test_scenario_5_api_reference_request(self):
        """Test Scenario 5: API reference request."""
        print("📝 Test Scenario 5: API Reference Request")
        
        user_input = "كيف أستخدم REST API endpoint للمصادقة؟"
        print(f"👤 User Input: {user_input}")
        
        context = self.manager.build_language_context(user_input)
        print(f"🔍 Detected Language: {context.user_query_language.value}")
        print(f"📂 Content Type: {context.content_type.value}")
        print(f"🔌 Has Technical Terms: {context.has_technical_terms}")
        
        response_lang = self.manager.determine_response_language(context)
        print(f"🗣️ Response Language: {response_lang.value}")
        
        # Test English term extraction
        terms = self.manager.extract_english_terms(user_input)
        print(f"🏷️ Extracted English Terms: {terms}")
        
        # Assertions
        assert context.user_query_language == Language.ARABIC
        assert context.content_type == ContentType.API_REFERENCE  # Should detect as API reference
        assert response_lang == Language.ENGLISH  # API references should be in English
        assert "REST" in terms or "API" in terms
        
        print("✅ Scenario 5 PASSED: API reference handled correctly\n")
    
    def test_scenario_6_code_snippet_help(self):
        """Test Scenario 6: Code snippet help."""
        print("📝 Test Scenario 6: Code Snippet Help")
        
        user_input = "ساعدني في كتابة دالة async def process_data():"
        print(f"👤 User Input: {user_input}")
        
        context = self.manager.build_language_context(user_input)
        print(f"🔍 Detected Language: {context.user_query_language.value}")
        print(f"📂 Content Type: {context.content_type.value}")
        print(f"💻 Has Code: {context.has_code}")
        
        response_lang = self.manager.determine_response_language(context)
        print(f"🗣️ Response Language: {response_lang.value}")
        
        # Assertions
        assert context.user_query_language == Language.ARABIC
        assert context.content_type == ContentType.CODE_SNIPPET
        assert context.has_code is True
        assert response_lang == Language.ENGLISH  # Code should be in English
        
        print("✅ Scenario 6 PASSED: Code snippet forces English response\n")
    
    def test_scenario_7_language_statistics(self):
        """Test Scenario 7: Language configuration statistics."""
        print("📝 Test Scenario 7: Language Configuration Statistics")
        
        stats = self.manager.get_language_stats()
        print(f"📊 Primary Language: {stats['primary_language']}")
        print(f"📊 Secondary Language: {stats['secondary_language']}")
        print(f"📊 Mixing Enabled: {stats['mixing_enabled']}")
        print(f"📊 Technical Domain: {stats['technical_domain']}")
        print(f"📊 Arabic Content Types: {stats['arabic_content_types']}")
        print(f"📊 English Content Types: {stats['english_content_types']}")
        
        # Assertions
        assert stats["primary_language"] == "ar"
        assert stats["secondary_language"] == "en"
        assert stats["mixing_enabled"] is True
        assert stats["technical_domain"] == "AI/ML systems"
        assert "general_conversation" in stats["arabic_content_types"]
        assert "code_snippet" in stats["english_content_types"]
        
        print("✅ Scenario 7 PASSED: Language statistics are correct\n")
    
    def run_all_tests(self):
        """Run all integration test scenarios."""
        print("🚀 Starting Bilingual AI Assistant Integration Tests\n")
        print("=" * 60)
        
        test_methods = [
            self.test_scenario_1_arabic_general_conversation,
            self.test_scenario_2_english_technical_query,
            self.test_scenario_3_mixed_language_code_request,
            self.test_scenario_4_project_structure_discussion,
            self.test_scenario_5_api_reference_request,
            self.test_scenario_6_code_snippet_help,
            self.test_scenario_7_language_statistics
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
            except Exception as e:
                print(f"❌ {test_method.__name__} FAILED: {str(e)}\n")
                failed += 1
        
        print("=" * 60)
        print(f"🏁 Integration Tests Complete!")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Bilingual AI Assistant is working correctly! 🎉")
        else:
            print(f"\n⚠️ {failed} test(s) failed. Please review the implementation.")
        
        return failed == 0


def main():
    """Main function to run the integration tests."""
    try:
        test_suite = BilingualIntegrationTest()
        success = test_suite.run_all_tests()
        
        if success:
            print("\n🌟 Integration Test Summary:")
            print("   • Arabic language detection works correctly")
            print("   • English language detection works correctly")
            print("   • Content type classification is accurate")
            print("   • Language mixing rules are properly applied")
            print("   • Technical content triggers appropriate language")
            print("   • Code and file paths are handled correctly")
            print("   • Configuration and statistics are accessible")
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"\n💥 Integration test failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())