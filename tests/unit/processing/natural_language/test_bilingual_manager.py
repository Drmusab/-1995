"""
Tests for Bilingual AI Assistant Manager
Author: AI Assistant
Last Modified: 2025-01-01

This module contains comprehensive tests for the bilingual language processing
functionality, including language detection, content type classification, and
language mixing capabilities.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.processing.natural_language.bilingual_manager import (
    BilingualManager,
    BilingualConfig,
    Language,
    ContentType,
    LanguageContext
)
from src.core.dependency_injection import Container
from src.core.config.loader import ConfigLoader


class TestBilingualManager:
    """Test cases for the BilingualManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        # Mock dependencies
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
        
        # Create manager instance
        self.manager = BilingualManager(self.container)
    
    def test_language_detection_arabic(self):
        """Test Arabic language detection."""
        arabic_text = "مرحبا كيف يمكنني مساعدتك اليوم؟"
        detected = self.manager.detect_language(arabic_text)
        assert detected == Language.ARABIC
    
    def test_language_detection_english(self):
        """Test English language detection."""
        english_text = "Hello, how can I help you today?"
        detected = self.manager.detect_language(english_text)
        assert detected == Language.ENGLISH
    
    def test_language_detection_mixed(self):
        """Test mixed language detection with English triggers."""
        mixed_text = "Can you help me with تطوير البرمجيات?"
        detected = self.manager.detect_language(mixed_text)
        assert detected == Language.ENGLISH  # English trigger "can you"
    
    def test_content_type_detection_code(self):
        """Test code snippet detection."""
        code_text = "Please check the function def process_request():"
        content_type = self.manager.detect_content_type(code_text)
        assert content_type == ContentType.CODE_SNIPPET
    
    def test_content_type_detection_file_path(self):
        """Test file path detection."""
        file_text = "Please modify the src/assistant/core_engine.py file"
        content_type = self.manager.detect_content_type(file_text)
        assert content_type == ContentType.FILE_PATH
    
    def test_content_type_detection_technical_term(self):
        """Test technical term detection."""
        tech_text = "The FastAPI endpoint is not responding"
        content_type = self.manager.detect_content_type(tech_text)
        assert content_type == ContentType.TECHNICAL_TERM
    
    def test_content_type_detection_api(self):
        """Test API reference detection."""
        api_text = "How do I use the REST API endpoint?"
        content_type = self.manager.detect_content_type(api_text)
        assert content_type == ContentType.API_REFERENCE
    
    def test_content_type_detection_debugging(self):
        """Test debugging content detection."""
        debug_text = "I'm getting an error in the application"
        content_type = self.manager.detect_content_type(debug_text)
        assert content_type == ContentType.DEBUGGING
    
    def test_content_type_detection_general(self):
        """Test general conversation detection."""
        general_text = "What is artificial intelligence?"
        content_type = self.manager.detect_content_type(general_text)
        assert content_type == ContentType.GENERAL_CONVERSATION
    
    def test_language_context_building(self):
        """Test building comprehensive language context."""
        text = "How do I fix the error in core_engine.py?"
        context = self.manager.build_language_context(text)
        
        assert context.user_query_language == Language.ENGLISH
        assert context.content_type == ContentType.DEBUGGING
        assert context.has_file_paths is True
        assert context.is_debugging is True
        assert context.confidence > 0.5
    
    def test_response_language_determination_english_query(self):
        """Test language determination for English queries."""
        context = LanguageContext(
            user_query_language=Language.ENGLISH,
            content_type=ContentType.GENERAL_CONVERSATION
        )
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.ENGLISH
    
    def test_response_language_determination_arabic_general(self):
        """Test language determination for Arabic general conversation."""
        context = LanguageContext(
            user_query_language=Language.ARABIC,
            content_type=ContentType.GENERAL_CONVERSATION
        )
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.ARABIC
    
    def test_response_language_determination_mixed_technical(self):
        """Test language determination for technical content with Arabic query."""
        context = LanguageContext(
            user_query_language=Language.ARABIC,
            content_type=ContentType.GENERAL_CONVERSATION,
            has_code=True,
            has_technical_terms=True
        )
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.MIXED
    
    def test_response_language_determination_debugging(self):
        """Test language determination for debugging scenarios."""
        context = LanguageContext(
            user_query_language=Language.ARABIC,
            content_type=ContentType.DEBUGGING,
            is_debugging=True
        )
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.ENGLISH
    
    def test_mixed_response_formatting(self):
        """Test formatting of mixed language responses."""
        arabic_text = "يرجى تعديل ملف {core_engine.py} في دالة {process_request}"
        english_terms = ["core_engine.py", "process_request"]
        
        formatted = self.manager.format_mixed_response(arabic_text, english_terms)
        expected = "يرجى تعديل ملف `core_engine.py` في دالة `process_request`"
        assert formatted == expected
    
    def test_bilingual_prompt_creation_arabic(self):
        """Test bilingual prompt creation for Arabic responses."""
        user_input = "ما هو الذكاء الاصطناعي؟"
        context = LanguageContext(
            user_query_language=Language.ARABIC,
            content_type=ContentType.GENERAL_CONVERSATION
        )
        
        prompt = self.manager.create_bilingual_prompt(user_input, context)
        assert "العربية الفصحى الحديثة" in prompt
        assert user_input in prompt
    
    def test_bilingual_prompt_creation_english(self):
        """Test bilingual prompt creation for English responses."""
        user_input = "How do I debug the core engine?"
        context = LanguageContext(
            user_query_language=Language.ENGLISH,
            content_type=ContentType.DEBUGGING,
            is_debugging=True
        )
        
        prompt = self.manager.create_bilingual_prompt(user_input, context)
        assert "Respond in English" in prompt
        assert user_input in prompt
    
    def test_english_terms_extraction(self):
        """Test extraction of English terms from text."""
        text = "يرجى استخدام `FastAPI` مع `PyTorch` في src/models/"
        terms = self.manager.extract_english_terms(text)
        
        expected_terms = ["FastAPI", "PyTorch"]
        for term in expected_terms:
            assert term in terms
    
    def test_response_post_processing(self):
        """Test post-processing of generated responses."""
        response_text = "لتنفيذ هذا، استخدم `def process_request()` في `core_engine.py`"
        context = LanguageContext(
            user_query_language=Language.ARABIC,
            content_type=ContentType.GENERAL_CONVERSATION,
            has_code=True,
            has_file_paths=True
        )
        
        processed = self.manager.process_response(response_text, context)
        assert "`def process_request()`" in processed
        assert "`core_engine.py`" in processed
    
    def test_language_stats(self):
        """Test language configuration statistics."""
        stats = self.manager.get_language_stats()
        
        assert stats["primary_language"] == "ar"
        assert stats["secondary_language"] == "en"
        assert stats["mixing_enabled"] is True
        assert "general_conversation" in stats["arabic_content_types"]
        assert "code_snippet" in stats["english_content_types"]
    
    def test_config_loading(self):
        """Test configuration loading and validation."""
        # Test with custom config
        custom_config = {
            "primary_language": "ar",
            "secondary_language": "en",
            "enable_mixing": False,
            "technical_domain": "Custom Domain"
        }
        self.config_loader.get.return_value = custom_config
        
        manager = BilingualManager(self.container)
        assert manager.config.primary_language == Language.ARABIC
        assert manager.config.enable_mixing is False
        assert manager.config.technical_domain == "Custom Domain"


class TestBilingualConfig:
    """Test cases for BilingualConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BilingualConfig()
        
        assert config.primary_language == Language.ARABIC
        assert config.secondary_language == Language.ENGLISH
        assert config.enable_mixing is True
        assert ContentType.GENERAL_CONVERSATION in config.use_arabic_for
        assert ContentType.CODE_SNIPPET in config.use_english_for
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = BilingualConfig(
            primary_language=Language.ENGLISH,
            enable_mixing=False,
            technical_domain="Custom Domain"
        )
        
        assert config.primary_language == Language.ENGLISH
        assert config.enable_mixing is False
        assert config.technical_domain == "Custom Domain"


class TestLanguageContext:
    """Test cases for LanguageContext dataclass."""
    
    def test_context_creation(self):
        """Test language context creation."""
        context = LanguageContext(
            user_query_language=Language.ARABIC,
            content_type=ContentType.CODE_SNIPPET,
            has_code=True,
            confidence=0.9
        )
        
        assert context.user_query_language == Language.ARABIC
        assert context.content_type == ContentType.CODE_SNIPPET
        assert context.has_code is True
        assert context.confidence == 0.9


# Integration tests
class TestBilingualIntegration:
    """Integration tests for bilingual functionality."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.container = Mock(spec=Container)
        self.config_loader = Mock(spec=ConfigLoader)
        self.config_loader.get.return_value = {}
        self.container.get.return_value = self.config_loader
        
        self.manager = BilingualManager(self.container)
    
    def test_full_processing_pipeline_arabic(self):
        """Test full processing pipeline for Arabic input."""
        user_input = "كيف يمكنني تحسين أداء النظام؟"
        
        # Build context
        context = self.manager.build_language_context(user_input)
        assert context.user_query_language == Language.ARABIC
        assert context.content_type == ContentType.GENERAL_CONVERSATION
        
        # Determine response language
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.ARABIC
        
        # Create prompt
        prompt = self.manager.create_bilingual_prompt(user_input, context)
        assert "العربية" in prompt
        assert user_input in prompt
    
    def test_full_processing_pipeline_technical(self):
        """Test full processing pipeline for technical content."""
        user_input = "There's an error in the FastAPI endpoint"
        
        # Build context
        context = self.manager.build_language_context(user_input)
        assert context.user_query_language == Language.ENGLISH
        assert context.has_technical_terms is True
        
        # Determine response language
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.ENGLISH
        
        # Create prompt
        prompt = self.manager.create_bilingual_prompt(user_input, context)
        assert "English" in prompt
    
    def test_full_processing_pipeline_mixed(self):
        """Test full processing pipeline for mixed language scenario."""
        user_input = "كيف أصلح خطأ في core_engine.py؟"
        
        # Build context
        context = self.manager.build_language_context(user_input)
        assert context.user_query_language == Language.ARABIC
        assert context.has_file_paths is True
        
        # Determine response language (should be mixed due to technical content)
        response_lang = self.manager.determine_response_language(context)
        assert response_lang == Language.MIXED
        
        # Process a mock response
        mock_response = "لإصلاح الخطأ، افتح ملف {core_engine.py} وتحقق من دالة {process_request}"
        processed = self.manager.process_response(mock_response, context)
        assert "`core_engine.py`" in processed