"""
Bilingual AI Assistant Manager
Author: AI Assistant
Last Modified: 2025-01-01

This module provides bilingual language support for the AI assistant, implementing
Arabic/English language switching, mixing rules, and context-aware language selection.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger


class Language(Enum):
    """Supported languages."""
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"


class ContentType(Enum):
    """Types of content for language selection."""
    GENERAL_CONVERSATION = "general_conversation"
    CODE_SNIPPET = "code_snippet"
    FILE_PATH = "file_path"
    TECHNICAL_TERM = "technical_term"
    API_REFERENCE = "api_reference"
    PROJECT_STRUCTURE = "project_structure"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    USER_GUIDANCE = "user_guidance"


@dataclass
class BilingualConfig:
    """Configuration for bilingual assistant behavior."""
    
    primary_language: Language = Language.ARABIC
    secondary_language: Language = Language.ENGLISH
    
    # Language usage rules
    use_arabic_for: Set[ContentType] = field(default_factory=lambda: {
        ContentType.GENERAL_CONVERSATION,
        ContentType.EXPLANATION,
        ContentType.USER_GUIDANCE
    })
    
    use_english_for: Set[ContentType] = field(default_factory=lambda: {
        ContentType.CODE_SNIPPET,
        ContentType.FILE_PATH,
        ContentType.TECHNICAL_TERM,
        ContentType.API_REFERENCE
    })
    
    # Switch to full English when
    full_english_triggers: Set[str] = field(default_factory=lambda: {
        "project_structure",
        "debugging",
        "technical_discussion"
    })
    
    # Language mixing settings
    enable_mixing: bool = True
    english_embedding_pattern: str = r'`([^`]+)`'  # Pattern for embedding English terms
    
    # Project context
    technical_domain: str = "AI/ML systems"
    codebase_language: str = "English"
    key_components: List[str] = field(default_factory=lambda: [
        "core_engine.py",
        "session_manager.py", 
        "llm integrations"
    ])


@dataclass
class LanguageContext:
    """Context for language processing decisions."""
    
    user_query_language: Language
    content_type: ContentType
    has_code: bool = False
    has_file_paths: bool = False
    has_technical_terms: bool = False
    is_debugging: bool = False
    is_project_discussion: bool = False
    confidence: float = 0.0


class BilingualManager:
    """
    Manages bilingual language processing for the AI assistant.
    
    Implements the language rules:
    1. Primary Language: Arabic (Modern Standard Arabic)
    2. Secondary Language: English  
    3. Use Arabic for general conversation, explanations, user guidance
    4. Use English for code snippets, file paths, technical terms, API references
    5. Language mixing for embedding English terms in Arabic sentences
    6. Switch to full English for English queries, project structure, debugging
    """
    
    def __init__(self, container: Container):
        """Initialize the bilingual manager."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config_loader = container.get(ConfigLoader)
        config_data = self.config_loader.get("bilingual", {})
        
        # Convert string values to enums if needed
        if "primary_language" in config_data and isinstance(config_data["primary_language"], str):
            config_data["primary_language"] = Language(config_data["primary_language"])
        if "secondary_language" in config_data and isinstance(config_data["secondary_language"], str):
            config_data["secondary_language"] = Language(config_data["secondary_language"])
        
        self.config = BilingualConfig(**config_data)
        
        # Language detection patterns
        self._setup_language_patterns()
        
        # Technical terms and code patterns
        self._setup_technical_patterns()
        
        self.logger.info("BilingualManager initialized with Arabic/English support")
    
    def _setup_language_patterns(self) -> None:
        """Setup patterns for language detection."""
        # Arabic text detection (Unicode ranges for Arabic script)
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        
        # English text detection
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
        # Mixed language indicators
        self.code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'function', 'var ', 'const ',
            'async ', 'await ', 'return', '()', '{', '}', ';', '->', '=>'
        ]
        
        # File path indicators
        self.file_path_indicators = [
            '/', '\\', '.py', '.js', '.json', '.yaml', '.yml', '.md', '.txt',
            'src/', 'config/', 'data/', '__init__.py'
        ]
        
        # Technical term indicators
        self.technical_indicators = [
            'API', 'HTTP', 'JSON', 'REST', 'GraphQL', 'WebSocket', 'JWT',
            'FastAPI', 'PyTorch', 'TensorFlow', 'Docker', 'Kubernetes',
            'asyncio', 'async/await', 'middleware', 'endpoint'
        ]
    
    def _setup_technical_patterns(self) -> None:
        """Setup patterns for technical content detection."""
        # Project structure keywords
        self.project_keywords = [
            'repository', 'codebase', 'architecture', 'module', 'component',
            'directory', 'folder', 'package', 'namespace', 'dependency'
        ]
        
        # Debugging keywords  
        self.debugging_keywords = [
            'error', 'exception', 'debug', 'trace', 'log', 'bug', 'issue',
            'troubleshoot', 'diagnose', 'fix', 'resolve'
        ]
        
        # English query triggers
        self.english_triggers = [
            'how to', 'what is', 'can you', 'please help', 'explain',
            'show me', 'debug', 'error', 'issue', 'problem'
        ]
    
    def detect_language(self, text: str) -> Language:
        """
        Detect the primary language of the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language
        """
        if not text or not text.strip():
            return self.config.primary_language
        
        # Count Arabic and English characters
        arabic_chars = len(self.arabic_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        # Simple heuristic based on character counts
        if arabic_chars > english_chars:
            return Language.ARABIC
        elif english_chars > arabic_chars:
            return Language.ENGLISH
        else:
            # Mixed or equal - check for specific patterns
            if any(trigger.lower() in text.lower() for trigger in self.english_triggers):
                return Language.ENGLISH
            return Language.ARABIC
    
    def detect_content_type(self, text: str) -> ContentType:
        """
        Detect the type of content to determine appropriate language.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected content type
        """
        text_lower = text.lower()
        
        # Check for file paths first (highest priority for mixed language)
        if any(indicator in text for indicator in self.file_path_indicators):
            return ContentType.FILE_PATH
        
        # Check for code snippets
        if any(indicator in text for indicator in self.code_indicators):
            return ContentType.CODE_SNIPPET
        
        # Check for debugging content (higher priority than technical terms)
        if any(keyword in text_lower for keyword in self.debugging_keywords):
            return ContentType.DEBUGGING
        
        # Check for project structure discussion
        if any(keyword in text_lower for keyword in self.project_keywords):
            return ContentType.PROJECT_STRUCTURE
        
        # Check for API references
        if 'api' in text_lower and ('endpoint' in text_lower or 'rest' in text_lower or 'graphql' in text_lower):
            return ContentType.API_REFERENCE
        
        # Check for technical terms
        if any(term.lower() in text_lower for term in self.technical_indicators):
            return ContentType.TECHNICAL_TERM
        
        # Default to general conversation
        return ContentType.GENERAL_CONVERSATION
    
    def build_language_context(self, text: str) -> LanguageContext:
        """
        Build comprehensive language context for processing decisions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language context with detection results
        """
        user_language = self.detect_language(text)
        content_type = self.detect_content_type(text)
        
        context = LanguageContext(
            user_query_language=user_language,
            content_type=content_type,
            has_code=any(indicator in text for indicator in self.code_indicators),
            has_file_paths=any(indicator in text for indicator in self.file_path_indicators),
            has_technical_terms=any(term.lower() in text.lower() 
                                  for term in self.technical_indicators),
            is_debugging=any(keyword in text.lower() 
                           for keyword in self.debugging_keywords),
            is_project_discussion=any(keyword in text.lower() 
                                    for keyword in self.project_keywords)
        )
        
        # Calculate confidence based on clear indicators
        confidence_factors = []
        if context.has_code:
            confidence_factors.append(0.9)
        if context.has_file_paths:
            confidence_factors.append(0.8)
        if context.has_technical_terms:
            confidence_factors.append(0.7)
        if context.is_debugging:
            confidence_factors.append(0.8)
        if context.is_project_discussion:
            confidence_factors.append(0.7)
        
        context.confidence = max(confidence_factors) if confidence_factors else 0.5
        
        return context
    
    def determine_response_language(self, context: LanguageContext) -> Language:
        """
        Determine the appropriate language for the response.
        
        Args:
            context: Language context from analysis
            
        Returns:
            Language to use for response
        """
        # Rule 6: Switch to full English for English queries
        if context.user_query_language == Language.ENGLISH:
            return Language.ENGLISH
        
        # Rule 6: Switch to full English for debugging scenarios
        if context.is_debugging:
            return Language.ENGLISH
        
        # Rule 6: Switch to full English for project structure discussion
        if context.is_project_discussion:
            return Language.ENGLISH
        
        # Special case: Arabic query asking about files should use mixed language
        if (context.user_query_language == Language.ARABIC and 
            context.content_type == ContentType.FILE_PATH):
            return Language.MIXED
        
        # Rule 4: Use English for pure technical content types (but not if Arabic query with code/files)
        if context.content_type in self.config.use_english_for:
            return Language.ENGLISH
        
        # Rule 5: Use mixed language if Arabic query but has technical/code elements
        if (context.user_query_language == Language.ARABIC and 
            context.content_type in self.config.use_arabic_for and
            (context.has_code or context.has_file_paths or context.has_technical_terms)):
            return Language.MIXED
        
        # Rule 3: Use Arabic for general conversation, explanations, guidance
        if context.content_type in self.config.use_arabic_for:
            return Language.ARABIC
        
        # Default to primary language
        return self.config.primary_language
    
    def format_mixed_response(self, arabic_text: str, english_terms: List[str]) -> str:
        """
        Format a mixed language response with embedded English terms.
        
        Args:
            arabic_text: Main Arabic text
            english_terms: English terms to embed
            
        Returns:
            Formatted mixed language text
        """
        if not self.config.enable_mixing:
            return arabic_text
        
        # Embed English terms using backticks for code/technical terms
        formatted_text = arabic_text
        for term in english_terms:
            # Use the embedding pattern for technical terms
            if any(indicator in term for indicator in self.code_indicators + self.file_path_indicators):
                formatted_text = formatted_text.replace(f"{{{term}}}", f"`{term}`")
            else:
                formatted_text = formatted_text.replace(f"{{{term}}}", term)
        
        return formatted_text
    
    def create_bilingual_prompt(self, user_input: str, context: LanguageContext) -> str:
        """
        Create an appropriate prompt for the LLM based on language context.
        
        Args:
            user_input: User's input text
            context: Language context analysis
            
        Returns:
            Formatted prompt for the LLM
        """
        response_language = self.determine_response_language(context)
        
        # Base system instructions
        base_instructions = """أنت مساعد ذكي ثنائي اللغة يعمل في بيئة تقنية. التزم بقواعد اللغة التالية:
1. اللغة الأساسية: العربية (العربية الفصحى الحديثة)
2. اللغة الثانوية: الإنجليزية
3. استخدم العربية للمحادثة العامة والشروحات والإرشادات
4. استخدم الإنجليزية لأجزاء الكود ومسارات الملفات والمصطلحات التقنية ومراجع API
5. عند خلط اللغات: ادمج المصطلحات الإنجليزية في الجمل العربية مثل: 'يُرجى تعديل ملف `processor_manager.py`'
6. انتقل للإنجليزية الكاملة عند: الاستعلامات الإنجليزية، مناقشة هيكل المشروع، سيناريوهات تصحيح الأخطاء

السياق التقني:
- المجال التقني: أنظمة الذكاء الاصطناعي/التعلم الآلي  
- لغة قاعدة الكود: الإنجليزية
- المكونات الرئيسية: core_engine.py، session_manager.py، تكاملات LLM"""

        if response_language == Language.ENGLISH:
            base_instructions = """You are a bilingual AI assistant operating in a technical environment. Follow these language rules:
1. Primary Language: Arabic (Modern Standard Arabic)  
2. Secondary Language: English
3. Use Arabic for general conversation, explanations, user guidance
4. Use English for code snippets, file/directory paths, technical terms, API/library references
5. When mixing languages: Embed English terms in Arabic sentences like: 'يُرجى تعديل ملف `processor_manager.py`'
6. Switch to full English when: User queries are in English, discussing project structure, handling technical debugging scenarios

Technical context:
- Technical domain: AI/ML systems
- Codebase language: English  
- Key components: core_engine.py, session_manager.py, LLM integrations

Respond in English for this query."""

        elif response_language == Language.MIXED:
            base_instructions += "\n\nللاستجابة الحالية: استخدم العربية كلغة أساسية مع دمج المصطلحات التقنية والكود بالإنجليزية باستخدام علامات التشفير `مثل هذا`."
        
        # Build the complete prompt
        prompt_parts = [
            base_instructions,
            f"\nاستعلام المستخدم: {user_input}",
            f"\nنوع المحتوى المكتشف: {context.content_type.value}",
            f"اللغة المطلوبة للاستجابة: {response_language.value}"
        ]
        
        return "\n".join(prompt_parts)
    
    def extract_english_terms(self, text: str) -> List[str]:
        """
        Extract English terms that should be preserved in mixed responses.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of English terms to preserve
        """
        terms = []
        
        # Extract code snippets
        code_matches = re.findall(r'`([^`]+)`', text)
        terms.extend(code_matches)
        
        # Extract file paths  
        for indicator in self.file_path_indicators:
            if indicator in text:
                # Find words containing the file extension
                words = text.split()
                for word in words:
                    if indicator in word:
                        terms.append(word)
        
        # Extract technical terms
        for term in self.technical_indicators:
            if term.lower() in text.lower():
                terms.append(term)
        
        return list(set(terms))  # Remove duplicates
    
    def process_response(self, response_text: str, context: LanguageContext) -> str:
        """
        Post-process the response to ensure proper language formatting.
        
        Args:
            response_text: Generated response text
            context: Original language context
            
        Returns:
            Post-processed response
        """
        if not response_text:
            return response_text
        
        response_language = self.determine_response_language(context)
        
        if response_language == Language.MIXED:
            # Ensure English terms are properly formatted
            english_terms = self.extract_english_terms(response_text)
            if english_terms:
                return self.format_mixed_response(response_text, english_terms)
        
        return response_text
    
    def get_language_stats(self) -> Dict[str, Any]:
        """
        Get statistics about language usage and configuration.
        
        Returns:
            Language configuration and usage statistics
        """
        return {
            "primary_language": self.config.primary_language.value,
            "secondary_language": self.config.secondary_language.value,
            "mixing_enabled": self.config.enable_mixing,
            "arabic_content_types": [ct.value for ct in self.config.use_arabic_for],
            "english_content_types": [ct.value for ct in self.config.use_english_for],
            "technical_domain": self.config.technical_domain,
            "key_components": self.config.key_components,
            "full_english_triggers": list(self.config.full_english_triggers)
        }