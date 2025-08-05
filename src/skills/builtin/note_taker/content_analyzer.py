"""
Content Analyzer for Note Taking
Author: Drmusab
Last Modified: 2025-01-08

Analyzes note content for categorization, action item extraction, definition detection,
and tagging. Integrates with existing NLP components.
"""

import re
from typing import Any, Dict, List, Optional

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain

from .note_taker_skill import NoteCategory


class ContentAnalyzer:
    """
    Analyzes note content for categorization, action items, definitions, and tagging.
    
    Integrates with existing NLP components to provide intelligent content analysis
    for the note-taking system.
    """
    
    def __init__(self, container: Container):
        """Initialize the content analyzer."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        
        # NLP components
        self.bilingual_manager = container.get(BilingualManager)
        self.entity_extractor = container.get_optional(EntityExtractor)
        self.intent_manager = container.get_optional(IntentManager)
        self.language_chain = container.get_optional(LanguageChain)
        
        # Setup patterns and keywords
        self._setup_analysis_patterns()
        
        self.logger.info("ContentAnalyzer initialized")
    
    def _setup_analysis_patterns(self) -> None:
        """Setup patterns for content analysis."""
        # Action item patterns (Arabic and English)
        self.action_patterns = [
            # English patterns
            r'\b(need to|should|must|have to|will|going to|plan to|todo|action)\s+([^.!?]+)',
            r'\b(follow up|follow-up|check|verify|confirm|schedule|call|email|send)\s+([^.!?]+)',
            r'\b(complete|finish|deliver|submit|review|update|create|prepare)\s+([^.!?]+)',
            
            # Arabic patterns
            r'\b(يجب|ينبغي|لازم|سوف|سأ|نحتاج|يلزم)\s+([^.!؟]+)',
            r'\b(متابعة|مراجعة|تأكد|اتصال|إرسال|إنجاز|تحضير)\s+([^.!؟]+)',
            r'\b(إكمال|انتهاء|تسليم|إرسال|مراجعة|تحديث|إنشاء)\s+([^.!؟]+)'
        ]
        
        # Definition patterns
        self.definition_patterns = [
            # English patterns
            r'([A-Z][a-zA-Z\s]+)\s+(?:is|means|refers to|defined as)\s+([^.!?]+)',
            r'([A-Z][a-zA-Z\s]+):\s*([^.!?]+)',
            r'(?:definition|meaning|concept)\s+of\s+([^:]+):\s*([^.!?]+)',
            
            # Arabic patterns
            r'([أ-ي\s]+)\s+(?:هو|هي|يعني|يشير إلى|يُعرَّف)\s+([^.!؟]+)',
            r'([أ-ي\s]+):\s*([^.!؟]+)',
            r'(?:تعريف|معنى|مفهوم)\s+([^:]+):\s*([^.!؟]+)'
        ]
        
        # Category keywords
        self.category_keywords = {
            NoteCategory.TASKS: {
                'english': ['task', 'todo', 'assignment', 'deadline', 'complete', 'finish', 'work', 'project'],
                'arabic': ['مهمة', 'عمل', 'تكليف', 'موعد', 'إنجاز', 'مشروع', 'واجب']
            },
            NoteCategory.IDEAS: {
                'english': ['idea', 'concept', 'thought', 'innovation', 'brainstorm', 'suggestion', 'proposal'],
                'arabic': ['فكرة', 'مفهوم', 'اقتراح', 'ابتكار', 'عصف ذهني', 'طرح']
            },
            NoteCategory.QUOTES: {
                'english': ['said', 'quote', 'mentioned', 'stated', 'according to', 'as per'],
                'arabic': ['قال', 'ذكر', 'اقتباس', 'صرح', 'حسب', 'وفقاً']
            },
            NoteCategory.MEMORY: {
                'english': ['remember', 'recall', 'memory', 'remind', 'note down', 'important'],
                'arabic': ['تذكر', 'ذاكرة', 'تذكير', 'مهم', 'احفظ', 'لا تنس']
            },
            NoteCategory.EXPERIENCE: {
                'english': ['experience', 'happened', 'event', 'story', 'incident', 'today', 'yesterday'],
                'arabic': ['تجربة', 'حدث', 'واقعة', 'قصة', 'اليوم', 'أمس', 'حصل']
            },
            NoteCategory.MEETING_NOTES: {
                'english': ['meeting', 'discussion', 'agenda', 'minutes', 'attendees', 'decisions'],
                'arabic': ['اجتماع', 'مناقشة', 'جدول أعمال', 'محضر', 'حاضرين', 'قرارات']
            },
            NoteCategory.STUDY_NOTES: {
                'english': ['study', 'learn', 'course', 'lecture', 'lesson', 'chapter', 'exam', 'test'],
                'arabic': ['دراسة', 'تعلم', 'دورة', 'محاضرة', 'درس', 'فصل', 'امتحان', 'اختبار']
            },
            NoteCategory.TECHNICAL: {
                'english': ['code', 'programming', 'development', 'software', 'algorithm', 'function', 'api'],
                'arabic': ['برمجة', 'تطوير', 'برنامج', 'خوارزمية', 'دالة', 'كود', 'تقني']
            }
        }
        
        # Common tag patterns
        self.tag_patterns = [
            r'#(\w+)',  # Hashtags
            r'@(\w+)',  # Mentions
            r'\b(important|urgent|todo|follow-up|deadline)\b',  # English keywords
            r'\b(مهم|عاجل|متابعة|موعد|تذكير)\b'  # Arabic keywords
        ]
    
    async def categorize_content(self, text: str) -> NoteCategory:
        """
        Categorize note content based on keywords and patterns.
        
        Args:
            text: Text content to categorize
            
        Returns:
            Detected note category
        """
        if not text or not text.strip():
            return NoteCategory.GENERAL
        
        text_lower = text.lower()
        
        # Calculate scores for each category
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            
            # Check English keywords
            for keyword in keywords['english']:
                if keyword.lower() in text_lower:
                    score += 1
            
            # Check Arabic keywords
            for keyword in keywords['arabic']:
                if keyword in text:
                    score += 1
            
            category_scores[category] = score
        
        # Find category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        # Use intent detection if available
        if self.intent_manager:
            try:
                intent = await self.intent_manager.extract_intent(text)
                if intent and intent.get('category'):
                    # Map intent to note category
                    intent_category = self._map_intent_to_category(intent['category'])
                    if intent_category:
                        return intent_category
            except Exception as e:
                self.logger.warning(f"Intent detection failed: {str(e)}")
        
        return NoteCategory.GENERAL
    
    def _map_intent_to_category(self, intent_category: str) -> Optional[NoteCategory]:
        """Map intent category to note category."""
        intent_mapping = {
            'task': NoteCategory.TASKS,
            'idea': NoteCategory.IDEAS,
            'memory': NoteCategory.MEMORY,
            'personal': NoteCategory.PERSONAL,
            'technical': NoteCategory.TECHNICAL,
            'study': NoteCategory.STUDY_NOTES,
            'meeting': NoteCategory.MEETING_NOTES
        }
        
        return intent_mapping.get(intent_category.lower())
    
    async def extract_action_items(self, text: str) -> List[str]:
        """
        Extract action items from note content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of extracted action items
        """
        action_items = []
        
        # Use regex patterns to find action items
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(2):
                    action_item = match.group(2).strip()
                    if action_item and len(action_item) > 3:  # Filter out very short matches
                        action_items.append(action_item)
        
        # Use language chain for more sophisticated extraction if available
        if self.language_chain and len(action_items) < 3:  # Only if few items found
            try:
                llm_actions = await self._extract_actions_with_llm(text)
                action_items.extend(llm_actions)
            except Exception as e:
                self.logger.warning(f"LLM action extraction failed: {str(e)}")
        
        # Remove duplicates and clean up
        unique_actions = []
        for action in action_items:
            cleaned_action = self._clean_extracted_text(action)
            if cleaned_action and cleaned_action not in unique_actions:
                unique_actions.append(cleaned_action)
        
        return unique_actions[:10]  # Limit to 10 action items
    
    async def _extract_actions_with_llm(self, text: str) -> List[str]:
        """Extract action items using LLM."""
        if not self.language_chain:
            return []
        
        # Determine language for appropriate prompt
        language_context = self.bilingual_manager.build_language_context(text)
        
        if language_context.user_query_language.value == 'ar':
            prompt = f"""استخرج عناصر العمل والمهام من النص التالي. أعطني قائمة بالمهام المحددة التي يجب القيام بها:

النص: {text}

اكتب فقط المهام المستخرجة، كل مهمة في سطر منفصل."""
        else:
            prompt = f"""Extract action items and tasks from the following text. Give me a list of specific tasks that need to be done:

Text: {text}

Write only the extracted tasks, one per line."""
        
        try:
            response = await self.language_chain.process_message(prompt)
            
            # Parse response into list
            lines = response.strip().split('\n')
            actions = []
            for line in lines:
                line = line.strip()
                # Remove bullet points, numbers, etc.
                line = re.sub(r'^[-•*\d+.)\s]+', '', line).strip()
                if line and len(line) > 5:
                    actions.append(line)
            
            return actions[:5]  # Limit LLM extracted actions
            
        except Exception as e:
            self.logger.warning(f"LLM action extraction error: {str(e)}")
            return []
    
    async def extract_definitions(self, text: str) -> Dict[str, str]:
        """
        Extract definitions and key concepts from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary of term -> definition mappings
        """
        definitions = {}
        
        # Use regex patterns to find definitions
        for pattern in self.definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    
                    if term and definition and len(definition) > 10:
                        # Clean up the term and definition
                        term = self._clean_extracted_text(term)
                        definition = self._clean_extracted_text(definition)
                        
                        if term and definition:
                            definitions[term] = definition
        
        # Use entity extraction to find technical terms
        if self.entity_extractor:
            try:
                entities = await self.entity_extractor.extract(text)
                for entity in entities:
                    if entity.get('label') in ['CONCEPT', 'TERM', 'TECHNOLOGY']:
                        term = entity.get('text', '').strip()
                        if term and term not in definitions:
                            # Try to find definition in surrounding context
                            definition = self._find_definition_context(text, term)
                            if definition:
                                definitions[term] = definition
            except Exception as e:
                self.logger.warning(f"Entity extraction for definitions failed: {str(e)}")
        
        return dict(list(definitions.items())[:10])  # Limit to 10 definitions
    
    def _find_definition_context(self, text: str, term: str) -> Optional[str]:
        """Find definition context for a term in the text."""
        # Look for term followed by explanatory text
        patterns = [
            f"{re.escape(term)}\\s*(?:is|means|refers to|:)\\s*([^.!?]+)",
            f"(?:definition of|meaning of)\\s*{re.escape(term)}\\s*[:-]\\s*([^.!?]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1):
                definition = match.group(1).strip()
                if len(definition) > 10:
                    return self._clean_extracted_text(definition)
        
        return None
    
    async def extract_tags(self, text: str) -> List[str]:
        """
        Extract tags and keywords from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of extracted tags
        """
        tags = set()
        
        # Extract hashtags and mentions
        for pattern in self.tag_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                tag = match.group(1) if match.groups() else match.group(0)
                tag = tag.strip('#@').lower()
                if tag and len(tag) > 2:
                    tags.add(tag)
        
        # Extract entities as tags
        if self.entity_extractor:
            try:
                entities = await self.entity_extractor.extract(text)
                for entity in entities:
                    if entity.get('label') in ['PERSON', 'ORG', 'TECHNOLOGY', 'CONCEPT']:
                        tag = entity.get('text', '').strip().lower()
                        if tag and len(tag) > 2:
                            tags.add(tag)
            except Exception as e:
                self.logger.warning(f"Entity extraction for tags failed: {str(e)}")
        
        # Add category-based tags
        category_tag = await self._get_category_tag(text)
        if category_tag:
            tags.add(category_tag)
        
        # Add language tag
        language_context = self.bilingual_manager.build_language_context(text)
        tags.add(f"lang_{language_context.user_query_language.value}")
        
        return sorted(list(tags))[:15]  # Limit to 15 tags
    
    async def _get_category_tag(self, text: str) -> Optional[str]:
        """Get category-based tag for the text."""
        category = await self.categorize_content(text)
        return category.value if category != NoteCategory.GENERAL else None
    
    async def generate_title(self, text: str, max_length: int = 50) -> str:
        """
        Generate a title for the note based on content.
        
        Args:
            text: Text content to analyze
            max_length: Maximum title length
            
        Returns:
            Generated title
        """
        if not text or not text.strip():
            return "Untitled Note"
        
        # Try to use first sentence if it's a good title
        sentences = re.split(r'[.!?؟]', text.strip())
        first_sentence = sentences[0].strip() if sentences else ""
        
        if first_sentence and len(first_sentence) <= max_length and len(first_sentence) > 5:
            return self._clean_extracted_text(first_sentence)
        
        # Use LLM to generate title if available
        if self.language_chain:
            try:
                llm_title = await self._generate_title_with_llm(text, max_length)
                if llm_title:
                    return llm_title
            except Exception as e:
                self.logger.warning(f"LLM title generation failed: {str(e)}")
        
        # Extract key phrases for title
        words = text.split()
        if len(words) > 5:
            # Take meaningful words from the beginning
            title_words = []
            for word in words[:10]:
                word = word.strip('.,!?؟').strip()
                if len(word) > 2 and not word.lower() in ['the', 'and', 'or', 'but', 'في', 'من', 'إلى', 'على']:
                    title_words.append(word)
                if len(' '.join(title_words)) > max_length - 10:
                    break
            
            if title_words:
                title = ' '.join(title_words)
                if len(title) <= max_length:
                    return title
        
        # Fallback: truncate first part of text
        truncated = text[:max_length].strip()
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]  # Remove partial word
        
        return truncated + "..." if len(text) > max_length else truncated
    
    async def _generate_title_with_llm(self, text: str, max_length: int) -> Optional[str]:
        """Generate title using LLM."""
        if not self.language_chain:
            return None
        
        # Determine language for appropriate prompt
        language_context = self.bilingual_manager.build_language_context(text)
        
        if language_context.user_query_language.value == 'ar':
            prompt = f"""اكتب عنواناً مختصراً ومناسباً للملاحظة التالية (أقل من {max_length} حرف):

النص: {text[:300]}...

العنوان:"""
        else:
            prompt = f"""Write a short and appropriate title for the following note (less than {max_length} characters):

Text: {text[:300]}...

Title:"""
        
        try:
            response = await self.language_chain.process_message(prompt)
            title = response.strip().strip('"\'')
            
            if title and len(title) <= max_length:
                return self._clean_extracted_text(title)
            
        except Exception as e:
            self.logger.warning(f"LLM title generation error: {str(e)}")
        
        return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common prefixes/suffixes
        text = re.sub(r'^(that|which|who|what|when|where|why|how)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(الذي|التي|الذين|اللذين|اللواتي|اللاتي|ما|من|متى|أين|كيف)\s+', '', text)
        
        # Remove trailing punctuation for action items
        text = re.sub(r'[,;:]$', '', text)
        
        return text.strip()
    
    async def analyze_content_quality(self, text: str) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of note content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Content quality metrics
        """
        if not text:
            return {"quality_score": 0.0, "issues": ["Empty content"]}
        
        analysis = {
            "word_count": len(text.split()),
            "character_count": len(text),
            "sentence_count": len(re.split(r'[.!?؟]', text)),
            "paragraph_count": len(text.split('\n\n')),
            "language_detected": None,
            "readability_score": 0.0,
            "quality_score": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # Detect language
        language_context = self.bilingual_manager.build_language_context(text)
        analysis["language_detected"] = language_context.user_query_language.value
        
        # Calculate basic quality metrics
        words = text.split()
        
        # Check for very short content
        if analysis["word_count"] < 5:
            analysis["issues"].append("Content is very short")
            analysis["quality_score"] -= 0.3
        
        # Check for very long sentences
        sentences = re.split(r'[.!?؟]', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length > 30:
            analysis["issues"].append("Sentences are too long")
            analysis["suggestions"].append("Consider breaking down long sentences")
        
        # Check for repetition
        unique_words = len(set(word.lower() for word in words))
        repetition_ratio = unique_words / max(len(words), 1)
        if repetition_ratio < 0.5:
            analysis["issues"].append("High word repetition")
            analysis["suggestions"].append("Try to use more varied vocabulary")
        
        # Calculate readability (simplified)
        analysis["readability_score"] = min(1.0, repetition_ratio * 1.2)
        
        # Overall quality score
        base_score = 0.7
        if not analysis["issues"]:
            base_score += 0.2
        if analysis["word_count"] >= 10:
            base_score += 0.1
        
        analysis["quality_score"] = max(0.0, min(1.0, base_score))
        
        return analysis