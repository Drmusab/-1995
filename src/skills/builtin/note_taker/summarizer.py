"""
Note Summarizer
Author: Drmusab
Last Modified: 2025-01-08

Provides text summarization and key point extraction for notes.
Integrates with existing language processing components.
"""

import re
from typing import Any, Dict, List, Optional

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.natural_language.language_chain import LanguageChain


class NoteSummarizer:
    """
    Provides intelligent summarization capabilities for note content.
    
    Features:
    - Extractive and abstractive summarization
    - Key point extraction
    - Bullet point generation
    - Timeline-based organization
    - Bilingual support (Arabic/English)
    """
    
    def __init__(self, container: Container):
        """Initialize the note summarizer."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        
        # NLP components
        self.bilingual_manager = container.get(BilingualManager)
        self.language_chain = container.get_optional(LanguageChain)
        
        # Configuration
        self._setup_summarization_config()
        
        self.logger.info("NoteSummarizer initialized")
    
    def _setup_summarization_config(self) -> None:
        """Setup summarization configuration."""
        summarizer_config = self.config.get("note_taker", {}).get("summarizer", {})
        
        self.max_summary_length = summarizer_config.get("max_summary_length", 200)
        self.max_key_points = summarizer_config.get("max_key_points", 5)
        self.min_sentence_length = summarizer_config.get("min_sentence_length", 10)
        self.use_llm = summarizer_config.get("use_llm", True)
        self.fallback_to_extractive = summarizer_config.get("fallback_to_extractive", True)
    
    async def create_summary(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Create a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length (defaults to config)
            
        Returns:
            Generated summary
        """
        if not text or not text.strip():
            return ""
        
        max_length = max_length or self.max_summary_length
        
        # If text is already short, return as is
        if len(text) <= max_length:
            return text.strip()
        
        # Try LLM-based summarization first
        if self.use_llm and self.language_chain:
            try:
                llm_summary = await self._create_llm_summary(text, max_length)
                if llm_summary:
                    return llm_summary
            except Exception as e:
                self.logger.warning(f"LLM summarization failed: {str(e)}")
        
        # Fallback to extractive summarization
        if self.fallback_to_extractive:
            return self._create_extractive_summary(text, max_length)
        
        # Simple truncation as last resort
        return self._truncate_text(text, max_length)
    
    async def _create_llm_summary(self, text: str, max_length: int) -> Optional[str]:
        """Create summary using LLM."""
        if not self.language_chain:
            return None
        
        # Determine language for appropriate prompt
        language_context = self.bilingual_manager.build_language_context(text)
        
        if language_context.user_query_language.value == 'ar':
            prompt = f"""اكتب ملخصاً مختصراً للنص التالي في أقل من {max_length} حرف. ركز على النقاط الرئيسية والمعلومات المهمة:

النص:
{text}

الملخص:"""
        else:
            prompt = f"""Write a concise summary of the following text in less than {max_length} characters. Focus on main points and important information:

Text:
{text}

Summary:"""
        
        try:
            response = await self.language_chain.process_message(prompt)
            summary = response.strip()
            
            if summary and len(summary) <= max_length * 1.2:  # Allow some flexibility
                return self._clean_summary(summary)
            
        except Exception as e:
            self.logger.warning(f"LLM summary generation error: {str(e)}")
        
        return None
    
    def _create_extractive_summary(self, text: str, max_length: int) -> str:
        """Create summary by extracting key sentences."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return self._truncate_text(text, max_length)
        
        # Score sentences based on various factors
        sentence_scores = self._score_sentences(sentences, text)
        
        # Select best sentences up to max_length
        selected_sentences = []
        current_length = 0
        
        # Sort by score (descending)
        sorted_sentences = sorted(
            zip(sentences, sentence_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        for sentence, score in sorted_sentences:
            sentence = sentence.strip()
            if len(sentence) < self.min_sentence_length:
                continue
            
            if current_length + len(sentence) <= max_length:
                selected_sentences.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
            else:
                # Try to fit a truncated version
                remaining_space = max_length - current_length
                if remaining_space > 20:  # Only if meaningful space left
                    truncated = self._truncate_text(sentence, remaining_space - 3) + "..."
                    selected_sentences.append(truncated)
                break
        
        if not selected_sentences:
            return self._truncate_text(sentences[0], max_length)
        
        # Join sentences and clean up
        summary = ' '.join(selected_sentences)
        return self._clean_summary(summary)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling both Arabic and English."""
        # Split on sentence endings
        sentences = re.split(r'[.!?؟]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.min_sentence_length:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _score_sentences(self, sentences: List[str], full_text: str) -> List[float]:
        """Score sentences for extractive summarization."""
        if not sentences:
            return []
        
        scores = []
        word_freq = self._calculate_word_frequency(full_text)
        
        for sentence in sentences:
            score = 0.0
            words = sentence.lower().split()
            
            # Score based on word frequency
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
            
            # Normalize by sentence length
            if len(words) > 0:
                score = score / len(words)
            
            # Bonus for first sentence
            if sentence == sentences[0]:
                score *= 1.2
            
            # Bonus for sentences with numbers or specific entities
            if re.search(r'\d+', sentence):
                score *= 1.1
            
            # Penalty for very short or very long sentences
            if len(words) < 5:
                score *= 0.8
            elif len(words) > 40:
                score *= 0.9
            
            scores.append(score)
        
        return scores
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, float]:
        """Calculate word frequency for scoring."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = {}
        
        for word in words:
            if len(word) > 2:  # Ignore very short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Normalize frequencies
        max_freq = max(word_count.values()) if word_count else 1
        return {word: count / max_freq for word, count in word_count.items()}
    
    async def extract_key_points(self, text: str, max_points: Optional[int] = None) -> List[str]:
        """
        Extract key points from text.
        
        Args:
            text: Text to analyze
            max_points: Maximum number of key points
            
        Returns:
            List of key points
        """
        if not text or not text.strip():
            return []
        
        max_points = max_points or self.max_key_points
        
        # Try LLM-based extraction first
        if self.use_llm and self.language_chain:
            try:
                llm_points = await self._extract_key_points_llm(text, max_points)
                if llm_points:
                    return llm_points
            except Exception as e:
                self.logger.warning(f"LLM key point extraction failed: {str(e)}")
        
        # Fallback to extractive method
        return self._extract_key_points_extractive(text, max_points)
    
    async def _extract_key_points_llm(self, text: str, max_points: int) -> Optional[List[str]]:
        """Extract key points using LLM."""
        if not self.language_chain:
            return None
        
        # Determine language for appropriate prompt
        language_context = self.bilingual_manager.build_language_context(text)
        
        if language_context.user_query_language.value == 'ar':
            prompt = f"""استخرج أهم {max_points} نقاط رئيسية من النص التالي. اكتب كل نقطة في سطر منفصل:

النص:
{text}

النقاط الرئيسية:"""
        else:
            prompt = f"""Extract the top {max_points} key points from the following text. Write each point on a separate line:

Text:
{text}

Key points:"""
        
        try:
            response = await self.language_chain.process_message(prompt)
            
            # Parse response into list
            lines = response.strip().split('\n')
            key_points = []
            
            for line in lines:
                line = line.strip()
                # Remove bullet points, numbers, etc.
                line = re.sub(r'^[-•*\d+.)\s]+', '', line).strip()
                if line and len(line) > 5:
                    key_points.append(line)
            
            return key_points[:max_points] if key_points else None
            
        except Exception as e:
            self.logger.warning(f"LLM key point extraction error: {str(e)}")
        
        return None
    
    def _extract_key_points_extractive(self, text: str, max_points: int) -> List[str]:
        """Extract key points using extractive methods."""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Score sentences
        sentence_scores = self._score_sentences(sentences, text)
        
        # Select top sentences as key points
        scored_sentences = list(zip(sentences, sentence_scores))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        key_points = []
        for sentence, score in scored_sentences[:max_points]:
            # Clean up sentence to make it a good key point
            point = self._clean_key_point(sentence)
            if point and len(point) > 10:
                key_points.append(point)
        
        return key_points
    
    def _clean_key_point(self, text: str) -> str:
        """Clean text to make it a good key point."""
        # Remove unnecessary prefixes
        text = re.sub(r'^(also|additionally|furthermore|moreover|besides)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(أيضاً|بالإضافة|علاوة على ذلك|كما)\s+', '', text)
        
        # Ensure it starts with capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Remove trailing punctuation except period
        text = re.sub(r'[,;:]$', '', text)
        if not text.endswith('.'):
            text += '.'
        
        return text.strip()
    
    async def create_bullet_points(self, text: str, max_points: int = 10) -> List[str]:
        """
        Create bullet points from text.
        
        Args:
            text: Text to convert to bullet points
            max_points: Maximum number of bullet points
            
        Returns:
            List of bullet points
        """
        # First extract key points
        key_points = await self.extract_key_points(text, max_points)
        
        # Format as bullet points
        bullet_points = []
        for point in key_points:
            # Remove trailing period for bullet point format
            point = point.rstrip('.')
            bullet_points.append(f"• {point}")
        
        return bullet_points
    
    async def create_structured_outline(self, text: str) -> Dict[str, Any]:
        """
        Create a structured outline of the text.
        
        Args:
            text: Text to outline
            
        Returns:
            Structured outline with sections and subsections
        """
        outline = {
            "summary": await self.create_summary(text),
            "key_points": await self.extract_key_points(text),
            "bullet_points": await self.create_bullet_points(text),
            "sections": []
        }
        
        # Try to identify sections in the text
        sections = self._identify_sections(text)
        
        for section_title, section_content in sections:
            section_outline = {
                "title": section_title,
                "content": section_content,
                "summary": await self.create_summary(section_content, max_length=100),
                "key_points": await self.extract_key_points(section_content, max_points=3)
            }
            outline["sections"].append(section_outline)
        
        return outline
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """Identify potential sections in the text."""
        # Look for section headers (simple heuristic)
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like a header
            if self._looks_like_header(line):
                # Save previous section
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Start new section
                current_section = line
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # No sections found, treat all as one section
                    if not sections:
                        sections.append(("Main Content", text))
                        break
        
        # Add last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        # If no sections found, create one main section
        if not sections:
            sections.append(("Main Content", text))
        
        return sections
    
    def _looks_like_header(self, line: str) -> bool:
        """Check if a line looks like a section header."""
        # Simple heuristics for headers
        if len(line) < 5 or len(line) > 100:
            return False
        
        # Check for header patterns
        header_patterns = [
            r'^\d+\.\s+.+',  # "1. Section title"
            r'^[A-Z][A-Za-z\s]+:$',  # "Section Title:"
            r'^[أ-ي][أ-ي\s]+:$',  # Arabic section title
            r'^\*\*(.+)\*\*$',  # **Section Title**
            r'^#{1,3}\s+.+',  # # Section Title
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line):
                return True
        
        # Check if line is all caps (potential header)
        if line.isupper() and len(line.split()) <= 6:
            return True
        
        return False
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length]
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]
        
        return truncated + "..."
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and normalize summary text."""
        if not summary:
            return ""
        
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary.strip())
        
        # Remove quotes if the entire summary is quoted
        if (summary.startswith('"') and summary.endswith('"')) or \
           (summary.startswith("'") and summary.endswith("'")):
            summary = summary[1:-1].strip()
        
        # Ensure proper capitalization
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure ends with proper punctuation
        if summary and not summary[-1] in '.!?؟':
            summary += '.'
        
        return summary
    
    async def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the structure and complexity of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Structure analysis results
        """
        sentences = self._split_into_sentences(text)
        paragraphs = text.split('\n\n')
        words = text.split()
        
        analysis = {
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "word_count": len(words),
            "character_count": len(text),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / max(len(sentences), 1),
            "avg_paragraph_length": sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1),
            "complexity_score": 0.0,
            "readability_level": "medium"
        }
        
        # Calculate complexity score
        complexity_factors = []
        
        # Sentence length factor
        if analysis["avg_sentence_length"] > 20:
            complexity_factors.append(0.3)
        elif analysis["avg_sentence_length"] < 10:
            complexity_factors.append(-0.1)
        
        # Paragraph length factor
        if analysis["avg_paragraph_length"] > 100:
            complexity_factors.append(0.2)
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        diversity_ratio = unique_words / max(len(words), 1)
        if diversity_ratio > 0.7:
            complexity_factors.append(0.2)
        elif diversity_ratio < 0.4:
            complexity_factors.append(-0.2)
        
        analysis["complexity_score"] = max(0.0, min(1.0, 0.5 + sum(complexity_factors)))
        
        # Readability level
        if analysis["complexity_score"] > 0.7:
            analysis["readability_level"] = "high"
        elif analysis["complexity_score"] < 0.3:
            analysis["readability_level"] = "low"
        
        return analysis