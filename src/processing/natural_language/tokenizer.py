"""
Advanced Tokenization System for AI Assistant
Author: Drmusab
Last Modified: 2025-05-27 13:38:47 UTC

This module provides comprehensive tokenization capabilities including multiple
tokenization strategies, multilingual support, caching, and integration with
the core system components.
"""

import gc
import hashlib
import json
import re
import threading
import time
import unicodedata
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import asyncio
import numpy as np
import tiktoken
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    T5Tokenizer,
    XLMRobertaTokenizer,
)

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CacheHit,
    CacheMiss,
    ComponentFailed,
    ComponentInitialized,
    ComponentRegistered,
    TokenizationCompleted,
    TokenizationFailed,
)
from src.core.health_check import HealthCheck

# Integration imports
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class TokenizationStrategy(Enum):
    """Tokenization strategies supported by the system."""

    WORD_LEVEL = "word_level"  # Word-based tokenization
    SUBWORD_BPE = "subword_bpe"  # Byte-Pair Encoding
    SUBWORD_WORDPIECE = "subword_wordpiece"  # WordPiece (BERT-style)
    SUBWORD_SENTENCEPIECE = "subword_sentencepiece"  # SentencePiece
    CHARACTER_LEVEL = "character_level"  # Character-based
    WHITESPACE = "whitespace"  # Simple whitespace splitting
    REGEX_BASED = "regex_based"  # Regular expression based
    NEURAL = "neural"  # Neural tokenization
    TIKTOKEN = "tiktoken"  # OpenAI tiktoken
    CUSTOM = "custom"  # Custom tokenization logic


class TokenType(Enum):
    """Types of tokens in the vocabulary."""

    WORD = "word"
    SUBWORD = "subword"
    CHARACTER = "character"
    SPECIAL = "special"  # [CLS], [SEP], etc.
    PADDING = "padding"  # [PAD]
    UNKNOWN = "unknown"  # [UNK], <unk>
    BEGINNING = "beginning"  # Beginning of word
    CONTINUATION = "continuation"  # Continuation of word
    PUNCTUATION = "punctuation"
    NUMBER = "number"
    EMOJI = "emoji"
    URL = "url"
    EMAIL = "email"
    HASHTAG = "hashtag"
    MENTION = "mention"


class TokenizationMode(Enum):
    """Tokenization processing modes."""

    STANDARD = "standard"  # Standard tokenization
    FAST = "fast"  # Fast tokenization with less preprocessing
    QUALITY = "quality"  # High-quality with extensive preprocessing
    STREAMING = "streaming"  # Streaming tokenization for large texts
    ADAPTIVE = "adaptive"  # Adaptive based on content type
    MULTILINGUAL = "multilingual"  # Optimized for multilingual content


class TextNormalization(Enum):
    """Text normalization strategies."""

    NONE = "none"  # No normalization
    BASIC = "basic"  # Basic lowercase and whitespace
    UNICODE = "unicode"  # Unicode normalization
    AGGRESSIVE = "aggressive"  # Aggressive normalization
    LANGUAGE_SPECIFIC = "language_specific"  # Language-specific rules
    CUSTOM = "custom"  # Custom normalization rules


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer instances."""

    strategy: TokenizationStrategy = TokenizationStrategy.SUBWORD_BPE
    mode: TokenizationMode = TokenizationMode.STANDARD
    normalization: TextNormalization = TextNormalization.BASIC

    # Model-specific settings
    model_name: Optional[str] = None
    vocab_size: int = 50000
    max_length: int = 512
    truncation: bool = True
    padding: bool = False

    # Language settings
    language: str = "en"
    languages: Set[str] = field(default_factory=lambda: {"en"})
    handle_multilingual: bool = True

    # Special tokens
    special_tokens: Dict[str, str] = field(default_factory=dict)
    add_special_tokens: bool = True

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_parallel_processing: bool = True
    batch_size: int = 32

    # Quality settings
    preserve_case: bool = False
    handle_oov: bool = True  # Out-of-vocabulary handling
    merge_spaces: bool = True
    strip_accents: bool = False

    # Advanced features
    enable_attention_mask: bool = True
    return_tensors: Optional[str] = None  # "pt", "tf", "np"
    enable_position_ids: bool = False
    enable_token_type_ids: bool = False


@dataclass
class Token:
    """Represents a single token with metadata."""

    text: str
    token_id: int
    token_type: TokenType = TokenType.WORD

    # Position information
    start_pos: int = -1
    end_pos: int = -1

    # Linguistic features
    is_word_start: bool = False
    is_word_end: bool = False
    original_text: Optional[str] = None

    # Metadata
    confidence: float = 1.0
    language: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizationResult:
    """Comprehensive tokenization result."""

    # Primary outputs
    tokens: List[Token]
    token_ids: List[int]
    attention_mask: Optional[List[int]] = None

    # Text reconstruction
    token_strings: List[str] = field(default_factory=list)
    offset_mapping: Optional[List[Tuple[int, int]]] = None

    # Metadata
    original_text: str = ""
    tokenizer_strategy: TokenizationStrategy = TokenizationStrategy.SUBWORD_BPE
    processing_time: float = 0.0

    # Quality metrics
    compression_ratio: float = 0.0
    oov_count: int = 0
    special_token_count: int = 0

    # Language analysis
    detected_languages: List[str] = field(default_factory=list)
    language_confidence: Dict[str, float] = field(default_factory=dict)

    # Advanced features
    token_type_ids: Optional[List[int]] = None
    position_ids: Optional[List[int]] = None
    word_ids: Optional[List[int]] = None

    # Statistics
    total_tokens: int = 0
    unique_tokens: int = 0
    vocabulary_coverage: float = 0.0

    # Errors and warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Caching
    cache_hit: bool = False
    cache_key: Optional[str] = None


class TokenizationError(Exception):
    """Custom exception for tokenization operations."""

    def __init__(
        self,
        message: str,
        strategy: Optional[TokenizationStrategy] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.strategy = strategy
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the tokenizer."""
        pass

    @abstractmethod
    async def tokenize(self, text: str, **kwargs) -> TokenizationResult:
        """Tokenize input text."""
        pass

    @abstractmethod
    async def detokenize(self, tokens: Union[List[str], List[int]], **kwargs) -> str:
        """Convert tokens back to text."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass

    @abstractmethod
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens mapping."""
        pass

    async def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        result = await self.tokenize(text, **kwargs)
        return result.token_ids

    async def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        return await self.detokenize(token_ids, **kwargs)

    def is_initialized(self) -> bool:
        """Check if tokenizer is initialized."""
        return self._initialized


class TransformerTokenizer(BaseTokenizer):
    """Tokenizer using HuggingFace transformers."""

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._model_cache: Dict[str, PreTrainedTokenizer] = {}

    async def initialize(self) -> None:
        """Initialize the transformer tokenizer."""
        try:
            model_name = self.config.model_name or "bert-base-uncased"

            # Load tokenizer with caching
            if model_name not in self._model_cache:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, trust_remote_code=False
                )
                self._model_cache[model_name] = self.tokenizer
            else:
                self.tokenizer = self._model_cache[model_name]

            # Configure special tokens
            if self.config.special_tokens:
                self.tokenizer.add_special_tokens(self.config.special_tokens)

            self._initialized = True
            self.logger.info(f"Initialized TransformerTokenizer with model: {model_name}")

        except Exception as e:
            raise TokenizationError(f"Failed to initialize transformer tokenizer: {str(e)}")

    async def tokenize(self, text: str, **kwargs) -> TokenizationResult:
        """Tokenize text using transformer tokenizer."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Tokenize with detailed output
            encoding = self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=self.config.truncation,
                padding=self.config.padding,
                return_tensors=self.config.return_tensors,
                return_attention_mask=self.config.enable_attention_mask,
                return_token_type_ids=self.config.enable_token_type_ids,
                return_offsets_mapping=True,
                add_special_tokens=self.config.add_special_tokens,
                **kwargs,
            )

            # Extract token information
            token_ids = encoding["input_ids"]
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()

            token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Create Token objects
            tokens = []
            for i, (token_str, token_id) in enumerate(zip(token_strings, token_ids)):
                token_type = self._classify_token_type(token_str)

                # Get position information if available
                start_pos, end_pos = -1, -1
                if "offset_mapping" in encoding and encoding["offset_mapping"] is not None:
                    offsets = encoding["offset_mapping"]
                    if hasattr(offsets, "tolist"):
                        offsets = offsets.tolist()
                    if i < len(offsets):
                        start_pos, end_pos = offsets[i]

                token = Token(
                    text=token_str,
                    token_id=token_id,
                    token_type=token_type,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    is_word_start=not token_str.startswith("##"),
                    original_text=token_str,
                )
                tokens.append(token)

            processing_time = time.time() - start_time

            # Calculate metrics
            compression_ratio = len(text) / len(token_ids) if token_ids else 0
            oov_count = sum(1 for tid in token_ids if tid == self.tokenizer.unk_token_id)
            special_token_count = sum(
                1 for token in token_strings if token in self.tokenizer.special_tokens_map.values()
            )

            # Build result
            result = TokenizationResult(
                tokens=tokens,
                token_ids=token_ids,
                token_strings=token_strings,
                original_text=text,
                tokenizer_strategy=self.config.strategy,
                processing_time=processing_time,
                compression_ratio=compression_ratio,
                oov_count=oov_count,
                special_token_count=special_token_count,
                total_tokens=len(token_ids),
                unique_tokens=len(set(token_ids)),
                attention_mask=encoding.get("attention_mask"),
                token_type_ids=encoding.get("token_type_ids"),
                offset_mapping=encoding.get("offset_mapping"),
            )

            return result

        except Exception as e:
            raise TokenizationError(f"Tokenization failed: {str(e)}", self.config.strategy)

    async def detokenize(self, tokens: Union[List[str], List[int]], **kwargs) -> str:
        """Convert tokens back to text."""
        if not self._initialized:
            await self.initialize()

        try:
            if isinstance(tokens[0], int):
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                # Convert token strings to IDs first
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                return self.tokenizer.decode(token_ids, skip_special_tokens=True)

        except Exception as e:
            raise TokenizationError(f"Detokenization failed: {str(e)}", self.config.strategy)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.tokenizer:
            return len(self.tokenizer)
        return 0

    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens mapping."""
        if self.tokenizer:
            return self.tokenizer.special_tokens_map
        return {}

    def _classify_token_type(self, token: str) -> TokenType:
        """Classify token type based on content."""
        if token in self.tokenizer.special_tokens_map.values():
            return TokenType.SPECIAL
        elif token.startswith("##"):
            return TokenType.CONTINUATION
        elif token == self.tokenizer.unk_token:
            return TokenType.UNKNOWN
        elif token == self.tokenizer.pad_token:
            return TokenType.PADDING
        elif token.isdigit():
            return TokenType.NUMBER
        elif any(char in token for char in ".,!?;:"):
            return TokenType.PUNCTUATION
        else:
            return TokenType.WORD


class TikTokenTokenizer(BaseTokenizer):
    """OpenAI tiktoken-based tokenizer."""

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.encoding = None
        self.model_name = config.model_name or "gpt-3.5-turbo"

    async def initialize(self) -> None:
        """Initialize tiktoken encoder."""
        try:
            if self.model_name.startswith("gpt-"):
                self.encoding = tiktoken.encoding_for_model(self.model_name)
            else:
                self.encoding = tiktoken.get_encoding(self.model_name)

            self._initialized = True
            self.logger.info(f"Initialized TikTokenTokenizer with model: {self.model_name}")

        except Exception as e:
            raise TokenizationError(f"Failed to initialize tiktoken: {str(e)}")

    async def tokenize(self, text: str, **kwargs) -> TokenizationResult:
        """Tokenize text using tiktoken."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Encode text
            token_ids = self.encoding.encode(text)

            # Decode back to get token strings
            token_strings = []
            for token_id in token_ids:
                token_str = self.encoding.decode([token_id])
                token_strings.append(token_str)

            # Create Token objects
            tokens = []
            for i, (token_str, token_id) in enumerate(zip(token_strings, token_ids)):
                token = Token(
                    text=token_str,
                    token_id=token_id,
                    token_type=TokenType.SUBWORD,
                    original_text=token_str,
                )
                tokens.append(token)

            processing_time = time.time() - start_time
            compression_ratio = len(text) / len(token_ids) if token_ids else 0

            result = TokenizationResult(
                tokens=tokens,
                token_ids=token_ids,
                token_strings=token_strings,
                original_text=text,
                tokenizer_strategy=TokenizationStrategy.TIKTOKEN,
                processing_time=processing_time,
                compression_ratio=compression_ratio,
                total_tokens=len(token_ids),
                unique_tokens=len(set(token_ids)),
            )

            return result

        except Exception as e:
            raise TokenizationError(
                f"TikToken tokenization failed: {str(e)}", TokenizationStrategy.TIKTOKEN
            )

    async def detokenize(self, tokens: Union[List[str], List[int]], **kwargs) -> str:
        """Convert tokens back to text."""
        if not self._initialized:
            await self.initialize()

        try:
            if isinstance(tokens[0], int):
                return self.encoding.decode(tokens)
            else:
                # For token strings, we need to encode them first
                raise NotImplementedError("TikToken doesn't support direct string token decoding")

        except Exception as e:
            raise TokenizationError(
                f"TikToken detokenization failed: {str(e)}", TokenizationStrategy.TIKTOKEN
            )

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.encoding:
            return self.encoding.n_vocab
        return 0

    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens mapping."""
        return {}  # TikToken doesn't expose special tokens directly


class CustomTokenizer(BaseTokenizer):
    """Custom tokenizer with flexible rules."""

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.token_patterns: List[Tuple[str, TokenType]] = []
        self._setup_patterns()

    def _setup_patterns(self) -> None:
        """Setup tokenization patterns."""
        # Define regex patterns for different token types
        self.token_patterns = [
            (r"https?://\S+", TokenType.URL),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", TokenType.EMAIL),
            (r"#\w+", TokenType.HASHTAG),
            (r"@\w+", TokenType.MENTION),
            (r"\d+", TokenType.NUMBER),
            (r"[^\w\s]", TokenType.PUNCTUATION),
            (r"\w+", TokenType.WORD),
        ]

    async def initialize(self) -> None:
        """Initialize custom tokenizer."""
        # Build vocabulary from patterns or load from config
        self.vocab = {"<unk>": 0, "<pad>": 1, "<start>": 2, "<end>": 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._initialized = True
        self.logger.info("Initialized CustomTokenizer")

    async def tokenize(self, text: str, **kwargs) -> TokenizationResult:
        """Tokenize text using custom rules."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            tokens = []
            token_ids = []
            token_strings = []
            current_pos = 0

            while current_pos < len(text):
                matched = False

                for pattern, token_type in self.token_patterns:
                    regex = re.compile(pattern)
                    match = regex.match(text, current_pos)

                    if match:
                        token_text = match.group()
                        start_pos = match.start()
                        end_pos = match.end()

                        # Get or create token ID
                        if token_text not in self.vocab:
                            token_id = len(self.vocab)
                            self.vocab[token_text] = token_id
                            self.reverse_vocab[token_id] = token_text
                        else:
                            token_id = self.vocab[token_text]

                        token = Token(
                            text=token_text,
                            token_id=token_id,
                            token_type=token_type,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            original_text=token_text,
                        )

                        tokens.append(token)
                        token_ids.append(token_id)
                        token_strings.append(token_text)

                        current_pos = end_pos
                        matched = True
                        break

                if not matched:
                    current_pos += 1  # Skip unmatched character

            processing_time = time.time() - start_time
            compression_ratio = len(text) / len(token_ids) if token_ids else 0

            result = TokenizationResult(
                tokens=tokens,
                token_ids=token_ids,
                token_strings=token_strings,
                original_text=text,
                tokenizer_strategy=TokenizationStrategy.CUSTOM,
                processing_time=processing_time,
                compression_ratio=compression_ratio,
                total_tokens=len(token_ids),
                unique_tokens=len(set(token_ids)),
            )

            return result

        except Exception as e:
            raise TokenizationError(
                f"Custom tokenization failed: {str(e)}", TokenizationStrategy.CUSTOM
            )

    async def detokenize(self, tokens: Union[List[str], List[int]], **kwargs) -> str:
        """Convert tokens back to text."""
        if not self._initialized:
            await self.initialize()

        try:
            if isinstance(tokens[0], int):
                token_strings = [self.reverse_vocab.get(tid, "<unk>") for tid in tokens]
            else:
                token_strings = tokens

            return " ".join(token_strings)

        except Exception as e:
            raise TokenizationError(
                f"Custom detokenization failed: {str(e)}", TokenizationStrategy.CUSTOM
            )

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens mapping."""
        return {
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "start_token": "<start>",
            "end_token": "<end>",
        }


class TextPreprocessor:
    """Advanced text preprocessing for tokenization."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.logger = get_logger(__name__)

    def preprocess(self, text: str) -> str:
        """Apply preprocessing to text."""
        if self.config.normalization == TextNormalization.NONE:
            return text

        processed_text = text

        # Basic normalization
        if self.config.normalization in [TextNormalization.BASIC, TextNormalization.AGGRESSIVE]:
            if not self.config.preserve_case:
                processed_text = processed_text.lower()

            if self.config.merge_spaces:
                processed_text = re.sub(r"\s+", " ", processed_text)

            processed_text = processed_text.strip()

        # Unicode normalization
        if self.config.normalization in [TextNormalization.UNICODE, TextNormalization.AGGRESSIVE]:
            processed_text = unicodedata.normalize("NFKC", processed_text)

            if self.config.strip_accents:
                processed_text = "".join(
                    char
                    for char in unicodedata.normalize("NFD", processed_text)
                    if unicodedata.category(char) != "Mn"
                )

        # Aggressive normalization
        if self.config.normalization == TextNormalization.AGGRESSIVE:
            # Remove or normalize special characters
            processed_text = re.sub(r'[^\w\s.,!?;:(){}[\]"\'-]', " ", processed_text)
            processed_text = re.sub(r"\s+", " ", processed_text)

        return processed_text


class TokenizerCache:
    """Caching system for tokenization results."""

    def __init__(self, cache_strategy: Optional[CacheStrategy] = None, ttl_seconds: int = 3600):
        self.cache_strategy = cache_strategy
        self.ttl_seconds = ttl_seconds
        self.local_cache: Dict[str, Tuple[TokenizationResult, datetime]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, text: str, config: TokenizerConfig) -> str:
        """Generate cache key for text and config."""
        config_hash = hashlib.md5(
            json.dumps(config.__dict__, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"tokenizer:{config_hash}:{text_hash}"

    async def get(self, text: str, config: TokenizerConfig) -> Optional[TokenizationResult]:
        """Get cached tokenization result."""
        cache_key = self._generate_cache_key(text, config)

        # Check local cache first
        if cache_key in self.local_cache:
            result, timestamp = self.local_cache[cache_key]
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()

            if age <= self.ttl_seconds:
                self.cache_hits += 1
                result.cache_hit = True
                result.cache_key = cache_key
                return result
            else:
                del self.local_cache[cache_key]

        # Check external cache if available
        if self.cache_strategy:
            try:
                cached_data = await self.cache_strategy.get(cache_key)
                if cached_data:
                    result = TokenizationResult(**cached_data)
                    result.cache_hit = True
                    result.cache_key = cache_key
                    self.cache_hits += 1
                    return result
            except Exception:
                pass  # Cache miss or error

        self.cache_misses += 1
        return None

    async def set(self, text: str, config: TokenizerConfig, result: TokenizationResult) -> None:
        """Cache tokenization result."""
        cache_key = self._generate_cache_key(text, config)
        timestamp = datetime.now(timezone.utc)

        # Store in local cache
        self.local_cache[cache_key] = (result, timestamp)

        # Store in external cache if available
        if self.cache_strategy:
            try:
                # Convert to serializable format
                cached_data = {
                    k: v
                    for k, v in result.__dict__.items()
                    if k not in ["tokens"]  # Skip complex objects
                }
                await self.cache_strategy.set(cache_key, cached_data, ttl=self.ttl_seconds)
            except Exception as e:
                # Log but don't fail on cache errors
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
        }


class TokenizerFactory:
    """Factory for creating tokenizer instances."""

    def __init__(self, logger):
        self.logger = logger
        self._tokenizer_cache: Dict[str, BaseTokenizer] = {}

    def create_tokenizer(self, config: TokenizerConfig) -> BaseTokenizer:
        """Create tokenizer based on strategy."""
        try:
            # Generate cache key
            cache_key = f"{config.strategy.value}:{config.model_name or 'default'}"

            # Return cached instance if available
            if cache_key in self._tokenizer_cache:
                return self._tokenizer_cache[cache_key]

            # Create new tokenizer
            if config.strategy == TokenizationStrategy.TIKTOKEN:
                tokenizer = TikTokenTokenizer(config)
            elif config.strategy in [
                TokenizationStrategy.SUBWORD_BPE,
                TokenizationStrategy.SUBWORD_WORDPIECE,
                TokenizationStrategy.SUBWORD_SENTENCEPIECE,
            ]:
                tokenizer = TransformerTokenizer(config)
            elif config.strategy == TokenizationStrategy.CUSTOM:
                tokenizer = CustomTokenizer(config)
            else:
                # Default to transformer tokenizer
                tokenizer = TransformerTokenizer(config)

            # Cache the tokenizer
            self._tokenizer_cache[cache_key] = tokenizer

            return tokenizer

        except Exception as e:
            raise TokenizationError(f"Failed to create tokenizer: {str(e)}", config.strategy)


class EnhancedTokenizer:
    """
    Enhanced tokenization system with multiple strategies, caching, and monitoring.

    Features:
    - Multiple tokenization strategies (BPE, WordPiece, TikToken, Custom)
    - Intelligent caching with TTL and invalidation
    - Comprehensive preprocessing and normalization
    - Multilingual support and language detection
    - Performance monitoring and metrics
    - Integration with core system components
    - Streaming tokenization for large texts
    - Adaptive strategy selection
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced tokenizer.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Core components
        self._tokenizer_factory = TokenizerFactory(self.logger)
        self._preprocessor: Optional[TextPreprocessor] = None
        self._cache: Optional[TokenizerCache] = None

        # Active tokenizers
        self._active_tokenizers: Dict[str, BaseTokenizer] = {}
        self._default_config: Optional[TokenizerConfig] = None

        # Performance tracking
        self._setup_monitoring()
        self._tokenization_stats = {
            "total_requests": 0,
            "successful_tokenizations": 0,
            "failed_tokenizations": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
        }

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Configuration
        self._max_concurrent_requests = self.config.get("tokenizer.max_concurrent_requests", 100)
        self._enable_caching = self.config.get("tokenizer.enable_caching", True)
        self._cache_ttl = self.config.get("tokenizer.cache_ttl_seconds", 3600)

        # Register health check
        self.health_check.register_component("tokenizer", self._health_check_callback)

        self.logger.info("EnhancedTokenizer initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)

            # Register tokenization metrics
            self.metrics.register_counter("tokenization_requests_total")
            self.metrics.register_counter("tokenization_success_total")
            self.metrics.register_counter("tokenization_errors_total")
            self.metrics.register_histogram("tokenization_duration_seconds")
            self.metrics.register_gauge("active_tokenizers")
            self.metrics.register_counter("tokenization_cache_hits_total")
            self.metrics.register_counter("tokenization_cache_misses_total")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    @handle_exceptions
    async def initialize(self) -> None:
        """Initialize the tokenizer system."""
        try:
            # Load default configuration
            await self._load_default_config()

            # Setup preprocessing
            self._preprocessor = TextPreprocessor(self._default_config)

            # Setup caching
            if self._enable_caching:
                await self._setup_caching()

            # Initialize default tokenizers
            await self._initialize_default_tokenizers()

            # Register event handlers
            await self._register_event_handlers()

            # Start background tasks
            await self._start_background_tasks()

            # Emit initialization event
            await self.event_bus.emit(
                ComponentInitialized(component_id="tokenizer", initialization_time=0.0)
            )

            self.logger.info("EnhancedTokenizer initialization completed")

        except Exception as e:
            await self.event_bus.emit(
                ComponentFailed(
                    component_id="tokenizer", error_message=str(e), error_type=type(e).__name__
                )
            )
            raise TokenizationError(f"Failed to initialize tokenizer: {str(e)}")

    async def _load_default_config(self) -> None:
        """Load default tokenizer configuration."""
        tokenizer_config = self.config.get("tokenizer", {})

        self._default_config = TokenizerConfig(
            strategy=TokenizationStrategy(tokenizer_config.get("default_strategy", "subword_bpe")),
            mode=TokenizationMode(tokenizer_config.get("default_mode", "standard")),
            normalization=TextNormalization(tokenizer_config.get("normalization", "basic")),
            model_name=tokenizer_config.get("default_model", "bert-base-uncased"),
            vocab_size=tokenizer_config.get("vocab_size", 50000),
            max_length=tokenizer_config.get("max_length", 512),
            enable_caching=tokenizer_config.get("enable_caching", True),
            cache_ttl_seconds=tokenizer_config.get("cache_ttl_seconds", 3600),
        )

    async def _setup_caching(self) -> None:
        """Setup tokenization caching."""
        try:
            cache_strategy = self.container.get(CacheStrategy)
            self._cache = TokenizerCache(cache_strategy, self._cache_ttl)
            self.logger.info("Tokenization caching enabled")
        except Exception as e:
            self.logger.warning(f"Failed to setup caching: {str(e)}")
            self._cache = TokenizerCache(None, self._cache_ttl)

    async def _initialize_default_tokenizers(self) -> None:
        """Initialize commonly used tokenizers."""
        default_strategies = [TokenizationStrategy.SUBWORD_BPE, TokenizationStrategy.TIKTOKEN]

        for strategy in default_strategies:
            try:
                config = TokenizerConfig(
                    strategy=strategy, model_name=self._get_default_model_for_strategy(strategy)
                )
                tokenizer = self._tokenizer_factory.create_tokenizer(config)
                await tokenizer.initialize()

                strategy_key = f"{strategy.value}:default"
                self._active_tokenizers[strategy_key] = tokenizer

            except Exception as e:
                self.logger.warning(f"Failed to initialize {strategy.value} tokenizer: {str(e)}")

    def _get_default_model_for_strategy(self, strategy: TokenizationStrategy) -> str:
        """Get default model for tokenization strategy."""
        defaults = {
            TokenizationStrategy.SUBWORD_BPE: "bert-base-uncased",
            TokenizationStrategy.SUBWORD_WORDPIECE: "bert-base-uncased",
            TokenizationStrategy.SUBWORD_SENTENCEPIECE: "t5-small",
            TokenizationStrategy.TIKTOKEN: "gpt-3.5-turbo",
        }
        return defaults.get(strategy, "bert-base-uncased")

    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # Register for relevant events if needed
        pass

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup task
        if self._cache:
            cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._background_tasks.add(cleanup_task)

        # Performance monitoring task
        monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        self._background_tasks.add(monitoring_task)

    @handle_exceptions
    async def tokenize(
        self,
        text: str,
        strategy: Optional[TokenizationStrategy] = None,
        config: Optional[TokenizerConfig] = None,
        **kwargs,
    ) -> TokenizationResult:
        """
        Tokenize input text using specified strategy.

        Args:
            text: Input text to tokenize
            strategy: Tokenization strategy to use
            config: Custom tokenizer configuration
            **kwargs: Additional tokenization parameters

        Returns:
            TokenizationResult with tokens and metadata
        """
        start_time = time.time()

        # Update statistics
        self._tokenization_stats["total_requests"] += 1

        if self.metrics:
            self.metrics.increment("tokenization_requests_total")

        with self.tracer.trace("tokenization") if self.tracer else None:
            try:
                # Use provided config or create from strategy
                if config is None:
                    strategy = strategy or self._default_config.strategy
                    config = TokenizerConfig(strategy=strategy)
                    if strategy == TokenizationStrategy.TIKTOKEN:
                        config.model_name = "gpt-3.5-turbo"

                # Check cache first
                if self._enable_caching and self._cache:
                    cached_result = await self._cache.get(text, config)
                    if cached_result:
                        if self.metrics:
                            self.metrics.increment("tokenization_cache_hits_total")
                        return cached_result
                    else:
                        if self.metrics:
                            self.metrics.increment("tokenization_cache_misses_total")

                # Preprocess text
                if self._preprocessor:
                    processed_text = self._preprocessor.preprocess(text)
                else:
                    processed_text = text

                # Get or create tokenizer
                tokenizer = await self._get_tokenizer(config)

                # Perform tokenization
                result = await tokenizer.tokenize(processed_text, **kwargs)

                # Cache result if caching is enabled
                if self._enable_caching and self._cache:
                    await self._cache.set(text, config, result)

                # Update statistics
                processing_time = time.time() - start_time
                self._tokenization_stats["successful_tokenizations"] += 1
                self._tokenization_stats["total_processing_time"] += processing_time
                self._tokenization_stats["average_processing_time"] = (
                    self._tokenization_stats["total_processing_time"]
                    / self._tokenization_stats["successful_tokenizations"]
                )

                if self.metrics:
                    self.metrics.increment("tokenization_success_total")
                    self.metrics.record("tokenization_duration_seconds", processing_time)

                # Emit completion event
                await self.event_bus.emit(
                    TokenizationCompleted(
                        text_length=len(text),
                        token_count=result.total_tokens,
                        processing_time=processing_time,
                        strategy=config.strategy.value,
                    )
                )

                return result

            except Exception as e:
                self._tokenization_stats["failed_tokenizations"] += 1

                if self.metrics:
                    self.metrics.increment("tokenization_errors_total")

                # Emit failure event
                await self.event_bus.emit(
                    TokenizationFailed(
                        text_length=len(text),
                        error_message=str(e),
                        strategy=config.strategy.value if config else "unknown",
                    )
                )

                self.logger.error(f"Tokenization failed: {str(e)}")
                raise TokenizationError(
                    f"Tokenization failed: {str(e)}", config.strategy if config else None
                )

    async def _get_tokenizer(self, config: TokenizerConfig) -> BaseTokenizer:
        """Get or create tokenizer for configuration."""
        tokenizer_key = f"{config.strategy.value}:{config.model_name or 'default'}"

        if tokenizer_key not in self._active_tokenizers:
            tokenizer = self._tokenizer_factory.create_tokenizer(config)
            await tokenizer.initialize()
            self._active_tokenizers[tokenizer_key] = tokenizer

            if self.metrics:
                self.metrics.set("active_tokenizers", len(self._active_tokenizers))

        return self._active_tokenizers[tokenizer_key]

    @handle_exceptions
    async def detokenize(
        self,
        tokens: Union[List[str], List[int]],
        strategy: Optional[TokenizationStrategy] = None,
        config: Optional[TokenizerConfig] = None,
        **kwargs,
    ) -> str:
        """
        Convert tokens back to text.

        Args:
            tokens: List of token strings or IDs
            strategy: Tokenization strategy used
            config: Tokenizer configuration
            **kwargs: Additional parameters

        Returns:
            Reconstructed text
        """
        try:
            # Use provided config or create from strategy
            if config is None:
                strategy = strategy or self._default_config.strategy
                config = TokenizerConfig(strategy=strategy)

            # Get tokenizer
            tokenizer = await self._get_tokenizer(config)

            # Perform detokenization
            text = await tokenizer.detokenize(tokens, **kwargs)

            return text

        except Exception as e:
            self.logger.error(f"Detokenization failed: {str(e)}")
            raise TokenizationError(
                f"Detokenization failed: {str(e)}", config.strategy if config else None
            )

    async def stream_tokenize(
        self,
        text_stream: AsyncGenerator[str, None],
        strategy: Optional[TokenizationStrategy] = None,
        config: Optional[TokenizerConfig] = None,
    ) -> AsyncGenerator[TokenizationResult, None]:
        """
        Stream tokenization for large texts.

        Args:
            text_stream: Async generator of text chunks
            strategy: Tokenization strategy
            config: Tokenizer configuration

        Yields:
            TokenizationResult for each chunk
        """
        async for text_chunk in text_stream:
            if text_chunk.strip():
                result = await self.tokenize(text_chunk, strategy, config)
                yield result

    def get_tokenization_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        stats = dict(self._tokenization_stats)

        if self._cache:
            cache_stats = self._cache.get_stats()
            stats.update(cache_stats)

        stats["active_tokenizers"] = len(self._active_tokenizers)

        return stats

    def list_available_strategies(self) -> List[str]:
        """List available tokenization strategies."""
        return [strategy.value for strategy in TokenizationStrategy]

    def get_strategy_info(self, strategy: TokenizationStrategy) -> Dict[str, Any]:
        """Get information about a tokenization strategy."""
        strategy_info = {
            "name": strategy.value,
            "description": self._get_strategy_description(strategy),
            "supported_models": self._get_supported_models(strategy),
            "default_model": self._get_default_model_for_strategy(strategy),
        }

        # Check if tokenizer is available
        tokenizer_key = f"{strategy.value}:default"
        strategy_info["available"] = tokenizer_key in self._active_tokenizers

        return strategy_info

    def _get_strategy_description(self, strategy: TokenizationStrategy) -> str:
        """Get description for tokenization strategy."""
        descriptions = {
            TokenizationStrategy.WORD_LEVEL: "Word-level tokenization splitting on whitespace",
            TokenizationStrategy.SUBWORD_BPE: "Subword tokenization using Byte-Pair Encoding",
            TokenizationStrategy.SUBWORD_WORDPIECE: "WordPiece tokenization (BERT-style)",
            TokenizationStrategy.SUBWORD_SENTENCEPIECE: "SentencePiece tokenization",
            TokenizationStrategy.CHARACTER_LEVEL: "Character-level tokenization",
            TokenizationStrategy.TIKTOKEN: "OpenAI's tiktoken tokenization",
            TokenizationStrategy.CUSTOM: "Custom tokenization with flexible rules",
        }
        return descriptions.get(strategy, "Unknown tokenization strategy")

    def _get_supported_models(self, strategy: TokenizationStrategy) -> List[str]:
        """Get supported models for tokenization strategy."""
        models = {
            TokenizationStrategy.SUBWORD_BPE: [
                "bert-base-uncased",
                "bert-base-cased",
                "roberta-base",
                "distilbert-base-uncased",
            ],
            TokenizationStrategy.SUBWORD_WORDPIECE: [
                "bert-base-uncased",
                "bert-base-cased",
                "bert-large-uncased",
            ],
            TokenizationStrategy.SUBWORD_SENTENCEPIECE: [
                "t5-small",
                "t5-base",
                "t5-large",
                "xlm-roberta-base",
            ],
            TokenizationStrategy.TIKTOKEN: [
                "gpt-3.5-turbo",
                "gpt-4",
                "text-davinci-003",
                "text-ada-001",
            ],
        }
        return models.get(strategy, [])

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                if self._cache:
                    # Clean up expired entries from local cache
                    current_time = datetime.now(timezone.utc)
                    expired_keys = []

                    for cache_key, (result, timestamp) in self._cache.local_cache.items():
                        age = (current_time - timestamp).total_seconds()
                        if age > self._cache.ttl_seconds:
                            expired_keys.append(cache_key)

                    for key in expired_keys:
                        del self._cache.local_cache[key]

                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute

                # Update metrics
                if self.metrics:
                    stats = self.get_tokenization_stats()
                    self.metrics.set("active_tokenizers", stats["active_tokenizers"])

                # Garbage collection for memory management
                if self._tokenization_stats["total_requests"] % 1000 == 0:
                    gc.collect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {str(e)}")

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the tokenizer."""
        try:
            stats = self.get_tokenization_stats()

            # Calculate health score based on success rate
            total_requests = stats["total_requests"]
            successful_requests = stats["successful_tokenizations"]

            if total_requests == 0:
                health_score = 1.0
                status = "healthy"
            else:
                success_rate = successful_requests / total_requests
                health_score = success_rate

                if success_rate >= 0.95:
                    status = "healthy"
                elif success_rate >= 0.80:
                    status = "degraded"
                else:
                    status = "unhealthy"

            return {
                "status": status,
                "health_score": health_score,
                "active_tokenizers": len(self._active_tokenizers),
                "total_requests": total_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "average_processing_time": stats["average_processing_time"],
                "cache_enabled": self._enable_caching,
                "cache_hit_rate": stats.get("hit_rate", 0) if self._cache else 0,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup tokenizer resources."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()

            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Clear caches
            if self._cache:
                self._cache.local_cache.clear()

            # Clear tokenizer instances
            self._active_tokenizers.clear()

            self.logger.info("EnhancedTokenizer cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during tokenizer cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, "_active_tokenizers") and self._active_tokenizers:
                self.logger.warning("EnhancedTokenizer destroyed with active tokenizers")
        except Exception:
            pass  # Ignore cleanup errors in destructor
