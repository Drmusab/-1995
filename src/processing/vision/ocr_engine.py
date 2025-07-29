"""
Enhanced OCR Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-06-03

This module provides comprehensive Optical Character Recognition (OCR) capabilities
for the AI assistant, including text detection, recognition, layout analysis, and
document understanding with full integration into the core system architecture.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Union, AsyncGenerator
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import numpy as np
import cv2
from contextlib import asynccontextmanager

# Core imports - following the established patterns
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentRegistered,
    ComponentInitialized,
    ComponentFailed,
    ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability - matching the system pattern
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Integrations
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager

# Learning and adaptation
from src.learning.feedback_processor import FeedbackProcessor

# Try to import OCR libraries with graceful fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class OCRMethod(Enum):
    """OCR recognition methods."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    GOOGLE_VISION = "google_vision"
    AZURE_COGNITIVE = "azure_cognitive"
    AWS_TEXTRACT = "aws_textract"
    ENSEMBLE = "ensemble"


class TextDetectionMethod(Enum):
    """Text detection methods."""
    EAST = "east"
    CRAFT = "craft"
    DB = "db"
    TEXTBOXES = "textboxes"
    PIXEL_LINK = "pixel_link"
    FOTS = "fots"


class DocumentType(Enum):
    """Types of documents for specialized processing."""
    GENERAL = "general"
    RECEIPT = "receipt"
    INVOICE = "invoice"
    FORM = "form"
    TABLE = "table"
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    BUSINESS_CARD = "business_card"
    LICENSE_PLATE = "license_plate"
    WHITEBOARD = "whiteboard"
    SCREENSHOT = "screenshot"
    PASSPORT = "passport"
    ID_CARD = "id_card"
    MEDICAL_REPORT = "medical_report"
    ACADEMIC_PAPER = "academic_paper"


class QualityLevel(Enum):
    """Quality levels for OCR processing."""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Speed/accuracy balance
    ACCURATE = "accurate"   # Accuracy optimized
    RESEARCH = "research"   # Maximum accuracy for research


class LayoutElement(Enum):
    """Document layout elements."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TITLE = "title"
    TABLE = "table"
    LIST = "list"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    SIGNATURE = "signature"
    LOGO = "logo"
    WATERMARK = "watermark"


class TextOrientation(Enum):
    """Text orientation angles."""
    HORIZONTAL = 0
    VERTICAL_90 = 90
    INVERTED_180 = 180
    VERTICAL_270 = 270


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0
    
    def to_polygon(self) -> List[Tuple[float, float]]:
        """Convert to polygon format."""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class TextRegion:
    """Detected text region with metadata."""
    text: str
    bbox: BoundingBox
    confidence: float
    language: str = "en"
    
    # Text properties
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    is_underlined: bool = False
    
    # Layout information
    layout_element: LayoutElement = LayoutElement.PARAGRAPH
    reading_order: int = 0
    line_number: int = 0
    word_count: int = 0
    
    # Quality metrics
    clarity_score: float = 0.0
    skew_angle: float = 0.0
    orientation: TextOrientation = TextOrientation.HORIZONTAL
    
    # Character-level information
    character_confidences: List[float] = field(default_factory=list)
    word_confidences: List[float] = field(default_factory=list)
    
    # Metadata
    detected_by: str = "unknown"
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentLayout:
    """Document layout analysis result."""
    page_number: int
    page_dimensions: Tuple[int, int]  # width, height
    text_regions: List[TextRegion]
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    
    # Layout metrics
    text_density: float = 0.0
    column_count: int = 1
    margin_info: Dict[str, float] = field(default_factory=dict)
    
    # Document structure
    headings: List[TextRegion] = field(default_factory=list)
    paragraphs: List[TextRegion] = field(default_factory=list)
    lists: List[List[TextRegion]] = field(default_factory=list)
    
    # Quality assessment
    layout_quality: float = 0.0
    text_quality: float = 0.0
    overall_quality: float = 0.0


@dataclass
class OCRResult:
    """Comprehensive OCR processing result."""
    success: bool
    request_id: str
    processing_time: float
    
    # Core results
    extracted_text: str
    document_layouts: List[DocumentLayout]
    total_pages: int
    
    # Recognition metadata
    method_used: OCRMethod
    quality_level: QualityLevel
    detected_languages: List[str]
    document_type: DocumentType
    
    # Image metadata
    image_dimensions: Tuple[int, int]  # height, width
    image_dpi: Optional[int] = None
    color_mode: str = "unknown"
    
    # Performance metrics
    preprocessing_time: float = 0.0
    detection_time: float = 0.0
    recognition_time: float = 0.0
    postprocessing_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Quality metrics
    overall_confidence: float = 0.0
    text_clarity: float = 0.0
    layout_accuracy: float = 0.0
    character_accuracy: float = 0.0
    
    # Structured data extraction
    structured_data: Dict[str, Any] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    key_value_pairs: Dict[str, str] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Processing chain
    preprocessing_steps: List[str] = field(default_factory=list)
    enhancement_applied: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRConfiguration:
    """Configuration for OCR processing."""
    # Method settings
    primary_method: OCRMethod = OCRMethod.TESSERACT
    fallback_methods: List[OCRMethod] = field(default_factory=lambda: [OCRMethod.EASYOCR])
    quality_level: QualityLevel = QualityLevel.BALANCED
    
    # Language settings
    languages: List[str] = field(default_factory=lambda: ["en"])
    auto_detect_language: bool = True
    language_confidence_threshold: float = 0.8
    
    # Document processing
    document_type: DocumentType = DocumentType.GENERAL
    auto_detect_document_type: bool = True
    enable_layout_analysis: bool = True
    enable_table_detection: bool = True
    
    # Image preprocessing
    enable_preprocessing: bool = True
    auto_rotate: bool = True
    auto_deskew: bool = True
    enhance_contrast: bool = True
    denoise: bool = True
    resize_factor: Optional[float] = None
    
    # Text detection
    text_detection_method: TextDetectionMethod = TextDetectionMethod.EAST
    min_text_size: int = 8
    max_text_size: int = 200
    text_confidence_threshold: float = 0.5
    
    # Recognition settings
    character_whitelist: Optional[str] = None
    character_blacklist: Optional[str] = None
    preserve_interword_spaces: bool = True
    output_format: str = "text"  # text, hocr, pdf, tsv
    
    # Performance settings
    enable_gpu: bool = True
    max_image_size: Tuple[int, int] = (4096, 4096)
    batch_size: int = 1
    num_threads: int = 4
    timeout_seconds: float = 120.0
    
    # Quality control
    min_confidence: float = 0.3
    enable_spell_check: bool = False
    enable_grammar_check: bool = False
    confidence_filtering: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 3600
    cache_preprocessed_images: bool = True
    
    # Output settings
    include_word_boxes: bool = True
    include_character_boxes: bool = False
    include_layout_info: bool = True
    include_confidence_scores: bool = True
    
    # Debugging
    save_debug_images: bool = False
    debug_output_dir: str = "/tmp/ocr_debug"
    enable_visualization: bool = False


class OCRError(Exception):
    """Custom exception for OCR operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 method: Optional[str] = None, image_path: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.method = method
        self.image_path = image_path
        self.timestamp = datetime.now(timezone.utc)


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the OCR engine."""
        pass
    
    @abstractmethod
    async def extract_text(self, image: np.ndarray, 
                          config: Optional[OCRConfiguration] = None) -> OCRResult:
        """Extract text from an image."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class TesseractOCREngine(OCREngine):
    """Tesseract-based OCR engine."""
    
    def __init__(self, logger):
        self.logger = logger
        self.is_initialized = False
        self.config_cache = {}
    
    async def initialize(self) -> None:
        """Initialize Tesseract OCR engine."""
        if not TESSERACT_AVAILABLE:
            raise OCRError("Tesseract is not available", "TESSERACT_NOT_FOUND")
        
        try:
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
            self.is_initialized = True
        except Exception as e:
            raise OCRError(f"Failed to initialize Tesseract: {str(e)}", "TESSERACT_INIT_FAILED")
    
    async def extract_text(self, image: np.ndarray, 
                          config: Optional[OCRConfiguration] = None) -> OCRResult:
        """Extract text using Tesseract."""
        if not self.is_initialized:
            await self.initialize()
        
        config = config or OCRConfiguration()
        start_time = time.time()
        
        try:
            # Prepare Tesseract configuration
            tesseract_config = self._build_tesseract_config(config)
            
            # Convert languages to Tesseract format
            lang_string = '+'.join(config.languages)
            
            # Extract text with detailed output
            data = pytesseract.image_to_data(
                image, 
                lang=lang_string,
                config=tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract full text
            full_text = pytesseract.image_to_string(
                image,
                lang=lang_string,
                config=tesseract_config
            )
            
            # Process results
            text_regions = self._process_tesseract_data(data, image.shape)
            
            # Create document layout
            layout = DocumentLayout(
                page_number=1,
                page_dimensions=(image.shape[1], image.shape[0]),
                text_regions=text_regions
            )
            
            processing_time = time.time() - start_time
            
            # Calculate confidence
            confidences = [r.confidence for r in text_regions if r.confidence > 0]
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                success=True,
                request_id=str(uuid.uuid4()),
                processing_time=processing_time,
                extracted_text=full_text,
                document_layouts=[layout],
                total_pages=1,
                method_used=OCRMethod.TESSERACT,
                quality_level=config.quality_level,
                detected_languages=config.languages,
                document_type=config.document_type,
                image_dimensions=(image.shape[0], image.shape[1]),
                overall_confidence=overall_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {str(e)}")
            raise OCRError(f"Text extraction failed: {str(e)}", "TESSERACT_EXTRACTION_FAILED")
    
    def _build_tesseract_config(self, config: OCRConfiguration) -> str:
        """Build Tesseract configuration string."""
        options = []
        
        # Page segmentation mode
        if config.document_type == DocumentType.GENERAL:
            options.append("--psm 3")  # Fully automatic page segmentation
        elif config.document_type == DocumentType.RECEIPT:
            options.append("--psm 6")  # Single uniform block of text
        elif config.document_type == DocumentType.LICENSE_PLATE:
            options.append("--psm 8")  # Single word
        else:
            options.append("--psm 3")
        
        # OCR Engine Mode
        options.append("--oem 3")  # Default, based on what is available
        
        # Character whitelist/blacklist
        if config.character_whitelist:
            options.append(f"-c tessedit_char_whitelist={config.character_whitelist}")
        if config.character_blacklist:
            options.append(f"-c tessedit_char_blacklist={config.character_blacklist}")
        
        # Preserve interword spaces
        if config.preserve_interword_spaces:
            options.append("-c preserve_interword_spaces=1")
        
        return " ".join(options)
    
    def _process_tesseract_data(self, data: Dict, image_shape: Tuple) -> List[TextRegion]:
        """Process Tesseract output data into TextRegion objects."""
        text_regions = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            
            conf = float(data['conf'][i])
            if conf < 0:  # Tesseract returns -1 for invalid detections
                continue
            
            # Create bounding box
            bbox = BoundingBox(
                x=float(data['left'][i]),
                y=float(data['top'][i]),
                width=float(data['width'][i]),
                height=float(data['height'][i]),
                confidence=conf / 100.0  # Convert to 0-1 range
            )
            
            # Create text region
            region = TextRegion(
                text=text,
                bbox=bbox,
                confidence=conf / 100.0,
                detected_by="tesseract",
                word_count=len(text.split())
            )
            
            text_regions.append(region)
        
        return text_regions
    
    def cleanup(self) -> None:
        """Cleanup Tesseract resources."""
        self.config_cache.clear()
        self.is_initialized = False


class EasyOCREngine(OCREngine):
    """EasyOCR-based OCR engine."""
    
    def __init__(self, logger):
        self.logger = logger
        self.reader = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize EasyOCR engine."""
        if not EASYOCR_AVAILABLE:
            raise OCRError("EasyOCR is not available", "EASYOCR_NOT_FOUND")
        
        try:
            self.reader = easyocr.Reader(['en'])  # Default to English
            self.is_initialized = True
            self.logger.info("EasyOCR initialized successfully")
        except Exception as e:
            raise OCRError(f"Failed to initialize EasyOCR: {str(e)}", "EASYOCR_INIT_FAILED")
    
    async def extract_text(self, image: np.ndarray, 
                          config: Optional[OCRConfiguration] = None) -> OCRResult:
        """Extract text using EasyOCR."""
        if not self.is_initialized:
            await self.initialize()
        
        config = config or OCRConfiguration()
        start_time = time.time()
        
        try:
            # Recreate reader if languages changed
            if set(config.languages) != set(self.reader.lang_list):
                self.reader = easyocr.Reader(config.languages)
            
            # Extract text with bounding boxes
            results = self.reader.readtext(
                image,
                detail=1,  # Return bounding box coordinates
                paragraph=config.enable_layout_analysis
            )
            
            # Process results
            text_regions = []
            full_text_parts = []
            
            for bbox_coords, text, confidence in results:
                if confidence < config.min_confidence:
                    continue
                
                # Convert bbox format
                bbox = self._convert_bbox_format(bbox_coords)
                
                region = TextRegion(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    detected_by="easyocr",
                    word_count=len(text.split())
                )
                
                text_regions.append(region)
                full_text_parts.append(text)
            
            # Create document layout
            layout = DocumentLayout(
                page_number=1,
                page_dimensions=(image.shape[1], image.shape[0]),
                text_regions=text_regions
            )
            
            processing_time = time.time() - start_time
            full_text = "\n".join(full_text_parts)
            
            overall_confidence = np.mean([r.confidence for r in text_regions]) if text_regions else 0.0
            
            return OCRResult(
                success=True,
                request_id=str(uuid.uuid4()),
                processing_time=processing_time,
                extracted_text=full_text,
                document_layouts=[layout],
                total_pages=1,
                method_used=OCRMethod.EASYOCR,
                quality_level=config.quality_level,
                detected_languages=config.languages,
                document_type=config.document_type,
                image_dimensions=(image.shape[0], image.shape[1]),
                overall_confidence=overall_confidence
            )
            
        except Exception as e:
            self.logger.error(f"EasyOCR failed: {str(e)}")
            raise OCRError(f"Text extraction failed: {str(e)}", "EASYOCR_EXTRACTION_FAILED")
    
    def _convert_bbox_format(self, bbox_coords: List[List[int]]) -> BoundingBox:
        """Convert EasyOCR bbox format to BoundingBox."""
        # EasyOCR returns 4 corner points
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        
        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - x
        height = max(y_coords) - y
        
        return BoundingBox(x=x, y=y, width=width, height=height)
    
    def cleanup(self) -> None:
        """Cleanup EasyOCR resources."""
        self.reader = None
        self.is_initialized = False


class ImagePreprocessor:
    """Image preprocessing for better OCR results."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def preprocess_image(self, image: np.ndarray, 
                        config: OCRConfiguration) -> Tuple[np.ndarray, List[str]]:
        """Apply preprocessing steps to improve OCR accuracy."""
        if not config.enable_preprocessing:
            return image, []
        
        processed_image = image.copy()
        applied_steps = []
        
        try:
            # Convert to grayscale if needed
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                applied_steps.append("grayscale_conversion")
            
            # Resize image if needed
            if config.resize_factor:
                height, width = processed_image.shape[:2]
                new_width = int(width * config.resize_factor)
                new_height = int(height * config.resize_factor)
                processed_image = cv2.resize(processed_image, (new_width, new_height))
                applied_steps.append(f"resize_{config.resize_factor}")
            
            # Auto-rotation and deskewing
            if config.auto_rotate or config.auto_deskew:
                processed_image, angle = self._correct_skew(processed_image)
                if abs(angle) > 0.5:
                    applied_steps.append(f"deskew_{angle:.1f}deg")
            
            # Denoising
            if config.denoise:
                processed_image = cv2.fastNlMeansDenoising(processed_image)
                applied_steps.append("denoise")
            
            # Contrast enhancement
            if config.enhance_contrast:
                processed_image = self._enhance_contrast(processed_image)
                applied_steps.append("contrast_enhancement")
            
            # Ensure image is not too large
            height, width = processed_image.shape[:2]
            max_width, max_height = config.max_image_size
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed_image = cv2.resize(processed_image, (new_width, new_height))
                applied_steps.append(f"downscale_{scale:.2f}")
            
            return processed_image, applied_steps
            
        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {str(e)}")
            return image, []
    
    def _correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct image skew."""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return image, 0.0
            
            # Calculate average angle
            angles = []
            for rho, theta in lines[:min(20, len(lines)), 0]:
                angle = theta * 180 / np.pi
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            
            if not angles:
                return image, 0.0
            
            avg_angle = np.median(angles)
            
            # Rotate image if angle is significant
            if abs(avg_angle) > 0.5:
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated, avg_angle
            
            return image, avg_angle
            
        except Exception:
            return image, 0.0
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        except Exception:
            return image


class DocumentTypeClassifier:
    """Classifies document types for optimized processing."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def classify_document(self, image: np.ndarray) -> DocumentType:
        """Classify document type based on image characteristics."""
        try:
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Simple heuristics for document classification
            if aspect_ratio > 3.0:  # Very wide - likely license plate
                return DocumentType.LICENSE_PLATE
            elif aspect_ratio > 1.5:  # Wide - likely receipt or business card
                if height < 200:
                    return DocumentType.BUSINESS_CARD
                else:
                    return DocumentType.RECEIPT
            elif 0.7 <= aspect_ratio <= 1.3:  # Square-ish - likely ID or form
                if self._has_table_structure(image):
                    return DocumentType.FORM
                else:
                    return DocumentType.ID_CARD
            else:  # Tall - likely document or invoice
                if self._has_table_structure(image):
                    return DocumentType.TABLE
                else:
                    return DocumentType.INVOICE
            
        except Exception as e:
            self.logger.warning(f"Document classification failed: {str(e)}")
            return DocumentType.GENERAL
    
    def _has_table_structure(self, image: np.ndarray) -> bool:
        """Detect if image contains table structure."""
        try:
            # Detect horizontal and vertical lines
            edges = cv2.Canny(image, 50, 150)
            
            # Horizontal kernel
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Vertical kernel
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Check for sufficient line density
            h_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
            v_density = np.sum(vertical_lines > 0) / vertical_lines.size
            
            return h_density > 0.01 and v_density > 0.01
            
        except Exception:
            return False


class EnhancedOCREngine:
    """
    Enhanced OCR Engine for AI Assistant System.
    
    Features:
    - Multiple OCR engine support with intelligent fallbacks
    - Advanced image preprocessing and enhancement
    - Document type classification and specialized processing
    - Layout analysis and structured data extraction
    - Comprehensive caching and performance optimization
    - Full integration with core system architecture
    - Real-time processing capabilities
    - Quality assessment and confidence scoring
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced OCR engine.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core dependencies - following established patterns
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Load configuration
        self._load_configuration()
        
        # Setup core components
        self._setup_core_components()
        
        # Setup OCR engines
        self._setup_ocr_engines()
        
        # Setup monitoring and caching
        self._setup_monitoring()
        self._setup_caching()
        
        # State management
        self._engines: Dict[str, OCREngine] = {}
        self._preprocessor = ImagePreprocessor(self.logger)
        self._document_classifier = DocumentTypeClassifier(self.logger)
        
        # Performance tracking
        self._performance_metrics = {
            "total_requests": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Register health check
        self.health_check.register_component("ocr_engine", self._health_check_callback)
        
        self.logger.info("EnhancedOCREngine initialized")
    
    def _load_configuration(self) -> None:
        """Load OCR configuration from config system."""
        try:
            self.ocr_config = OCRConfiguration(
                primary_method=OCRMethod(self.config.get("ocr.primary_method", "tesseract")),
                fallback_methods=[OCRMethod(m) for m in self.config.get("ocr.fallback_methods", ["easyocr"])],
                quality_level=QualityLevel(self.config.get("ocr.quality_level", "balanced")),
                languages=self.config.get("ocr.languages", ["en"]),
                enable_preprocessing=self.config.get("ocr.enable_preprocessing", True),
                enable_caching=self.config.get("ocr.enable_caching", True),
                cache_ttl=self.config.get("ocr.cache_ttl", 3600),
                enable_gpu=self.config.get("ocr.enable_gpu", True),
                timeout_seconds=self.config.get("ocr.timeout_seconds", 120.0)
            )
        except Exception as e:
            self.logger.warning(f"Failed to load OCR configuration: {str(e)}, using defaults")
            self.ocr_config = OCRConfiguration()
    
    def _setup_core_components(self) -> None:
        """Setup core system components."""
        try:
            # Optional components with graceful fallbacks
            try:
                self.feedback_processor = self.container.get(FeedbackProcessor)
            except Exception:
                self.feedback_processor = None
                self.logger.warning("FeedbackProcessor not available")
            
            try:
                self.database = self.container.get(DatabaseManager)
            except Exception:
                self.database = None
                self.logger.warning("DatabaseManager not available")
                
        except Exception as e:
            self.logger.error(f"Failed to setup core components: {str(e)}")
    
    def _setup_ocr_engines(self) -> None:
        """Setup available OCR engines."""
        self._available_engines = {}
        
        # Tesseract
        if TESSERACT_AVAILABLE:
            self._available_engines[OCRMethod.TESSERACT] = TesseractOCREngine
            
        # EasyOCR
        if EASYOCR_AVAILABLE:
            self._available_engines[OCRMethod.EASYOCR] = EasyOCREngine
        
        # Log available engines
        available = list(self._available_engines.keys())
        self.logger.info(f"Available OCR engines: {[e.value for e in available]}")
        
        if not available:
            self.logger.error("No OCR engines available!")
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register OCR-specific metrics
            self.metrics.register_counter("ocr_requests_total")
            self.metrics.register_counter("ocr_successes_total")
            self.metrics.register_counter("ocr_failures_total")
            self.metrics.register_histogram("ocr_processing_duration_seconds")
            self.metrics.register_histogram("ocr_text_length")
            self.metrics.register_gauge("ocr_average_confidence")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None
    
    def _setup_caching(self) -> None:
        """Setup caching system."""
        try:
            if self.ocr_config.enable_caching:
                self.cache = self.container.get(RedisCache)
                self.logger.info("OCR caching enabled with Redis")
            else:
                self.cache = None
                self.logger.info("OCR caching disabled")
        except Exception as e:
            self.logger.warning(f"Failed to setup caching: {str(e)}")
            self.cache = None
    
    async def initialize(self) -> None:
        """Initialize the OCR engine and all sub-engines."""
        try:
            # Initialize available engines
            for method, engine_class in self._available_engines.items():
                try:
                    engine = engine_class(self.logger)
                    await engine.initialize()
                    self._engines[method] = engine
                    self.logger.info(f"Initialized {method.value} OCR engine")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {method.value}: {str(e)}")
            
            if not self._engines:
                raise OCRError("No OCR engines could be initialized", "NO_ENGINES_AVAILABLE")
            
            # Emit initialization event
            await self.event_bus.emit(ComponentInitialized(
                component_id="ocr_engine",
                initialization_time=0.0  # Would track actual time
            ))
            
            self.logger.info("OCR engine initialization completed")
            
        except Exception as e:
            self.logger.error(f"OCR engine initialization failed: {str(e)}")
            await self.event_bus.emit(ComponentFailed(
                component_id="ocr_engine",
                error_message=str(e),
                error_type=type(e).__name__
            ))
            raise
    
    @handle_exceptions
    async def extract_text(self, image: Union[np.ndarray, str, Path], 
                          config: Optional[OCRConfiguration] = None,
                          session_id: Optional[str] = None) -> OCRResult:
        """
        Extract text from an image using the best available OCR method.
        
        Args:
            image: Image as numpy array or path to image file
            config: Optional OCR configuration
            session_id: Optional session ID for tracking
            
        Returns:
            OCRResult with extracted text and metadata
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("ocr_requests_total")
        
        self._performance_metrics["total_requests"] += 1
        
        # Use provided config or default
        ocr_config = config or self.ocr_config
        
        try:
            with self.tracer.trace("ocr_text_extraction") if self.tracer else nullcontext():
                # Load and validate image
                if isinstance(image, (str, Path)):
                    image_array = cv2.imread(str(image))
                    if image_array is None:
                        raise OCRError(f"Could not load image from {image}", "IMAGE_LOAD_FAILED")
                else:
                    image_array = image.copy()
                
                # Check cache first
                cache_key = None
                if self.cache and ocr_config.enable_caching:
                    cache_key = self._generate_cache_key(image_array, ocr_config)
                    cached_result = await self._get_cached_result(cache_key)
                    if cached_result:
                        self._performance_metrics["cache_hits"] += 1
                        return cached_result
                    self._performance_metrics["cache_misses"] += 1
                
                # Auto-detect document type if enabled
                if ocr_config.auto_detect_document_type:
                    detected_type = self._document_classifier.classify_document(image_array)
                    if detected_type != DocumentType.GENERAL:
                        ocr_config.document_type = detected_type
                        self.logger.info(f"Detected document type: {detected_type.value}")
                
                # Preprocess image
                preprocessing_start = time.time()
                processed_image, preprocessing_steps = self._preprocessor.preprocess_image(
                    image_array, ocr_config
                )
                preprocessing_time = time.time() - preprocessing_start
                
                # Select OCR method
                method = self._select_best_method(ocr_config, processed_image)
                
                # Extract text using selected method
                result = await self._extract_with_method(
                    method, processed_image, ocr_config, request_id
                )
                
                # Post-process result
                result.preprocessing_time = preprocessing_time
                result.preprocessing_steps = preprocessing_steps
                result.processing_time = time.time() - start_time
                
                # Cache result if successful
                if result.success and cache_key:
                    await self._cache_result(cache_key, result, ocr_config.cache_ttl)
                
                # Update metrics
                self._update_metrics(result)
                
                # Store for learning if available
                if self.feedback_processor:
                    await self._store_extraction_for_learning(result, ocr_config, session_id)
                
                return result
                
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            
            # Update error metrics
            if self.metrics:
                self.metrics.increment("ocr_failures_total")
            self._performance_metrics["failed_extractions"] += 1
            
            # Return error result
            return OCRResult(
                success=False,
                request_id=request_id,
                processing_time=time.time() - start_time,
                extracted_text="",
                document_layouts=[],
                total_pages=0,
                method_used=OCRMethod.TESSERACT,  # Default
                quality_level=ocr_config.quality_level,
                detected_languages=ocr_config.languages,
                document_type=ocr_config.document_type,
                image_dimensions=(0, 0),
                errors=[str(e)]
            )
    
    def _select_best_method(self, config: OCRConfiguration, 
                           image: np.ndarray) -> OCRMethod:
        """Select the best OCR method based on configuration and image characteristics."""
        # Check if primary method is available
        if config.primary_method in self._engines:
            return config.primary_method
        
        # Try fallback methods
        for method in config.fallback_methods:
            if method in self._engines:
                self.logger.warning(f"Using fallback method {method.value}")
                return method
        
        # Use any available engine
        if self._engines:
            method = next(iter(self._engines.keys()))
            self.logger.warning(f"Using default available method {method.value}")
            return method
        
        raise OCRError("No OCR engines available", "NO_ENGINES_AVAILABLE")
    
    async def _extract_with_method(self, method: OCRMethod, image: np.ndarray,
                                  config: OCRConfiguration, request_id: str) -> OCRResult:
        """Extract text using specified method with timeout and error handling."""
        engine = self._engines[method]
        
        try:
            # Extract with timeout
            result = await asyncio.wait_for(
                engine.extract_text(image, config),
                timeout=config.timeout_seconds
            )
            
            result.request_id = request_id
            return result
            
        except asyncio.TimeoutError:
            raise OCRError(f"OCR extraction timed out after {config.timeout_seconds}s", 
                          "EXTRACTION_TIMEOUT", method.value)
        except Exception as e:
            raise OCRError(f"OCR extraction failed with {method.value}: {str(e)}",
                          "EXTRACTION_FAILED", method.value)
    
    def _generate_cache_key(self, image: np.ndarray, config: OCRConfiguration) -> str:
        """Generate cache key for image and configuration."""
        # Create hash of image
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # Create hash of config
        config_dict = {
            "method": config.primary_method.value,
            "languages": config.languages,
            "document_type": config.document_type.value,
            "quality_level": config.quality_level.value,
            "preprocessing": config.enable_preprocessing
        }
        config_hash = hashlib.md5(str(sorted(config_dict.items())).encode()).hexdigest()
        
        return f"ocr:{image_hash}:{config_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[OCRResult]:
        """Get cached OCR result."""
        try:
            if self.cache:
                cached_data = await self.cache.get(cache_key)
                if cached_data:
                    try:
                        # Deserialize OCRResult from cached JSON data
                        data = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                        
                        # Reconstruct OCRResult object
                        result = OCRResult(
                            success=data.get('success', False),
                            request_id=data.get('request_id', ''),
                            processing_time=data.get('processing_time', 0.0),
                            extracted_text=data.get('extracted_text', ''),
                            document_layouts=[
                                self._deserialize_document_layout(layout_data)
                                for layout_data in data.get('document_layouts', [])
                            ],
                            total_pages=data.get('total_pages', 0),
                            method_used=OCRMethod(data.get('method_used', 'tesseract')),
                            quality_level=QualityLevel(data.get('quality_level', 'medium')),
                            detected_languages=data.get('detected_languages', []),
                            document_type=DocumentType(data.get('document_type', 'document')),
                            image_dimensions=tuple(data.get('image_dimensions', [0, 0])),
                            image_dpi=data.get('image_dpi'),
                            color_mode=data.get('color_mode', 'unknown'),
                            processing_metadata=data.get('processing_metadata', {}),
                            quality_metrics=data.get('quality_metrics', {}),
                            confidence_scores=data.get('confidence_scores', {}),
                            errors=data.get('errors', []),
                            warnings=data.get('warnings', [])
                        )
                        
                        return result
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.logger.warning(f"Failed to deserialize cached OCR result: {e}")
                        return None
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
        return None
    
    def _deserialize_document_layout(self, layout_data: Dict[str, Any]) -> 'DocumentLayout':
        """Deserialize a document layout from cached data."""
        # Import here to avoid circular imports
        from src.processing.vision.layout_analyzer import DocumentLayout, TextBlock, BoundingBox
        
        text_blocks = []
        for block_data in layout_data.get('text_blocks', []):
            bbox_data = block_data.get('bounding_box', {})
            bounding_box = BoundingBox(
                x=bbox_data.get('x', 0),
                y=bbox_data.get('y', 0),
                width=bbox_data.get('width', 0),
                height=bbox_data.get('height', 0),
                confidence=bbox_data.get('confidence', 0.0)
            )
            
            text_block = TextBlock(
                text=block_data.get('text', ''),
                bounding_box=bounding_box,
                confidence=block_data.get('confidence', 0.0),
                font_size=block_data.get('font_size'),
                font_family=block_data.get('font_family'),
                text_direction=block_data.get('text_direction', 'ltr'),
                language=block_data.get('language', 'en'),
                block_type=block_data.get('block_type', 'paragraph')
            )
            text_blocks.append(text_block)
        
        return DocumentLayout(
            page_number=layout_data.get('page_number', 1),
            page_dimensions=tuple(layout_data.get('page_dimensions', [0, 0])),
            text_blocks=text_blocks,
            detected_tables=layout_data.get('detected_tables', []),
            detected_images=layout_data.get('detected_images', []),
            reading_order=layout_data.get('reading_order', []),
            layout_confidence=layout_data.get('layout_confidence', 0.0)
        )
    
    async def _cache_result(self, cache_key: str, result: OCRResult, ttl: int) -> None:
        """Cache OCR result."""
        try:
            if self.cache:
                # Would serialize OCRResult for caching
                # This is a placeholder for actual serialization
                await self.cache.set(cache_key, str(result), ttl)
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")
    
    def _update_metrics(self, result: OCRResult) -> None:
        """Update performance metrics."""
        if result.success:
            self._performance_metrics["successful_extractions"] += 1
            if self.metrics:
                self.metrics.increment("ocr_successes_total")
                self.metrics.record("ocr_processing_duration_seconds", result.processing_time)
                self.metrics.record("ocr_text_length", len(result.extracted_text))
                self.metrics.set("ocr_average_confidence", result.overall_confidence)
        
        # Update average processing time
        total = self._performance_metrics["total_requests"]
        current_avg = self._performance_metrics["average_processing_time"]
        self._performance_metrics["average_processing_time"] = (
            (current_avg * (total - 1) + result.processing_time) / total
        )
    
    async def _store_extraction_for_learning(self, result: OCRResult, 
                                           config: OCRConfiguration,
                                           session_id: Optional[str]) -> None:
        """Store extraction result for learning and improvement."""
        try:
            if self.feedback_processor:
                await self.feedback_processor.process_ocr_extraction(
                    result, config, session_id
                )
        except Exception as e:
            self.logger.warning(f"Failed to store extraction for learning: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Engine status
            engine_status = {}
            for method, engine in self._engines.items():
                try:
                    # Would check engine health
                    engine_status[method.value] = {"status": "healthy", "initialized": True}
                except Exception as e:
                    engine_status[method.value] = {"status": "unhealthy", "error": str(e)}
            
            return {
                "service": "ocr_engine",
                "status": "healthy" if engine_status else "unhealthy",
                "engines": engine_status,
                "available_engines": [e.value for e in self._engines.keys()],
                "configuration": {
                    "primary_method": self.ocr_config.primary_method.value,
                    "fallback_methods": [m.value for m in self.ocr_config.fallback_methods],
                    "languages": self.ocr_config.languages,
                    "caching_enabled": self.ocr_config.enable_caching
                },
                "performance_metrics": self._performance_metrics,
                "uptime": time.time()  # Would track actual uptime
            }
            
        except Exception as e:
            return {
                "service": "ocr_engine",
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the health monitoring system."""
        try:
            total_engines = len(self._available_engines)
            healthy_engines = len(self._engines)
            
            health_score = healthy_engines / total_engines if total_engines > 0 else 0.0
            status = "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy"
            
            return {
                "status": status,
                "health_score": health_score,
                "total_engines": total_engines,
                "healthy_engines": healthy_engines,
                "performance_metrics": self._performance_metrics
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup all OCR engines and resources."""
        try:
            # Cleanup all engines
            for engine in self._engines.values():
                try:
                    engine.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up engine: {str(e)}")
            
            self._engines.clear()
            self.logger.info("OCR engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during OCR engine cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_engines') and self._engines:
                self.logger.warning("OCREngine destroyed with active engines")
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Null context manager for when tracer is not available
class nullcontext:
    """Null context manager for when tracer is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
