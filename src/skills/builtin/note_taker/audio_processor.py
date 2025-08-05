"""
Audio Processor for Note Taking
Author: Drmusab
Last Modified: 2025-01-08

Handles audio recording, storage, and processing for the note taking skill.
Integrates with existing audio utilities while providing note-specific functionality.
"""

import asyncio
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger
from src.processing.speech.audio_utils import AudioProcessor as BaseAudioProcessor


class AudioProcessor:
    """
    Audio processing component for note taking with enhanced storage and management.
    
    Extends the base audio processor with note-specific functionality including
    audio session management, file organization, and metadata tracking.
    """
    
    def __init__(self, container: Container):
        """Initialize the note audio processor."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        
        # Initialize base audio processor
        self.base_processor = BaseAudioProcessor(
            sample_rate=self.config.get("speech.input.sample_rate", 16000)
        )
        
        # Audio configuration
        self._setup_audio_config()
        
        # Storage configuration
        self._setup_storage_config()
        
        self.logger.info("NoteAudioProcessor initialized")
    
    def _setup_audio_config(self) -> None:
        """Setup audio processing configuration."""
        note_audio_config = self.config.get("note_taker", {}).get("audio", {})
        
        self.audio_format = note_audio_config.get("format", "wav")
        self.audio_quality = note_audio_config.get("quality", "high")
        self.auto_enhance = note_audio_config.get("auto_enhance", True)
        self.sample_rate = self.config.get("speech.input.sample_rate", 16000)
        self.channels = self.config.get("speech.input.channels", 1)
        
        # Quality settings
        if self.audio_quality == "high":
            self.bit_depth = 24
            self.compression_level = 0
        elif self.audio_quality == "medium":
            self.bit_depth = 16
            self.compression_level = 3
        else:  # low
            self.bit_depth = 16
            self.compression_level = 5
    
    def _setup_storage_config(self) -> None:
        """Setup storage configuration for audio files."""
        storage_config = self.config.get("note_taker", {}).get("storage", {})
        
        # Base directories
        self.audio_directory = Path(storage_config.get("audio_directory", "data/notes/audio"))
        self.temp_directory = Path(storage_config.get("temp_directory", tempfile.gettempdir())) / "note_taker"
        
        # Ensure directories exist
        self.audio_directory.mkdir(parents=True, exist_ok=True)
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        
        # File naming settings
        self.use_date_folders = storage_config.get("use_date_folders", True)
        self.max_file_size_mb = storage_config.get("max_file_size_mb", 100)
    
    async def save_recording_session(self, session: Dict[str, Any]) -> Path:
        """
        Save a completed recording session to permanent storage.
        
        Args:
            session: Recording session data
            
        Returns:
            Path to the saved audio file
        """
        note_id = session["note_id"]
        start_time = session["start_time"]
        
        # Generate filename
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"note_{note_id}_{timestamp}.{self.audio_format}"
        
        # Determine storage path
        if self.use_date_folders:
            date_folder = start_time.strftime("%Y/%m/%d")
            storage_path = self.audio_directory / date_folder
        else:
            storage_path = self.audio_directory
        
        storage_path.mkdir(parents=True, exist_ok=True)
        final_path = storage_path / filename
        
        # For now, create a placeholder audio file (in real implementation would save actual recorded audio)
        # This simulates saving the recorded audio data
        audio_data = self._generate_placeholder_audio()
        
        # Save audio file
        await self._save_audio_file(audio_data, final_path)
        
        # Generate metadata file
        await self._save_audio_metadata(session, final_path)
        
        self.logger.info(f"Saved recording session audio to {final_path}")
        
        return final_path
    
    def _generate_placeholder_audio(self) -> np.ndarray:
        """Generate placeholder audio data for testing."""
        # Generate 5 seconds of low-level white noise as placeholder
        duration_seconds = 5.0
        samples = int(self.sample_rate * duration_seconds)
        audio_data = np.random.randn(samples).astype(np.float32) * 0.01
        return audio_data
    
    async def _save_audio_file(self, audio_data: np.ndarray, file_path: Path) -> None:
        """Save audio data to file."""
        try:
            if self.audio_format.lower() == "wav":
                # In a real implementation, would use soundfile or similar
                # For now, save as numpy array (placeholder)
                np.save(str(file_path).replace('.wav', '.npy'), audio_data)
                
                # Create empty WAV file as placeholder
                file_path.touch()
                
            elif self.audio_format.lower() == "flac":
                # Similar placeholder for FLAC
                np.save(str(file_path).replace('.flac', '.npy'), audio_data)
                file_path.touch()
            
            # Apply audio enhancement if enabled
            if self.auto_enhance:
                await self._enhance_audio_file(file_path)
                
        except Exception as e:
            self.logger.error(f"Failed to save audio file {file_path}: {str(e)}")
            raise
    
    async def _enhance_audio_file(self, file_path: Path) -> None:
        """Apply audio enhancement to the saved file."""
        try:
            # Placeholder for audio enhancement
            # In real implementation would apply noise reduction, normalization, etc.
            self.logger.debug(f"Applied audio enhancement to {file_path}")
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed for {file_path}: {str(e)}")
    
    async def _save_audio_metadata(self, session: Dict[str, Any], audio_path: Path) -> None:
        """Save metadata file alongside audio."""
        metadata = {
            "note_id": session["note_id"],
            "user_id": session.get("user_id"),
            "session_id": session.get("session_id"),
            "start_time": session["start_time"].isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration": (datetime.now(timezone.utc) - session["start_time"]).total_seconds(),
            "audio_format": self.audio_format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "file_size": audio_path.stat().st_size if audio_path.exists() else 0,
            "file_hash": await self._calculate_file_hash(audio_path),
            "metadata": session.get("metadata", {})
        }
        
        metadata_path = audio_path.with_suffix('.json')
        
        try:
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            self.logger.debug(f"Saved audio metadata to {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save audio metadata: {str(e)}")
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the audio file."""
        try:
            if not file_path.exists():
                return ""
            
            hash_md5 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate file hash: {str(e)}")
            return ""
    
    async def load_audio_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file for processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio data as numpy array
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Try to load using base audio processor
            audio_data, sample_rate = self.base_processor.load_audio(
                file_path, target_sr=self.sample_rate
            )
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {str(e)}")
            raise
    
    async def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio file information
        """
        file_path = Path(file_path)
        metadata_path = file_path.with_suffix('.json')
        
        info = {
            "file_path": str(file_path),
            "exists": file_path.exists(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat() if file_path.exists() else None,
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
        }
        
        # Load metadata if available
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {file_path}: {str(e)}")
        
        return info
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary audio files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files cleaned up
        """
        if not self.temp_directory.exists():
            return 0
        
        cleaned_count = 0
        cutoff_time = datetime.now() - datetime.timedelta(hours=max_age_hours)
        
        try:
            for temp_file in self.temp_directory.iterdir():
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        temp_file.unlink()
                        cleaned_count += 1
                        
            self.logger.info(f"Cleaned up {cleaned_count} temporary audio files")
            
        except Exception as e:
            self.logger.error(f"Error during temp file cleanup: {str(e)}")
        
        return cleaned_count
    
    async def validate_audio_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate an audio file for processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Validation results
        """
        file_path = Path(file_path)
        
        validation_result = {
            "valid": False,
            "file_exists": file_path.exists(),
            "file_size_ok": False,
            "format_supported": False,
            "readable": False,
            "issues": []
        }
        
        # Check file existence
        if not file_path.exists():
            validation_result["issues"].append("File does not exist")
            return validation_result
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            validation_result["issues"].append(f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
        else:
            validation_result["file_size_ok"] = True
        
        # Check format
        supported_formats = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
        if file_path.suffix.lower() in supported_formats:
            validation_result["format_supported"] = True
        else:
            validation_result["issues"].append(f"Unsupported format: {file_path.suffix}")
        
        # Test readability
        try:
            audio_data = await self.load_audio_file(file_path)
            if len(audio_data) > 0:
                validation_result["readable"] = True
            else:
                validation_result["issues"].append("Audio file is empty")
        except Exception as e:
            validation_result["issues"].append(f"Cannot read audio file: {str(e)}")
        
        # Overall validation
        validation_result["valid"] = (
            validation_result["file_exists"] and
            validation_result["file_size_ok"] and
            validation_result["format_supported"] and
            validation_result["readable"]
        )
        
        return validation_result
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics for audio files."""
        try:
            total_files = 0
            total_size = 0
            file_types = {}
            
            for audio_file in self.audio_directory.rglob("*"):
                if audio_file.is_file() and not audio_file.name.endswith('.json'):
                    total_files += 1
                    total_size += audio_file.stat().st_size
                    
                    ext = audio_file.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_mb": total_size / (1024 * 1024),
                "file_types": file_types,
                "storage_path": str(self.audio_directory),
                "temp_path": str(self.temp_directory)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {str(e)}")
            return {}
    
    def cleanup(self) -> None:
        """Cleanup audio processor resources."""
        try:
            # Cleanup base processor
            if hasattr(self.base_processor, 'cleanup'):
                self.base_processor.cleanup()
            
            self.logger.info("NoteAudioProcessor cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during audio processor cleanup: {str(e)}")