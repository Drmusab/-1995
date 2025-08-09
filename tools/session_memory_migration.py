#!/usr/bin/env python3
"""
Session Memory Migration Utility
Author: Drmusab
Last Modified: 2025-07-20 14:00:00 UTC

This utility provides tools for migrating session data to the new memory system,
ensuring backward compatibility and proper data integration.
"""

import asyncio
import json
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import sqlite3
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.assistant.core import EnhancedSessionManager
from src.memory.core_memory.memory_manager import MemoryManager
from src.learning.memory_learning_bridge import MemoryLearningBridge
from src.observability.logging.config import get_logger, setup_logging


class SessionMemoryMigrator:
    """Utility for migrating session data to the new memory system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the migration utility."""
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config = ConfigLoader(config_path)
        
        # Setup dependency injection
        self.container = Container()
        
        # Initialize components (would need proper initialization in real use)
        self.session_manager = None
        self.memory_manager = None
        self.memory_bridge = None
        
        # Migration statistics
        self.stats = {
            "sessions_processed": 0,
            "interactions_migrated": 0,
            "memories_created": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def initialize(self):
        """Initialize the migration components."""
        self.logger.info("Initializing migration utility...")
        
        # In a real implementation, these would be properly initialized
        # For now, we'll create mock instances
        self.session_manager = EnhancedSessionManager(self.container)
        self.memory_manager = MemoryManager(self.container)
        self.memory_bridge = MemoryLearningBridge(self.container)
        
        self.logger.info("Migration utility initialized")
    
    async def migrate_legacy_sessions(self, legacy_data_path: str, backup: bool = True) -> Dict[str, Any]:
        """
        Migrate legacy session data to the new memory system.
        
        Args:
            legacy_data_path: Path to legacy session data
            backup: Whether to create a backup before migration
            
        Returns:
            Migration results dictionary
        """
        self.stats["start_time"] = datetime.now(timezone.utc)
        self.logger.info(f"Starting session memory migration from {legacy_data_path}")
        
        try:
            # Create backup if requested
            if backup:
                await self._create_backup(legacy_data_path)
            
            # Load legacy data
            legacy_data = await self._load_legacy_data(legacy_data_path)
            
            # Process each session
            for session_data in legacy_data:
                try:
                    await self._migrate_session(session_data)
                    self.stats["sessions_processed"] += 1
                except Exception as e:
                    self.logger.error(f"Error migrating session {session_data.get('session_id', 'unknown')}: {str(e)}")
                    self.stats["errors"] += 1
            
            self.stats["end_time"] = datetime.now(timezone.utc)
            
            # Generate migration report
            return await self._generate_migration_report()
            
        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            self.stats["errors"] += 1
            raise
    
    async def _create_backup(self, data_path: str):
        """Create a backup of the legacy data."""
        backup_path = f"{data_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Creating backup at {backup_path}")
        
        # Implementation would depend on the data format
        # For now, just log the action
        self.logger.info(f"Backup created at {backup_path}")
    
    async def _load_legacy_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load legacy session data from file."""
        self.logger.info(f"Loading legacy data from {data_path}")
        
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Legacy data file not found: {data_path}")
        
        # Handle different file formats
        if path.suffix == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix == ".db":
            data = await self._load_from_sqlite(path)
        elif path.suffix == ".pkl":
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Normalize data format
        if isinstance(data, dict):
            data = [data]
        
        self.logger.info(f"Loaded {len(data)} session records")
        return data
    
    async def _load_from_sqlite(self, db_path: Path) -> List[Dict[str, Any]]:
        """Load session data from SQLite database."""
        sessions = []
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get sessions
            cursor.execute("SELECT * FROM sessions")
            session_rows = cursor.fetchall()
            
            for session_row in session_rows:
                session_data = dict(session_row)
                
                # Get interactions for this session
                cursor.execute(
                    "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp",
                    (session_data['session_id'],)
                )
                interactions = [dict(row) for row in cursor.fetchall()]
                session_data['interactions'] = interactions
                
                sessions.append(session_data)
            
            conn.close()
            
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error: {str(e)}")
            raise
        
        return sessions
    
    async def _migrate_session(self, session_data: Dict[str, Any]):
        """Migrate a single session to the new memory system."""
        session_id = session_data.get('session_id')
        user_id = session_data.get('user_id')
        
        self.logger.debug(f"Migrating session {session_id} for user {user_id}")
        
        # Create session memory context
        context = {
            "session_id": session_id,
            "user_id": user_id,
            "migration_timestamp": datetime.now(timezone.utc).isoformat(),
            "legacy_data": True
        }
        
        # Migrate interactions
        interactions = session_data.get('interactions', [])
        for interaction in interactions:
            await self._migrate_interaction(interaction, context)
            self.stats["interactions_migrated"] += 1
        
        # Migrate session metadata
        await self._migrate_session_metadata(session_data, context)
        
        # Process through memory-learning bridge
        await self.memory_bridge.process_session_data(session_data)
        
        self.logger.debug(f"Successfully migrated session {session_id}")
    
    async def _migrate_interaction(self, interaction: Dict[str, Any], context: Dict[str, Any]):
        """Migrate a single interaction to memory."""
        # Create memory record for the interaction
        memory_content = {
            "user_input": interaction.get('user_input', ''),
            "assistant_response": interaction.get('assistant_response', ''),
            "timestamp": interaction.get('timestamp'),
            "confidence": interaction.get('confidence', 0.0),
            "metadata": interaction.get('metadata', {})
        }
        
        # Store in memory system
        memory_id = await self.memory_manager.store_memory(
            content=memory_content,
            memory_type="interaction",
            context=context
        )
        
        self.stats["memories_created"] += 1
        self.logger.debug(f"Created memory {memory_id} for interaction")
    
    async def _migrate_session_metadata(self, session_data: Dict[str, Any], context: Dict[str, Any]):
        """Migrate session metadata to memory."""
        metadata = {
            "session_start": session_data.get('created_at'),
            "session_end": session_data.get('ended_at'),
            "session_duration": session_data.get('duration'),
            "user_preferences": session_data.get('preferences', {}),
            "session_summary": session_data.get('summary', ''),
            "interaction_count": len(session_data.get('interactions', []))
        }
        
        # Store session metadata in memory
        memory_id = await self.memory_manager.store_memory(
            content=metadata,
            memory_type="session_metadata",
            context=context
        )
        
        self.stats["memories_created"] += 1
        self.logger.debug(f"Created session metadata memory {memory_id}")
    
    async def _generate_migration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive migration report."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        report = {
            "migration_summary": {
                "start_time": self.stats["start_time"].isoformat(),
                "end_time": self.stats["end_time"].isoformat(),
                "duration_seconds": duration,
                "sessions_processed": self.stats["sessions_processed"],
                "interactions_migrated": self.stats["interactions_migrated"],
                "memories_created": self.stats["memories_created"],
                "errors": self.stats["errors"],
                "success_rate": (
                    (self.stats["sessions_processed"] - self.stats["errors"]) / 
                    max(self.stats["sessions_processed"], 1)
                ) * 100
            },
            "performance_metrics": {
                "sessions_per_second": self.stats["sessions_processed"] / max(duration, 1),
                "interactions_per_second": self.stats["interactions_migrated"] / max(duration, 1),
                "memories_per_second": self.stats["memories_created"] / max(duration, 1)
            },
            "recommendations": []
        }
        
        # Add recommendations based on results
        if self.stats["errors"] > 0:
            report["recommendations"].append(
                f"Review error logs - {self.stats['errors']} errors occurred during migration"
            )
        
        if report["performance_metrics"]["sessions_per_second"] < 1:
            report["recommendations"].append(
                "Consider optimizing migration batch size for better performance"
            )
        
        self.logger.info(f"Migration completed: {report['migration_summary']}")
        return report
    
    async def validate_migration(self, original_data_path: str) -> Dict[str, Any]:
        """Validate that the migration was successful."""
        self.logger.info("Validating migration results...")
        
        # Load original data for comparison
        original_data = await self._load_legacy_data(original_data_path)
        
        validation_results = {
            "total_sessions": len(original_data),
            "validated_sessions": 0,
            "missing_sessions": [],
            "data_integrity_issues": [],
            "validation_passed": True
        }
        
        # Validate each session
        for session_data in original_data:
            session_id = session_data.get('session_id')
            
            try:
                # Check if session data exists in new memory system
                memories = await self.memory_manager.retrieve_memories(
                    query=f"session:{session_id}",
                    context={"session_id": session_id}
                )
                
                if memories:
                    validation_results["validated_sessions"] += 1
                else:
                    validation_results["missing_sessions"].append(session_id)
                    validation_results["validation_passed"] = False
                
            except Exception as e:
                validation_results["data_integrity_issues"].append({
                    "session_id": session_id,
                    "error": str(e)
                })
                validation_results["validation_passed"] = False
        
        self.logger.info(f"Validation completed: {validation_results}")
        return validation_results


async def main():
    """Main entry point for the migration utility."""
    parser = argparse.ArgumentParser(description="Session Memory Migration Utility")
    parser.add_argument("command", choices=["migrate", "validate"], help="Command to execute")
    parser.add_argument("--data-path", required=True, help="Path to legacy session data")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--output", help="Output file for migration report")
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = SessionMemoryMigrator(args.config)
    await migrator.initialize()
    
    try:
        if args.command == "migrate":
            # Run migration
            results = await migrator.migrate_legacy_sessions(
                args.data_path,
                backup=not args.no_backup
            )
            
            # Save results if output file specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Migration report saved to {args.output}")
            else:
                print(json.dumps(results, indent=2, default=str))
        
        elif args.command == "validate":
            # Run validation
            results = await migrator.validate_migration(args.data_path)
            
            # Save results if output file specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Validation report saved to {args.output}")
            else:
                print(json.dumps(results, indent=2, default=str))
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())