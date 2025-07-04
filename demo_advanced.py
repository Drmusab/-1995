#!/usr/bin/env python3
"""
Advanced Skill Factory Demo
Author: Drmusab
Last Modified: 2025-01-20 12:30:00 UTC

This script demonstrates advanced features of the skill factory including
skill composition, circuit breaker patterns, and performance monitoring.
"""

import asyncio
import json
from pathlib import Path
from test_standalone import (
    SimpleSkillRegistry, SimpleSkillFactory, SkillInterface, SkillMetadata, 
    SkillType, SkillCapability, SkillExecutionContext, MockLogger
)
from datetime import datetime, timezone
from typing import Any, Dict

class TextProcessorSkill(SkillInterface):
    """Skill that processes text input."""
    
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="nlp.text_processor",
            name="Text Processor",
            version="1.0.0",
            description="Processes and cleans text input",
            author="Demo",
            skill_type=SkillType.CUSTOM,
            capabilities=[
                SkillCapability(
                    name="process_text",
                    description="Process and clean text",
                    input_types=["string", "dict"],
                    output_types=["dict"]
                )
            ],
            tags=["nlp", "text", "preprocessing"]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.processing_count = 0
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        self.processing_count += 1
        
        # Extract text from input
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, dict) and 'text' in input_data:
            text = input_data['text']
        else:
            raise ValueError("Input must be string or dict with 'text' key")
        
        # Simulate text processing
        processed_text = text.strip().lower()
        word_count = len(processed_text.split())
        
        await asyncio.sleep(0.01)  # Simulate processing time
        
        return {
            "original_text": text,
            "processed_text": processed_text,
            "word_count": word_count,
            "processing_count": self.processing_count,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

class SentimentAnalyzerSkill(SkillInterface):
    """Skill that analyzes sentiment."""
    
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="nlp.sentiment_analyzer",
            name="Sentiment Analyzer",
            version="1.0.0",
            description="Analyzes text sentiment",
            author="Demo",
            skill_type=SkillType.CUSTOM,
            capabilities=[
                SkillCapability(
                    name="analyze_sentiment",
                    description="Analyze text sentiment",
                    input_types=["dict"],
                    output_types=["dict"]
                )
            ],
            tags=["nlp", "sentiment", "analysis"]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.analysis_count = 0
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        self.analysis_count += 1
        
        if not isinstance(input_data, dict) or 'processed_text' not in input_data:
            raise ValueError("Input must be dict with 'processed_text' key")
        
        text = input_data['processed_text']
        
        # Simple sentiment analysis (demo)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry']
        
        words = text.split()
        positive_score = sum(1 for word in words if word in positive_words)
        negative_score = sum(1 for word in words if word in negative_words)
        
        if positive_score > negative_score:
            sentiment = "positive"
            confidence = positive_score / len(words) if words else 0
        elif negative_score > positive_score:
            sentiment = "negative"
            confidence = negative_score / len(words) if words else 0
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        await asyncio.sleep(0.005)  # Simulate analysis time
        
        result = input_data.copy()
        result.update({
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "analysis_count": self.analysis_count,
            "analyzed_at": datetime.now(timezone.utc).isoformat()
        })
        
        return result

class UnreliableSkill(SkillInterface):
    """Skill that fails periodically to test circuit breaker."""
    
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="test.unreliable",
            name="Unreliable Skill",
            version="1.0.0",
            description="Skill that fails periodically for testing",
            author="Demo",
            skill_type=SkillType.CUSTOM,
            capabilities=[
                SkillCapability(
                    name="unreliable_process",
                    description="Process that may fail",
                    input_types=["any"],
                    output_types=["dict"]
                )
            ]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.execution_count = 0
        self.failure_rate = config.get('failure_rate', 0.3)  # 30% failure rate
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        self.execution_count += 1
        
        # Simulate failure based on execution count
        if self.execution_count % 3 == 0:  # Fail every 3rd execution
            raise Exception(f"Simulated failure on execution {self.execution_count}")
        
        return {
            "result": f"Success on execution {self.execution_count}",
            "input": input_data,
            "execution_count": self.execution_count
        }

async def demo_advanced_features():
    """Demonstrate advanced skill factory features."""
    print("=" * 70)
    print("ADVANCED SKILL FACTORY DEMO")
    print("=" * 70)
    
    # Initialize components
    print("\n1. Setting up Skill System...")
    skill_registry = SimpleSkillRegistry()
    skill_factory = SimpleSkillFactory(skill_registry)
    
    # Register multiple skills
    skills_to_register = [
        ("nlp.text_processor", TextProcessorSkill),
        ("nlp.sentiment_analyzer", SentimentAnalyzerSkill),
        ("test.unreliable", UnreliableSkill)
    ]
    
    for skill_id, skill_class in skills_to_register:
        success = await skill_registry.register_skill(skill_id, skill_class)
        print(f"  ✓ Registered {skill_id}: {success}")
    
    # Demo 1: Basic Skill Execution
    print("\n2. Basic Skill Execution...")
    context = SkillExecutionContext(
        session_id="demo-session",
        user_id="demo-user"
    )
    
    text_input = "This is a wonderful day! I feel amazing and happy."
    
    # Process text
    result1 = await skill_factory.execute_skill(
        "nlp.text_processor", 
        {"text": text_input}, 
        context
    )
    print(f"  Text Processing: {result1.success} ({result1.execution_time_ms:.1f}ms)")
    if result1.success:
        print(f"    Word count: {result1.result['word_count']}")
        print(f"    Processed: {result1.result['processed_text'][:50]}...")
    
    # Analyze sentiment
    if result1.success:
        result2 = await skill_factory.execute_skill(
            "nlp.sentiment_analyzer",
            result1.result,
            context
        )
        print(f"  Sentiment Analysis: {result2.success} ({result2.execution_time_ms:.1f}ms)")
        if result2.success:
            print(f"    Sentiment: {result2.result['sentiment']} "
                  f"(confidence: {result2.result['confidence']:.2f})")
    
    # Demo 2: Skill Composition Pipeline
    print("\n3. Skill Composition Pipeline...")
    texts_to_analyze = [
        "This product is absolutely terrible and awful!",
        "I love this amazing software, it's fantastic!",
        "The weather is okay today, nothing special.",
        "Best purchase ever! Highly recommend this excellent tool."
    ]
    
    for i, text in enumerate(texts_to_analyze):
        print(f"\n  Pipeline {i+1}: '{text[:40]}...'")
        
        # Step 1: Process text
        step1 = await skill_factory.execute_skill(
            "nlp.text_processor",
            {"text": text},
            context
        )
        
        if step1.success:
            # Step 2: Analyze sentiment
            step2 = await skill_factory.execute_skill(
                "nlp.sentiment_analyzer",
                step1.result,
                context
            )
            
            if step2.success:
                result = step2.result
                print(f"    → {result['sentiment'].upper()} "
                      f"(conf: {result['confidence']:.2f}, "
                      f"words: {result['word_count']})")
            else:
                print(f"    → Sentiment analysis failed: {step2.error}")
        else:
            print(f"    → Text processing failed: {step1.error}")
    
    # Demo 3: Circuit Breaker Testing
    print("\n4. Circuit Breaker Testing...")
    print("  Testing unreliable skill (fails every 3rd execution):")
    
    for i in range(8):
        result = await skill_factory.execute_skill(
            "test.unreliable",
            {"attempt": i + 1},
            context
        )
        
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        print(f"    Attempt {i+1}: {status} ({result.execution_time_ms:.1f}ms)")
        
        if not result.success:
            print(f"      Error: {result.error}")
    
    # Demo 4: Performance Statistics
    print("\n5. Performance Statistics...")
    stats = skill_factory.get_statistics()
    print(f"  Active Skills: {stats['active_skills']}")
    print(f"  Total Executions: {stats['total_executions']}")
    print(f"  Successful Executions: {stats['successful_executions']}")
    
    success_rate = (stats['successful_executions'] / stats['total_executions']) * 100
    print(f"  Overall Success Rate: {success_rate:.1f}%")
    
    # Demo 5: Execution History Analysis
    print("\n6. Execution History Analysis...")
    history = skill_factory.execution_history
    
    # Group by skill
    skill_stats = {}
    for result in history:
        if result.skill_id not in skill_stats:
            skill_stats[result.skill_id] = {
                'total': 0, 
                'successful': 0, 
                'total_time': 0,
                'avg_time': 0
            }
        
        stats = skill_stats[result.skill_id]
        stats['total'] += 1
        if result.success:
            stats['successful'] += 1
        stats['total_time'] += result.execution_time_ms
        stats['avg_time'] = stats['total_time'] / stats['total']
    
    for skill_id, stats in skill_stats.items():
        success_rate = (stats['successful'] / stats['total']) * 100
        print(f"  {skill_id}:")
        print(f"    Executions: {stats['total']} "
              f"(success rate: {success_rate:.1f}%)")
        print(f"    Avg execution time: {stats['avg_time']:.2f}ms")
    
    # Demo 6: Concurrent Execution
    print("\n7. Concurrent Execution Test...")
    start_time = asyncio.get_event_loop().time()
    
    # Execute multiple skills concurrently
    tasks = []
    for i in range(5):
        task = skill_factory.execute_skill(
            "nlp.text_processor",
            {"text": f"Concurrent execution test {i}"},
            context
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = asyncio.get_event_loop().time()
    
    successful = sum(1 for r in results if isinstance(r, object) and hasattr(r, 'success') and r.success)
    total_time = (end_time - start_time) * 1000
    
    print(f"  Concurrent executions: {len(tasks)}")
    print(f"  Successful: {successful}")
    print(f"  Total time: {total_time:.1f}ms")
    print(f"  Average time per execution: {total_time/len(tasks):.1f}ms")
    
    print("\n" + "=" * 70)
    print("✓ ADVANCED DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Final statistics
    final_stats = skill_factory.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Skills Created: {final_stats['active_skills']}")
    print(f"  Total Executions: {final_stats['total_executions']}")
    print(f"  Overall Success Rate: {(final_stats['successful_executions']/final_stats['total_executions']*100):.1f}%")

if __name__ == "__main__":
    asyncio.run(demo_advanced_features())