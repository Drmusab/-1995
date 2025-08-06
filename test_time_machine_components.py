#!/usr/bin/env python3
"""
Simple component test for Time Machine skill
Author: Drmusab
Last Modified: 2025-01-08

Tests individual components of the Time Machine skill without full system dependencies.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_query_parser():
    """Test the query parser component."""
    print("=== Testing Query Parser ===")
    
    try:
        # Mock container
        class MockContainer:
            def get(self, service_type):
                if 'BilingualManager' in str(service_type):
                    from src.processing.natural_language.bilingual_manager import Language
                    
                    class MockBilingualManager:
                        def detect_language(self, text):
                            arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
                            english_chars = len([c for c in text if c.isalpha() and c.isascii()])
                            return Language.ARABIC if arabic_chars > english_chars else Language.ENGLISH
                    
                    return MockBilingualManager()
                else:
                    class GenericMock:
                        pass
                    return GenericMock()
        
        container = MockContainer()
        
        # Import and test query parser
        from src.skills.builtin.time_machine.query_parser import QueryParser, QueryType
        
        parser = QueryParser(container)
        
        # Test Arabic query
        arabic_query = "ماذا تحدثنا عنه الأسبوع الماضي؟"
        parsed = await parser.parse_query(arabic_query)
        
        print(f"Arabic Query: {arabic_query}")
        print(f"Detected Language: {parsed.language.value}")
        print(f"Query Type: {parsed.query_type.value}")
        print(f"Intent Confidence: {parsed.intent_confidence:.2f}")
        print(f"Time Range: {parsed.time_range.original_text if parsed.time_range else 'None'}")
        
        # Test English query
        english_query = "What did we talk about last week?"
        parsed = await parser.parse_query(english_query)
        
        print(f"\nEnglish Query: {english_query}")
        print(f"Detected Language: {parsed.language.value}")
        print(f"Query Type: {parsed.query_type.value}")
        print(f"Intent Confidence: {parsed.intent_confidence:.2f}")
        print(f"Time Range: {parsed.time_range.original_text if parsed.time_range else 'None'}")
        
        print("\nQuery Parser test passed!")
        return True
        
    except Exception as e:
        print(f"Query Parser test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_reflection_analyzer():
    """Test the reflection analyzer component."""
    print("\n=== Testing Reflection Analyzer ===")
    
    try:
        # Mock container
        class MockContainer:
            def get(self, service_type):
                if 'BilingualManager' in str(service_type):
                    from src.processing.natural_language.bilingual_manager import Language
                    
                    class MockBilingualManager:
                        def detect_language(self, text):
                            arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
                            english_chars = len([c for c in text if c.isalpha() and c.isascii()])
                            return Language.ARABIC if arabic_chars > english_chars else Language.ENGLISH
                    
                    return MockBilingualManager()
                else:
                    class GenericMock:
                        pass
                    return GenericMock()
        
        container = MockContainer()
        
        # Import analyzer and create mock conversation data
        from src.skills.builtin.time_machine.reflection_analyzer import ReflectionAnalyzer
        from src.skills.builtin.time_machine.memory_replayer import ConversationThread, ConversationSegment
        from src.processing.natural_language.bilingual_manager import Language
        
        analyzer = ReflectionAnalyzer(container)
        
        # Create mock conversation threads
        segments = [
            ConversationSegment(
                id="seg1",
                content="أنا سعيد جداً بالتقدم في المشروع",
                timestamp=datetime.now(timezone.utc),
                participants=["user", "assistant"],
                topics=["project"],
                importance=0.8
            ),
            ConversationSegment(
                id="seg2", 
                content="أعتقد أننا نحرز تقدماً ممتازاً",
                timestamp=datetime.now(timezone.utc),
                participants=["user", "assistant"],
                topics=["project"],
                importance=0.7
            )
        ]
        
        thread = ConversationThread(
            id="thread1",
            title="مناقشة المشروع",
            segments=segments,
            participants={"user", "assistant"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            total_duration=300,
            topics={"project"}
        )
        
        # Analyze behavioral patterns
        analysis = await analyzer.analyze_behavioral_patterns(
            conversation_threads=[thread],
            language=Language.ARABIC,
            user_id="test_user"
        )
        
        print(f"Analysis completed with confidence: {analysis.confidence_score:.2f}")
        print(f"Number of trends identified: {len(analysis.trends)}")
        
        if analysis.overall_insights:
            print("Overall insights:")
            for insight in analysis.overall_insights:
                print(f"- {insight}")
        
        if analysis.recommendations:
            print("Recommendations:")
            for rec in analysis.recommendations:
                print(f"- {rec}")
        
        print("\nReflection Analyzer test passed!")
        return True
        
    except Exception as e:
        print(f"Reflection Analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_visualization_engine():
    """Test the visualization engine component."""
    print("\n=== Testing Visualization Engine ===")
    
    try:
        # Mock container  
        class MockContainer:
            def get(self, service_type):
                class GenericMock:
                    pass
                return GenericMock()
        
        container = MockContainer()
        
        # Import and test visualization engine
        from src.skills.builtin.time_machine.visualization import VisualizationEngine, ChartType, ExportFormat
        from src.skills.builtin.time_machine.reflection_analyzer import (
            BehavioralAnalysis, BehavioralTrend, BehavioralMetric, TrendDirection, BehavioralDataPoint
        )
        from src.processing.natural_language.bilingual_manager import Language
        
        engine = VisualizationEngine(container)
        
        # Create mock analysis data
        data_points = [
            BehavioralDataPoint(
                timestamp=datetime.now(timezone.utc),
                metric=BehavioralMetric.CONFIDENCE,
                value=0.7,
                confidence=0.8
            ),
            BehavioralDataPoint(
                timestamp=datetime.now(timezone.utc),
                metric=BehavioralMetric.CONFIDENCE,
                value=0.8,
                confidence=0.8
            )
        ]
        
        trend = BehavioralTrend(
            metric=BehavioralMetric.CONFIDENCE,
            direction=TrendDirection.IMPROVING,
            change_magnitude=0.1,
            confidence=0.8,
            data_points=data_points,
            time_span=(datetime.now(timezone.utc), datetime.now(timezone.utc))
        )
        
        analysis = BehavioralAnalysis(
            user_id="test_user",
            analysis_period=(datetime.now(timezone.utc), datetime.now(timezone.utc)),
            trends=[trend],
            overall_insights=["التحسن العام في الثقة"],
            recommendations=["استمر في النهج الحالي"],
            confidence_score=0.8,
            language=Language.ARABIC
        )
        
        # Create chart
        result = await engine.create_behavioral_trends_chart(
            analysis=analysis,
            chart_type=ChartType.LINE_CHART,
            export_format=ExportFormat.JSON
        )
        
        print(f"Chart created successfully")
        print(f"Chart type: {result.chart_config.chart_type.value}")
        print(f"Export format: {result.export_format.value}")
        print(f"Number of datasets: {len(result.datasets)}")
        
        if result.insights:
            print("Chart insights:")
            for insight in result.insights:
                print(f"- {insight}")
        
        # Test ASCII chart
        ascii_result = await engine.create_behavioral_trends_chart(
            analysis=analysis,
            chart_type=ChartType.LINE_CHART,
            export_format=ExportFormat.ASCII
        )
        
        print(f"\nASCII Chart:")
        print(ascii_result.chart_data.get('ascii', 'No ASCII data'))
        
        print("\nVisualization Engine test passed!")
        return True
        
    except Exception as e:
        print(f"Visualization Engine test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("Time Machine Components Test")
    print("===========================")
    
    tests_passed = 0
    total_tests = 3
    
    if await test_query_parser():
        tests_passed += 1
    
    if await test_reflection_analyzer():
        tests_passed += 1
    
    if await test_visualization_engine():
        tests_passed += 1
    
    print(f"\n" + "="*50)
    print(f"Tests Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("All Time Machine components are working correctly!")
        return 0
    else:
        print("Some components failed tests.")
        return 1


if __name__ == "__main__":
    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)