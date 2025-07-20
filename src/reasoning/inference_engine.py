"""
Advanced Inference Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:45:18 UTC

This module provides comprehensive inference capabilities including logical reasoning,
probabilistic inference, knowledge-based deduction, causal reasoning, and temporal
reasoning with full integration into the AI assistant's core systems.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import math
import itertools
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import networkx as nx
from scipy import stats
from sympy import symbols, And, Or, Not, Implies, satisfiable
from sympy.logic.boolalg import BooleanFunction

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    InferenceStarted, InferenceCompleted, InferenceFailed, InferenceRuleApplied,
    KnowledgeUpdated, CausalInferencePerformed, UncertaintyComputed, 
    TemporalReasoningCompleted, SpatialReasoningCompleted, RuleValidated,
    KnowledgeGraphUpdated, InferenceStrategyChanged, ErrorOccurred,
    ComponentHealthChanged, PerformanceThresholdExceeded
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Assistant components
from src.assistant.component_manager import EnhancedComponentManager
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator

# Knowledge and reasoning
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.reasoning.planning.task_planner import TaskPlanner
from src.reasoning.decision_making.decision_tree import DecisionTree

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory
from src.memory.storage.vector_store import VectorStore

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class InferenceType(Enum):
    """Types of inference supported by the engine."""
    LOGICAL = "logical"                    # Formal logical reasoning
    PROBABILISTIC = "probabilistic"       # Bayesian and statistical inference
    CAUSAL = "causal"                     # Causal reasoning and intervention
    TEMPORAL = "temporal"                 # Time-based reasoning
    SPATIAL = "spatial"                   # Geometric and topological reasoning
    FUZZY = "fuzzy"                       # Fuzzy logic reasoning
    MODAL = "modal"                       # Modal logic (necessity, possibility)
    ABDUCTIVE = "abductive"               # Best explanation inference
    INDUCTIVE = "inductive"               # Generalization from examples
    DEDUCTIVE = "deductive"               # Logical deduction


class InferenceStrategy(Enum):
    """Inference execution strategies."""
    FORWARD_CHAINING = "forward_chaining"   # Data-driven reasoning
    BACKWARD_CHAINING = "backward_chaining" # Goal-driven reasoning
    MIXED_STRATEGY = "mixed_strategy"       # Hybrid approach
    BEST_FIRST = "best_first"              # Heuristic-guided search
    BREADTH_FIRST = "breadth_first"        # Breadth-first exploration
    DEPTH_FIRST = "depth_first"            # Depth-first exploration
    MONTE_CARLO = "monte_carlo"            # Probabilistic sampling
    BEAM_SEARCH = "beam_search"            # Limited beam search


class ConfidenceLevel(Enum):
    """Confidence levels for inference results."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


class LogicalOperator(Enum):
    """Logical operators for rule construction."""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    XOR = "xor"  # exclusive or
    FORALL = "forall"  # universal quantifier
    EXISTS = "exists"  # existential quantifier


@dataclass
class InferenceRule:
    """Represents an inference rule."""
    rule_id: str
    name: str
    rule_type: InferenceType
    
    # Rule structure
    premises: List[str] = field(default_factory=list)
    conclusion: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Rule metadata
    confidence: float = 1.0
    priority: int = 5  # 1-10 scale
    is_active: bool = True
    
    # Application tracking
    application_count: int = 0
    success_rate: float = 1.0
    last_applied: Optional[datetime] = None
    
    # Rule learning
    learned_from_examples: bool = False
    strength: float = 1.0  # Rule strength based on evidence
    
    # Temporal constraints
    temporal_validity: Optional[Tuple[datetime, datetime]] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    description: Optional[str] = None


@dataclass
class InferenceQuery:
    """Represents an inference query."""
    query_id: str
    query_type: InferenceType
    question: str
    
    # Query context
    premises: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Execution preferences
    strategy: InferenceStrategy = InferenceStrategy.MIXED_STRATEGY
    max_depth: int = 10
    timeout_seconds: float = 30.0
    min_confidence: float = 0.5
    
    # Session context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 5


@dataclass
class InferenceResult:
    """Comprehensive inference result."""
    query_id: str
    inference_type: InferenceType
    success: bool
    
    # Core results
    conclusion: Optional[str] = None
    confidence: float = 0.0
    certainty_level: ConfidenceLevel = ConfidenceLevel.VERY_LOW
    
    # Evidence and reasoning
    evidence: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    
    # Probabilistic results
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    uncertainty_measures: Dict[str, float] = field(default_factory=dict)
    
    # Alternative conclusions
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    
    # Causal information
    causal_chains: List[List[str]] = field(default_factory=list)
    intervention_effects: Dict[str, float] = field(default_factory=dict)
    
    # Temporal information
    temporal_constraints: List[str] = field(default_factory=list)
    temporal_ordering: List[Tuple[str, str]] = field(default_factory=list)
    
    # Performance metrics
    inference_time: float = 0.0
    rules_evaluated: int = 0
    knowledge_accessed: int = 0
    
    # Quality metrics
    consistency_score: float = 1.0
    completeness_score: float = 1.0
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_used: Optional[InferenceStrategy] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class InferenceError(Exception):
    """Custom exception for inference operations."""
    
    def __init__(self, message: str, query_id: Optional[str] = None, 
                 error_code: Optional[str] = None, inference_type: Optional[InferenceType] = None):
        super().__init__(message)
        self.query_id = query_id
        self.error_code = error_code
        self.inference_type = inference_type
        self.timestamp = datetime.now(timezone.utc)


class BaseInferenceEngine(ABC):
    """Abstract base class for specific inference engines."""
    
    @abstractmethod
    async def infer(self, query: InferenceQuery, context: Dict[str, Any]) -> InferenceResult:
        """Perform inference based on the query."""
        pass
    
    @abstractmethod
    def can_handle(self, inference_type: InferenceType) -> bool:
        """Check if this engine can handle the inference type."""
        pass
    
    @abstractmethod
    async def validate_query(self, query: InferenceQuery) -> bool:
        """Validate if the query is well-formed for this engine."""
        pass


class LogicalInferenceEngine(BaseInferenceEngine):
    """Engine for formal logical reasoning."""
    
    def __init__(self, logic_engine: LogicEngine):
        self.logic_engine = logic_engine
        self.logger = get_logger(__name__)
        self._logical_cache: Dict[str, Any] = {}
    
    def can_handle(self, inference_type: InferenceType) -> bool:
        """Check if this engine handles logical inference."""
        return inference_type in [InferenceType.LOGICAL, InferenceType.DEDUCTIVE, InferenceType.MODAL]
    
    async def validate_query(self, query: InferenceQuery) -> bool:
        """Validate logical query structure."""
        try:
            # Check if premises and conclusion are well-formed logical statements
            for premise in query.premises:
                if not self._is_valid_logical_statement(premise):
                    return False
            
            if query.question and not self._is_valid_logical_statement(query.question):
                return False
            
            return True
        except Exception:
            return False
    
    def _is_valid_logical_statement(self, statement: str) -> bool:
        """Check if a statement is a valid logical expression."""
        try:
            # This would implement proper logical statement validation
            # For now, basic validation
            forbidden_chars = ['`', ';', 'exec', 'eval']
            return not any(char in statement.lower() for char in forbidden_chars)
        except Exception:
            return False
    
    async def infer(self, query: InferenceQuery, context: Dict[str, Any]) -> InferenceResult:
        """Perform logical inference."""
        start_time = time.time()
        result = InferenceResult(
            query_id=query.query_id,
            inference_type=query.query_type,
            success=False,
            strategy_used=query.strategy
        )
        
        try:
            # Convert premises to logical forms
            logical_premises = await self._convert_to_logical_form(query.premises, context)
            
            # Apply logical inference rules
            if query.strategy == InferenceStrategy.FORWARD_CHAINING:
                conclusion = await self._forward_chain(logical_premises, context)
            elif query.strategy == InferenceStrategy.BACKWARD_CHAINING:
                conclusion = await self._backward_chain(query.question, logical_premises, context)
            else:
                # Mixed strategy - try both
                conclusion = await self._mixed_inference(query.question, logical_premises, context)
            
            if conclusion:
                result.conclusion = str(conclusion)
                result.confidence = self._calculate_logical_confidence(conclusion, logical_premises)
                result.certainty_level = self._confidence_to_level(result.confidence)
                result.success = True
                result.reasoning_chain = await self._get_reasoning_chain(conclusion, logical_premises)
            
            result.inference_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            result.inference_time = time.time() - start_time
            return result
    
    async def _convert_to_logical_form(self, premises: List[str], context: Dict[str, Any]) -> List[Any]:
        """Convert natural language premises to logical form."""
        logical_forms = []
        
        for premise in premises:
            # This would implement NL to logic conversion
            # For now, assume premises are already in logical form
            logical_forms.append(premise)
        
        return logical_forms
    
    async def _forward_chain(self, premises: List[Any], context: Dict[str, Any]) -> Optional[str]:
        """Forward chaining inference."""
        # Implementation would apply rules forward from premises
        # This is a simplified version
        for premise in premises:
            # Apply modus ponens and other inference rules
            if "implies" in str(premise).lower():
                # Extract implication and apply rule
                parts = str(premise).lower().split("implies")
                if len(parts) == 2:
                    antecedent, consequent = parts[0].strip(), parts[1].strip()
                    if antecedent in [str(p).lower() for p in premises]:
                        return consequent
        
        return None
    
    async def _backward_chain(self, goal: str, premises: List[Any], context: Dict[str, Any]) -> Optional[str]:
        """Backward chaining inference."""
        # Implementation would work backward from goal
        # This is a simplified version
        if goal.lower() in [str(p).lower() for p in premises]:
            return goal
        
        # Look for rules that conclude the goal
        for premise in premises:
            if "implies" in str(premise).lower():
                parts = str(premise).lower().split("implies")
                if len(parts) == 2 and parts[1].strip() == goal.lower():
                    antecedent = parts[0].strip()
                    # Recursively try to prove antecedent
                    if await self._backward_chain(antecedent, premises, context):
                        return goal
        
        return None
    
    async def _mixed_inference(self, goal: str, premises: List[Any], context: Dict[str, Any]) -> Optional[str]:
        """Mixed inference strategy."""
        # Try forward chaining first
        forward_result = await self._forward_chain(premises, context)
        if forward_result:
            return forward_result
        
        # Try backward chaining
        backward_result = await self._backward_chain(goal, premises, context)
        return backward_result
    
    def _calculate_logical_confidence(self, conclusion: str, premises: List[Any]) -> float:
        """Calculate confidence in logical conclusion."""
        # In classical logic, valid inferences have confidence 1.0
        # This could be extended for fuzzy logic
        return 1.0 if conclusion else 0.0
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _get_reasoning_chain(self, conclusion: str, premises: List[Any]) -> List[str]:
        """Get the chain of reasoning leading to conclusion."""
        # This would trace back the inference steps
        return [f"Applied logical inference to premises", f"Concluded: {conclusion}"]


class ProbabilisticInferenceEngine(BaseInferenceEngine):
    """Engine for probabilistic and Bayesian inference."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._bayesian_networks: Dict[str, nx.DiGraph] = {}
        self._probability_cache: Dict[str, float] = {}
    
    def can_handle(self, inference_type: InferenceType) -> bool:
        """Check if this engine handles probabilistic inference."""
        return inference_type == InferenceType.PROBABILISTIC
    
    async def validate_query(self, query: InferenceQuery) -> bool:
        """Validate probabilistic query."""
        try:
            # Check if query contains probabilistic statements
            probabilistic_keywords = ['probability', 'likely', 'chance', 'odds', 'given']
            query_text = query.question.lower()
            return any(keyword in query_text for keyword in probabilistic_keywords)
        except Exception:
            return False
    
    async def infer(self, query: InferenceQuery, context: Dict[str, Any]) -> InferenceResult:
        """Perform probabilistic inference."""
        start_time = time.time()
        result = InferenceResult(
            query_id=query.query_id,
            inference_type=query.query_type,
            success=False,
            strategy_used=query.strategy
        )
        
        try:
            # Parse probabilistic query
            target_variable, evidence = await self._parse_probabilistic_query(query, context)
            
            # Perform Bayesian inference
            if target_variable:
                probability = await self._bayesian_inference(target_variable, evidence, context)
                
                result.conclusion = f"P({target_variable}|evidence) = {probability:.4f}"
                result.confidence = probability
                result.certainty_level = self._confidence_to_level(probability)
                result.probability_distribution = {target_variable: probability}
                result.success = True
                
                # Calculate uncertainty measures
                result.uncertainty_measures = {
                    'entropy': self._calculate_entropy([probability, 1 - probability]),
                    'variance': probability * (1 - probability)
                }
            
            result.inference_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            result.inference_time = time.time() - start_time
            return result
    
    async def _parse_probabilistic_query(self, query: InferenceQuery, context: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse probabilistic query to extract target variable and evidence."""
        # This would implement sophisticated probabilistic query parsing
        # For now, simplified extraction
        question = query.question.lower()
        
        if 'probability of' in question:
            target_start = question.find('probability of') + len('probability of')
            target_end = question.find(' given', target_start)
            if target_end == -1:
                target_end = len(question)
            
            target_variable = question[target_start:target_end].strip()
            
            # Extract evidence
            evidence = {}
            if ' given ' in question:
                evidence_text = question[question.find(' given ') + len(' given '):]
                # Parse evidence (simplified)
                evidence = {'condition': evidence_text.strip()}
            
            return target_variable, evidence
        
        return None, {}
    
    async def _bayesian_inference(self, target: str, evidence: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Perform Bayesian inference."""
        try:
            # This would implement full Bayesian network inference
            # For now, simplified calculation
            
            # Get prior probability
            prior = context.get(f'prior_{target}', 0.5)
            
            # Apply Bayes' theorem with evidence
            if evidence:
                likelihood = self._calculate_likelihood(target, evidence, context)
                marginal = self._calculate_marginal_probability(evidence, context)
                
                if marginal > 0:
                    posterior = (likelihood * prior) / marginal
                    return min(max(posterior, 0.0), 1.0)  # Clamp to [0, 1]
            
            return prior
            
        except Exception as e:
            self.logger.warning(f"Bayesian inference error: {str(e)}")
            return 0.5  # Default uncertain probability
    
    def _calculate_likelihood(self, target: str, evidence: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate likelihood P(evidence|target)."""
        # Simplified likelihood calculation
        return context.get(f'likelihood_{target}', 0.8)
    
    def _calculate_marginal_probability(self, evidence: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate marginal probability P(evidence)."""
        # Simplified marginal calculation
        return context.get('marginal_probability', 0.6)
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy."""
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class CausalInferenceEngine(BaseInferenceEngine):
    """Engine for causal reasoning and intervention analysis."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.logger = get_logger(__name__)
        self._causal_graphs: Dict[str, nx.DiGraph] = {}
    
    def can_handle(self, inference_type: InferenceType) -> bool:
        """Check if this engine handles causal inference."""
        return inference_type == InferenceType.CAUSAL
    
    async def validate_query(self, query: InferenceQuery) -> bool:
        """Validate causal query."""
        try:
            causal_keywords = ['cause', 'effect', 'because', 'leads to', 'results in', 'intervention']
            query_text = query.question.lower()
            return any(keyword in query_text for keyword in causal_keywords)
        except Exception:
            return False
    
    async def infer(self, query: InferenceQuery, context: Dict[str, Any]) -> InferenceResult:
        """Perform causal inference."""
        start_time = time.time()
        result = InferenceResult(
            query_id=query.query_id,
            inference_type=query.query_type,
            success=False,
            strategy_used=query.strategy
        )
        
        try:
            # Parse causal query
            cause_var, effect_var, intervention = await self._parse_causal_query(query, context)
            
            if cause_var and effect_var:
                # Build or retrieve causal graph
                causal_graph = await self._get_causal_graph(context)
                
                # Perform causal inference
                causal_effect = await self._estimate_causal_effect(
                    cause_var, effect_var, causal_graph, intervention
                )
                
                if causal_effect is not None:
                    result.conclusion = f"Causal effect of {cause_var} on {effect_var}: {causal_effect:.4f}"
                    result.confidence = abs(causal_effect) if abs(causal_effect) <= 1 else 1.0
                    result.certainty_level = self._confidence_to_level(result.confidence)
                    result.success = True
                    
                    # Extract causal chains
                    result.causal_chains = await self._find_causal_chains(
                        cause_var, effect_var, causal_graph
                    )
                    
                    # Calculate intervention effects
                    if intervention:
                        result.intervention_effects = await self._calculate_intervention_effects(
                            intervention, causal_graph, context
                        )
            
            result.inference_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            result.inference_time = time.time() - start_time
            return result
    
    async def _parse_causal_query(self, query: InferenceQuery, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Parse causal query to extract cause, effect, and intervention."""
        question = query.question.lower()
        
        cause_var = None
        effect_var = None
        intervention = None
        
        # Extract cause and effect variables
        if 'cause' in question and 'effect' in question:
            # Extract from "what is the effect of X on Y" pattern
            words = question.split()
            if 'of' in words and 'on' in words:
                of_idx = words.index('of')
                on_idx = words.index('on')
                if of_idx < on_idx:
                    cause_var = ' '.join(words[of_idx + 1:on_idx])
                    effect_var = ' '.join(words[on_idx + 1:])
        
        # Extract intervention
        if 'intervention' in question or 'do(' in question:
            intervention = {'type': 'intervention', 'variables': [cause_var] if cause_var else []}
        
        return cause_var, effect_var, intervention
    
    async def _get_causal_graph(self, context: Dict[str, Any]) -> nx.DiGraph:
        """Get or build causal graph from knowledge."""
        # Try to get from context first
        if 'causal_graph' in context:
            return context['causal_graph']
        
        # Build from knowledge graph
        causal_graph = nx.DiGraph()
        
        # Extract causal relationships from knowledge graph
        if hasattr(self.knowledge_graph, 'get_causal_relationships'):
            causal_relationships = await self.knowledge_graph.get_causal_relationships()
            for rel in causal_relationships:
                causal_graph.add_edge(rel['cause'], rel['effect'], weight=rel.get('strength', 1.0))
        
        return causal_graph
    
    async def _estimate_causal_effect(self, cause: str, effect: str, causal_graph: nx.DiGraph, intervention: Optional[Dict[str, Any]]) -> Optional[float]:
        """Estimate causal effect using do-calculus or other methods."""
        try:
            if causal_graph.has_edge(cause, effect):
                # Direct causal relationship
                edge_data = causal_graph.get_edge_data(cause, effect)
                return edge_data.get('weight', 1.0)
            
            # Check for indirect causal paths
            if nx.has_path(causal_graph, cause, effect):
                # Calculate effect through causal paths
                paths = list(nx.all_simple_paths(causal_graph, cause, effect))
                if paths:
                    # Take the strongest path
                    max_effect = 0
                    for path in paths:
                        path_effect = self._calculate_path_effect(path, causal_graph)
                        max_effect = max(max_effect, abs(path_effect))
                    return max_effect
            
            return 0.0  # No causal relationship found
            
        except Exception as e:
            self.logger.warning(f"Causal effect estimation error: {str(e)}")
            return None
    
    def _calculate_path_effect(self, path: List[str], causal_graph: nx.DiGraph) -> float:
        """Calculate the effect along a causal path."""
        effect = 1.0
        for i in range(len(path) - 1):
            edge_data = causal_graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                effect *= edge_data.get('weight', 1.0)
        return effect
    
    async def _find_causal_chains(self, cause: str, effect: str, causal_graph: nx.DiGraph) -> List[List[str]]:
        """Find all causal chains between cause and effect."""
        try:
            if nx.has_path(causal_graph, cause, effect):
                paths = list(nx.all_simple_paths(causal_graph, cause, effect, cutoff=5))
                return paths[:10]  # Limit to top 10 paths
            return []
        except Exception:
            return []
    
    async def _calculate_intervention_effects(self, intervention: Dict[str, Any], causal_graph: nx.DiGraph, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effects of interventions."""
        effects = {}
        
        intervention_vars = intervention.get('variables', [])
        for var in intervention_vars:
            if var in causal_graph.nodes:
                # Calculate effect on all downstream variables
                descendants = list(nx.descendants(causal_graph, var))
                for desc in descendants:
                    effects[desc] = await self._estimate_causal_effect(var, desc, causal_graph, intervention)
        
        return effects
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class TemporalInferenceEngine(BaseInferenceEngine):
    """Engine for temporal reasoning and time-based inference."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._temporal_relations: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    
    def can_handle(self, inference_type: InferenceType) -> bool:
        """Check if this engine handles temporal inference."""
        return inference_type == InferenceType.TEMPORAL
    
    async def validate_query(self, query: InferenceQuery) -> bool:
        """Validate temporal query."""
        try:
            temporal_keywords = ['before', 'after', 'during', 'when', 'while', 'then', 'sequence']
            query_text = query.question.lower()
            return any(keyword in query_text for keyword in temporal_keywords)
        except Exception:
            return False
    
    async def infer(self, query: InferenceQuery, context: Dict[str, Any]) -> InferenceResult:
        """Perform temporal inference."""
        start_time = time.time()
        result = InferenceResult(
            query_id=query.query_id,
            inference_type=query.query_type,
            success=False,
            strategy_used=query.strategy
        )
        
        try:
            # Parse temporal query
            events, temporal_constraints = await self._parse_temporal_query(query, context)
            
            if events:
                # Perform temporal reasoning
                temporal_ordering = await self._infer_temporal_ordering(events, temporal_constraints, context)
                
                if temporal_ordering:
                    result.conclusion = f"Temporal ordering inferred for {len(events)} events"
                    result.confidence = 0.8  # Default confidence for temporal reasoning
                    result.certainty_level = self._confidence_to_level(result.confidence)
                    result.temporal_ordering = temporal_ordering
                    result.temporal_constraints = [str(c) for c in temporal_constraints]
                    result.success = True
                    
                    # Generate reasoning chain
                    result.reasoning_chain = await self._generate_temporal_reasoning_chain(
                        events, temporal_ordering, temporal_constraints
                    )
            
            result.inference_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            result.inference_time = time.time() - start_time
            return result
    
    async def _parse_temporal_query(self, query: InferenceQuery, context: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Parse temporal query to extract events and constraints."""
        events = []
        constraints = []
        
        # Extract events from premises and question
        for premise in query.premises:
            events.extend(self._extract_events_from_text(premise))
        
        events.extend(self._extract_events_from_text(query.question))
        
        # Extract temporal constraints
        all_text = ' '.join(query.premises + [query.question])
        constraints = self._extract_temporal_constraints(all_text, events)
        
        return list(set(events)), constraints
    
    def _extract_events_from_text(self, text: str) -> List[str]:
        """Extract events from text."""
        # Simplified event extraction
        words = text.lower().split()
        events = []
        
        # Look for verb phrases that indicate events
        for i, word in enumerate(words):
            if word in ['happened', 'occurred', 'started', 'ended', 'began', 'finished']:
                if i > 0:
                    events.append(f"{words[i-1]}_{word}")
        
        return events
    
    def _extract_temporal_constraints(self, text: str, events: List[str]) -> List[Tuple[str, str, str]]:
        """Extract temporal constraints between events."""
        constraints = []
        text_lower = text.lower()
        
        # Look for temporal relationship indicators
        if 'before' in text_lower:
            # Extract "A before B" patterns
            parts = text_lower.split(' before ')
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    event_a = self._find_nearest_event(parts[i], events, 'right')
                    event_b = self._find_nearest_event(parts[i + 1], events, 'left')
                    if event_a and event_b:
                        constraints.append((event_a, 'before', event_b))
        
        if 'after' in text_lower:
            # Extract "A after B" patterns
            parts = text_lower.split(' after ')
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    event_a = self._find_nearest_event(parts[i], events, 'right')
                    event_b = self._find_nearest_event(parts[i + 1], events, 'left')
                    if event_a and event_b:
                        constraints.append((event_a, 'after', event_b))
        
        return constraints
    
    def _find_nearest_event(self, text: str, events: List[str], direction: str) -> Optional[str]:
        """Find the nearest event in the given direction."""
        words = text.strip().split()
        
        if direction == 'right':
            # Find event at the end of text
            for event in events:
                if any(word in event.lower() for word in words[-3:]):
                    return event
        else:
            # Find event at the beginning of text
            for event in events:
                if any(word in event.lower() for word in words[:3]):
                    return event
        
        return None
    
    async def _infer_temporal_ordering(self, events: List[str], constraints: List[Tuple[str, str, str]], context: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Infer temporal ordering of events."""
        # Build temporal constraint graph
        temporal_graph = nx.DiGraph()
        temporal_graph.add_nodes_from(events)
        
        for constraint in constraints:
            event_a, relation, event_b = constraint
            if relation == 'before':
                temporal_graph.add_edge(event_a, event_b)
            elif relation == 'after':
                temporal_graph.add_edge(event_b, event_a)
        
        # Check for cycles (temporal contradictions)
        if not nx.is_directed_acyclic_graph(temporal_graph):
            self.logger.warning("Temporal contradictions detected")
            return []
        
        # Generate topological ordering
        try:
            ordering = list(nx.topological_sort(temporal_graph))
            pairs = [(ordering[i], ordering[i + 1]) for i in range(len(ordering) - 1)]
            return pairs
        except Exception:
            return []
    
    async def _generate_temporal_reasoning_chain(self, events: List[str], ordering: List[Tuple[str, str]], constraints: List[Tuple[str, str, str]]) -> List[str]:
        """Generate reasoning chain for temporal inference."""
        chain = []
        chain.append(f"Identified {len(events)} events for temporal analysis")
        chain.append(f"Applied {len(constraints)} temporal constraints")
        
        if ordering:
            chain.append(f"Derived temporal ordering with {len(ordering)} relationships")
            for event_a, event_b in ordering[:5]:  # Show first 5
                chain.append(f"  {event_a} occurs before {event_b}")
        
        return chain
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class SpatialInferenceEngine(BaseInferenceEngine):
    """Engine for spatial and geometric reasoning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._spatial_relations: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    
    def can_handle(self, inference_type: InferenceType) -> bool:
        """Check if this engine handles spatial inference."""
        return inference_type == InferenceType.SPATIAL
    
    async def validate_query(self, query: InferenceQuery) -> bool:
        """Validate spatial query."""
        try:
            spatial_keywords = ['near', 'far', 'above', 'below', 'left', 'right', 'inside', 'outside', 'between']
            query_text = query.question.lower()
            return any(keyword in query_text for keyword in spatial_keywords)
        except Exception:
            return False
    
    async def infer(self, query: InferenceQuery, context: Dict[str, Any]) -> InferenceResult:
        """Perform spatial inference."""
        start_time = time.time()
        result = InferenceResult(
            query_id=query.query_id,
            inference_type=query.query_type,
            success=False,
            strategy_used=query.strategy
        )
        
        try:
            # Parse spatial query
            objects, spatial_constraints = await self._parse_spatial_query(query, context)
            
            if objects and spatial_constraints:
                # Perform spatial reasoning
                spatial_conclusions = await self._infer_spatial_relationships(
                    objects, spatial_constraints, context
                )
                
                if spatial_conclusions:
                    result.conclusion = f"Inferred {len(spatial_conclusions)} spatial relationships"
                    result.confidence = 0.7  # Default confidence for spatial reasoning
                    result.certainty_level = self._confidence_to_level(result.confidence)
                    result.success = True
                    
                    # Add spatial conclusions to evidence
                    result.evidence = [str(conclusion) for conclusion in spatial_conclusions]
                    
                    # Generate reasoning chain
                    result.reasoning_chain = await self._generate_spatial_reasoning_chain(
                        objects, spatial_constraints, spatial_conclusions
                    )
            
            result.inference_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            result.inference_time = time.time() - start_time
            return result
    
    async def _parse_spatial_query(self, query: InferenceQuery, context: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Parse spatial query to extract objects and spatial constraints."""
        objects = []
        constraints = []
        
        # Extract objects and spatial relationships
        all_text = ' '.join(query.premises + [query.question])
        objects = self._extract_spatial_objects(all_text)
        constraints = self._extract_spatial_constraints(all_text, objects)
        
        return objects, constraints
    
    def _extract_spatial_objects(self, text: str) -> List[str]:
        """Extract spatial objects from text."""
        # Simplified object extraction
        words = text.lower().split()
        objects = []
        
        # Look for nouns that could be spatial objects
        potential_objects = ['table', 'chair', 'book', 'cup', 'room', 'wall', 'door', 'window']
        for word in words:
            if word in potential_objects:
                objects.append(word)
        
        return list(set(objects))
    
    def _extract_spatial_constraints(self, text: str, objects: List[str]) -> List[Tuple[str, str, str]]:
        """Extract spatial constraints between objects."""
        constraints = []
        text_lower = text.lower()
        
        spatial_relations = {
            'near': 'near',
            'close to': 'near',
            'above': 'above',
            'over': 'above',
            'below': 'below',
            'under': 'below',
            'left of': 'left',
            'right of': 'right',
            'inside': 'inside',
            'in': 'inside',
            'outside': 'outside',
            'between': 'between'
        }
        
        for relation_phrase, relation_type in spatial_relations.items():
            if relation_phrase in text_lower:
                # Extract objects around the relation
                parts = text_lower.split(relation_phrase)
                if len(parts) >= 2:
                    obj_a = self._find_nearest_object(parts[0], objects, 'right')
                    obj_b = self._find_nearest_object(parts[1], objects, 'left')
                    if obj_a and obj_b:
                        constraints.append((obj_a, relation_type, obj_b))
        
        return constraints
    
    def _find_nearest_object(self, text: str, objects: List[str], direction: str) -> Optional[str]:
        """Find the nearest object in the given direction."""
        words = text.strip().split()
        
        if direction == 'right':
            # Find object at the end of text
            for obj in objects:
                if obj in words[-3:]:
                    return obj
        else:
            # Find object at the beginning of text
            for obj in objects:
                if obj in words[:3]:
                    return obj
        
        return None
    
    async def _infer_spatial_relationships(self, objects: List[str], constraints: List[Tuple[str, str, str]], context: Dict[str, Any]) -> List[str]:
        """Infer additional spatial relationships from given constraints."""
        conclusions = []
        
        # Apply spatial reasoning rules
        for i, constraint1 in enumerate(constraints):
            for j, constraint2 in enumerate(constraints):
                if i != j:
                    conclusion = self._apply_spatial_transitivity(constraint1, constraint2)
                    if conclusion and conclusion not in conclusions:
                        conclusions.append(conclusion)
        
        return conclusions
    
    def _apply_spatial_transitivity(self, constraint1: Tuple[str, str, str], constraint2: Tuple[str, str, str]) -> Optional[str]:
        """Apply spatial transitivity rules."""
        obj1_a, rel1, obj1_b = constraint1
        obj2_a, rel2, obj2_b = constraint2
        
        # If A is above B and B is above C, then A is above C
        if obj1_b == obj2_a and rel1 == 'above' and rel2 == 'above':
            return f"{obj1_a} is above {obj2_b}"
        
        # If A is left of B and B is left of C, then A is left of C
        if obj1_b == obj2_a and rel1 == 'left' and rel2 == 'left':
            return f"{obj1_a} is left of {obj2_b}"
        
        # If A is inside B and B is inside C, then A is inside C
        if obj1_b == obj2_a and rel1 == 'inside' and rel2 == 'inside':
            return f"{obj1_a} is inside {obj2_b}"
        
        return None
    
    async def _generate_spatial_reasoning_chain(self, objects: List[str], constraints: List[Tuple[str, str, str]], conclusions: List[str]) -> List[str]:
        """Generate reasoning chain for spatial inference."""
        chain = []
        chain.append(f"Identified {len(objects)} spatial objects")
        chain.append(f"Applied {len(constraints)} spatial constraints")
        
        if conclusions:
            chain.append(f"Derived {len(conclusions)} additional spatial relationships")
            for conclusion in conclusions[:3]:  # Show first 3
                chain.append(f"  Inferred: {conclusion}")
        
        return chain
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class RuleManager:
    """Manages inference rules including validation, optimization, and learning."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, feedback_processor: FeedbackProcessor):
        self.knowledge_graph = knowledge_graph
        self.feedback_processor = feedback_processor
        self.logger = get_logger(__name__)
        self._rules: Dict[str, InferenceRule] = {}
        self._rule_index: Dict[InferenceType, List[str]] = defaultdict(list)
        self._rule_performance: Dict[str, List[float]] = defaultdict(list)
    
    def register_rule(self, rule: InferenceRule) -> None:
        """Register a new inference rule."""
        self._rules[rule.rule_id] = rule
        self._rule_index[rule.rule_type].append(rule.rule_id)
        
        self.logger.info(f"Registered inference rule: {rule.rule_id} ({rule.rule_type.value})")
    
    def get_rules_for_type(self, inference_type: InferenceType) -> List[InferenceRule]:
        """Get all active rules for a specific inference type."""
        rule_ids = self._rule_index.get(inference_type, [])
        return [self._rules[rule_id] for rule_id in rule_ids if self._rules[rule_id].is_active]
    
    async def validate_rule(self, rule: InferenceRule) -> bool:
        """Validate a rule for correctness and consistency."""
        try:
            # Check rule structure
            if not rule.premises or not rule.conclusion:
                return False
            
            # Check for logical consistency
            if rule.rule_type == InferenceType.LOGICAL:
                return await self._validate_logical_rule(rule)
            elif rule.rule_type == InferenceType.PROBABILISTIC:
                return await self._validate_probabilistic_rule(rule)
            elif rule.rule_type == InferenceType.CAUSAL:
                return await self._validate_causal_rule(rule)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rule validation error for {rule.rule_id}: {str(e)}")
            return False
    
    async def _validate_logical_rule(self, rule: InferenceRule) -> bool:
        """Validate logical rule structure."""
        # Check if premises and conclusion are well-formed
        try:
            # This would implement formal logical validation
            return len(rule.premises) > 0 and rule.conclusion
        except Exception:
            return False
    
    async def _validate_probabilistic_rule(self, rule: InferenceRule) -> bool:
        """Validate probabilistic rule."""
        # Check probability constraints
        return 0 <= rule.confidence <= 1
    
    async def _validate_causal_rule(self, rule: InferenceRule) -> bool:
        """Validate causal rule."""
        # Check causal structure
        return 'cause' in rule.conclusion.lower() or 'effect' in rule.conclusion.lower()
    
    def update_rule_performance(self, rule_id: str, success: bool, execution_time: float) -> None:
        """Update rule performance metrics."""
        if rule_id in self._rules:
            rule = self._rules[rule_id]
            rule.application_count += 1
            
            # Update success rate
            if success:
                rule.success_rate = (rule.success_rate * (rule.application_count - 1) + 1) / rule.application_count
            else:
                rule.success_rate = (rule.success_rate * (rule.application_count - 1)) / rule.application_count
            
            rule.last_applied = datetime.now(timezone.utc)
            
            # Track performance
            self._rule_performance[rule_id].append(execution_time)
    
    async def optimize_rules(self) -> None:
        """Optimize rules based on performance data."""
        for rule_id, rule in self._rules.items():
            if rule.application_count >= 10:  # Need enough data
                # Adjust rule priority based on success rate
                if rule.success_rate > 0.8:
                    rule.priority = min(rule.priority + 1, 10)
                elif rule.success_rate < 0.4:
                    rule.priority = max(rule.priority - 1, 1)
                
                # Adjust rule strength
                performance_times = self._rule_performance[rule_id]
                if performance_times:
                    avg_time = sum(performance_times) / len(performance_times)
                    if avg_time < 1.0:  # Fast execution
                        rule.strength = min(rule.strength + 0.1, 2.0)
    
    async def learn_new_rules(self, execution_examples: List[Dict[str, Any]]) -> List[InferenceRule]:
        """Learn new rules from successful inference examples."""
        learned_rules = []
        
        # Group examples by inference type
        examples_by_type = defaultdict(list)
        for example in execution_examples:
            if 'inference_type' in example:
                examples_by_type[example['inference_type']].append(example)
        
        # Learn rules for each type
        for inference_type, examples in examples_by_type.items():
            if len(examples) >= 3:  # Need minimum examples
                new_rule = await self._extract_rule_pattern(inference_type, examples)
                if new_rule:
                    learned_rules.append(new_rule)
        
        return learned_rules
    
    async def _extract_rule_pattern(self, inference_type: str, examples: List[Dict[str, Any]]) -> Optional[InferenceRule]:
        """Extract a rule pattern from examples."""
        try:
            # Find common patterns in successful examples
            common_premises = self._find_common_premises(examples)
            common_conclusion_pattern = self._find_common_conclusion_pattern(examples)
            
            if common_premises and common_conclusion_pattern:
                rule = InferenceRule(
                    rule_id=f"learned_{inference_type}_{int(time.time())}",
                    name=f"Learned {inference_type} rule",
                    rule_type=InferenceType(inference_type),
                    premises=common_premises,
                    conclusion=common_conclusion_pattern,
                    confidence=self._calculate_learned_confidence(examples),
                    learned_from_examples=True,
                    created_by="rule_learning_system"
                )
                
                return rule
            
        except Exception as e:
            self.logger.error(f"Rule learning error: {str(e)}")
        
        return None
    
    def _find_common_premises(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Find common premises across examples."""
        all_premises = []
        for example in examples:
            premises = example.get('premises', [])
            all_premises.extend(premises)
        
        # Find premises that appear in most examples
        premise_counts = defaultdict(int)
        for premise in all_premises:
            premise_counts[premise] += 1
        
        # Return premises that appear in at least half the examples
        threshold = len(examples) // 2
        common_premises = [premise for premise, count in premise_counts.items() if count >= threshold]
        
        return common_premises[:5]  # Limit to top 5
    
    def _find_common_conclusion_pattern(self, examples: List[Dict[str, Any]]) -> str:
        """Find common conclusion pattern across examples."""
        conclusions = [example.get('conclusion', '') for example in examples]
        
        # Simple pattern matching - find most common conclusion template
        conclusion_counts = defaultdict(int)
        for conclusion in conclusions:
            # Extract pattern (simplified)
            pattern = self._extract_conclusion_pattern(conclusion)
            conclusion_counts[pattern] += 1
        
        if conclusion_counts:
            most_common = max(conclusion_counts, key=conclusion_counts.get)
            return most_common
        
        return ""
    
    def _extract_conclusion_pattern(self, conclusion: str) -> str:
        """Extract a pattern from a conclusion."""
        # Simplified pattern extraction
        words = conclusion.lower().split()
        
        # Look for common conclusion patterns
        if 'therefore' in words:
            return "therefore [conclusion]"
        elif 'implies' in words:
            return "[antecedent] implies [consequent]"
        elif 'probability' in words:
            return "probability of [event] is [value]"
        else:
            return conclusion  # Return as-is if no pattern found
    
    def _calculate_learned_confidence(self, examples: List[Dict[str, Any]]) -> float:
        """Calculate confidence for a learned rule."""
        confidences = [example.get('confidence', 0.5) for example in examples]
        return sum(confidences) / len(confidences) if confidences else 0.5


class EnhancedInferenceEngine:
    """
    Advanced Inference Engine for the AI Assistant.
    
    This engine provides comprehensive reasoning capabilities including:
    - Logical reasoning (deductive, inductive, abductive)
    - Probabilistic inference (Bayesian, statistical)
    - Causal reasoning (interventions, counterfactuals)
    - Temporal reasoning (time-based logic)
    - Spatial reasoning (geometric relationships)
    - Fuzzy logic and modal logic
    - Rule-based expert systems
    - Machine learning integration for rule discovery
    - Performance optimization and caching
    - Integration with knowledge graphs and memory systems
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced inference engine.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Knowledge and reasoning components
        self.logic_engine = container.get(LogicEngine)
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.task_planner = container.get(TaskPlanner)
        self.decision_tree = container.get(DecisionTree)
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        self.vector_store = container.get(VectorStore)
        
        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Inference engines for different types
        self._setup_inference_engines()
        
        # Rule management
        self.rule_manager = RuleManager(self.knowledge_graph, self.feedback_processor)
        
        # State management
        self.active_queries: Dict[str, InferenceQuery] = {}
        self.query_history: deque = deque(maxlen=1000)
        self.inference_cache: Dict[str, InferenceResult] = {}
        
        # Performance tracking
        self.inference_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.engine_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.max_concurrent_inferences = self.config.get("inference.max_concurrent", 10)
        self.default_timeout = self.config.get("inference.default_timeout", 30.0)
        self.enable_caching = self.config.get("inference.enable_caching", True)
        self.enable_learning = self.config.get("inference.enable_learning", True)
        self.cache_ttl = self.config.get("inference.cache_ttl", 3600)
        
        # Concurrency control
        self.inference_semaphore = asyncio.Semaphore(self.max_concurrent_inferences)
        
        # Setup monitoring and health checks
        self._setup_monitoring()
        self.health_check.register_component("inference_engine", self._health_check_callback)
        
        self.logger.info("EnhancedInferenceEngine initialized successfully")

    def _setup_inference_engines(self) -> None:
        """Initialize all inference engines."""
        try:
            # Initialize logical reasoning engine
            self.logical_engine = LogicalReasoningEngine(
                config=self.config.get('logical_reasoning', {}),
                logger=self.logger
            )
            
            # Initialize probabilistic inference engine  
            self.probabilistic_engine = ProbabilisticInferenceEngine(
                config=self.config.get('probabilistic_inference', {}),
                logger=self.logger
            )
            
            # Initialize temporal reasoning engine
            self.temporal_engine = TemporalReasoningEngine(
                config=self.config.get('temporal_reasoning', {}),
                logger=self.logger
            )
            
            # Initialize causal reasoning engine
            self.causal_engine = CausalReasoningEngine(
                config=self.config.get('causal_reasoning', {}),
                logger=self.logger
            )
            
            self.logger.info("All inference engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup inference engines: {e}")
            raise
