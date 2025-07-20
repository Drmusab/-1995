"""
Advanced Logic Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-06-12 13:00:00 UTC

This module provides comprehensive logical reasoning capabilities for the AI assistant,
including propositional logic, predicate logic, rule-based reasoning, constraint satisfaction,
temporal logic, fuzzy logic, and automated theorem proving with seamless core system integration.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import re
import json
import hashlib
import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import logging
import inspect
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import weakref

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    LogicReasoningStarted, LogicReasoningCompleted, LogicReasoningFailed,
    FactAdded, RuleAdded, RuleTriggered, ConflictDetected, ConflictResolved,
    ProofGenerated, ExplanationGenerated, KnowledgeBaseUpdated,
    InferencePerformed, QueryProcessed, RuleLearned, LogicEngineHealthChanged,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Memory integration
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import SemanticMemory, WorkingMemory

# Learning integration
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor

# Processing integration
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.entity_extractor import EntityExtractor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class LogicType(Enum):
    """Types of logical reasoning supported."""
    PROPOSITIONAL = "propositional"
    PREDICATE = "predicate"
    TEMPORAL = "temporal"
    FUZZY = "fuzzy"
    MODAL = "modal"
    CONSTRAINT = "constraint"
    RULE_BASED = "rule_based"
    CAUSAL = "causal"


class ReasoningMode(Enum):
    """Reasoning execution modes."""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    MIXED_CHAINING = "mixed_chaining"
    RESOLUTION = "resolution"
    NATURAL_DEDUCTION = "natural_deduction"
    TABLEAU = "tableau"
    SAT_SOLVING = "sat_solving"


class TruthValue(Enum):
    """Truth values for logical statements."""
    TRUE = 1.0
    FALSE = 0.0
    UNKNOWN = 0.5
    INCONSISTENT = -1.0


class FactType(Enum):
    """Types of logical facts."""
    ATOMIC = "atomic"
    COMPOUND = "compound"
    RULE = "rule"
    CONSTRAINT = "constraint"
    TEMPORAL = "temporal"
    FUZZY = "fuzzy"
    MODAL = "modal"


class RuleType(Enum):
    """Types of logical rules."""
    IMPLICATION = "implication"
    EQUIVALENCE = "equivalence"
    UNIVERSAL = "universal"
    EXISTENTIAL = "existential"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"


@dataclass
class LogicalOperator:
    """Represents a logical operator."""
    symbol: str
    name: str
    arity: int  # Number of operands
    precedence: int
    associativity: str = "left"  # left, right, none
    truth_function: Optional[Callable] = None
    
    def evaluate(self, *operands: TruthValue) -> TruthValue:
        """Evaluate the operator with given operands."""
        if self.truth_function:
            return self.truth_function(*operands)
        return TruthValue.UNKNOWN


@dataclass
class Term:
    """Represents a logical term."""
    name: str
    term_type: str = "constant"  # constant, variable, function
    arguments: List['Term'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.arguments:
            args_str = ", ".join(str(arg) for arg in self.arguments)
            return f"{self.name}({args_str})"
        return self.name
    
    def is_variable(self) -> bool:
        """Check if term is a variable."""
        return self.term_type == "variable"
    
    def is_constant(self) -> bool:
        """Check if term is a constant."""
        return self.term_type == "constant"
    
    def is_function(self) -> bool:
        """Check if term is a function."""
        return self.term_type == "function"
    
    def get_variables(self) -> Set[str]:
        """Get all variables in the term."""
        variables = set()
        if self.is_variable():
            variables.add(self.name)
        for arg in self.arguments:
            variables.update(arg.get_variables())
        return variables


@dataclass
class Predicate:
    """Represents a logical predicate."""
    name: str
    arguments: List[Term] = field(default_factory=list)
    negated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        predicate_str = f"{self.name}({args_str})"
        return f"¬{predicate_str}" if self.negated else predicate_str
    
    def negate(self) -> 'Predicate':
        """Return the negation of this predicate."""
        return Predicate(
            name=self.name,
            arguments=self.arguments.copy(),
            negated=not self.negated,
            metadata=self.metadata.copy()
        )
    
    def get_variables(self) -> Set[str]:
        """Get all variables in the predicate."""
        variables = set()
        for arg in self.arguments:
            variables.update(arg.get_variables())
        return variables


@dataclass
class LogicalFormula:
    """Represents a logical formula."""
    formula_id: str
    content: Union[Predicate, str]
    logic_type: LogicType = LogicType.PROPOSITIONAL
    truth_value: TruthValue = TruthValue.UNKNOWN
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return str(self.content)
    
    def is_ground(self) -> bool:
        """Check if formula is ground (no variables)."""
        if isinstance(self.content, Predicate):
            return len(self.content.get_variables()) == 0
        return True  # String formulas are considered ground


@dataclass
class LogicalRule:
    """Represents a logical rule."""
    rule_id: str
    premises: List[LogicalFormula]
    conclusion: LogicalFormula
    rule_type: RuleType = RuleType.IMPLICATION
    strength: float = 1.0
    priority: int = 0
    enabled: bool = True
    learned: bool = False
    usage_count: int = 0
    success_rate: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        premises_str = " ∧ ".join(str(p) for p in self.premises)
        if self.rule_type == RuleType.IMPLICATION:
            return f"{premises_str} → {self.conclusion}"
        elif self.rule_type == RuleType.EQUIVALENCE:
            return f"{premises_str} ↔ {self.conclusion}"
        else:
            return f"{premises_str} |= {self.conclusion}"
    
    def get_variables(self) -> Set[str]:
        """Get all variables in the rule."""
        variables = set()
        for premise in self.premises:
            if isinstance(premise.content, Predicate):
                variables.update(premise.content.get_variables())
        if isinstance(self.conclusion.content, Predicate):
            variables.update(self.conclusion.content.get_variables())
        return variables


@dataclass
class Substitution:
    """Represents a variable substitution."""
    bindings: Dict[str, Term] = field(default_factory=dict)
    
    def apply_to_term(self, term: Term) -> Term:
        """Apply substitution to a term."""
        if term.is_variable() and term.name in self.bindings:
            return self.bindings[term.name]
        
        if term.arguments:
            new_args = [self.apply_to_term(arg) for arg in term.arguments]
            return Term(term.name, term.term_type, new_args, term.metadata.copy())
        
        return term
    
    def apply_to_predicate(self, predicate: Predicate) -> Predicate:
        """Apply substitution to a predicate."""
        new_args = [self.apply_to_term(arg) for arg in predicate.arguments]
        return Predicate(
            name=predicate.name,
            arguments=new_args,
            negated=predicate.negated,
            metadata=predicate.metadata.copy()
        )
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose this substitution with another."""
        result = Substitution()
        
        # Apply other to our bindings
        for var, term in self.bindings.items():
            result.bindings[var] = other.apply_to_term(term)
        
        # Add bindings from other that are not in self
        for var, term in other.bindings.items():
            if var not in result.bindings:
                result.bindings[var] = term
        
        return result


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    success: bool
    conclusions: List[LogicalFormula] = field(default_factory=list)
    proofs: List['Proof'] = field(default_factory=list)
    substitutions: List[Substitution] = field(default_factory=list)
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    confidence: float = 0.0
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_conclusion(self, formula: LogicalFormula, proof: Optional['Proof'] = None) -> None:
        """Add a conclusion with optional proof."""
        self.conclusions.append(formula)
        if proof:
            self.proofs.append(proof)


@dataclass
class Proof:
    """Represents a logical proof."""
    proof_id: str
    conclusion: LogicalFormula
    steps: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "resolution"
    valid: bool = True
    completeness: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_step(self, step_type: str, description: str, formulas: List[LogicalFormula] = None) -> None:
        """Add a proof step."""
        step = {
            'step_number': len(self.steps) + 1,
            'step_type': step_type,
            'description': description,
            'formulas': formulas or [],
            'timestamp': datetime.now(timezone.utc)
        }
        self.steps.append(step)


@dataclass
class Query:
    """Represents a logical query."""
    query_id: str
    content: Union[LogicalFormula, str]
    query_type: str = "existential"  # existential, universal, boolean
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    max_results: int = 100
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LogicError(Exception):
    """Custom exception for logic engine operations."""
    
    def __init__(self, message: str, logic_type: Optional[LogicType] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.logic_type = logic_type
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class UnificationEngine:
    """Handles unification of terms and predicates."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def unify_terms(self, term1: Term, term2: Term, substitution: Optional[Substitution] = None) -> Optional[Substitution]:
        """Unify two terms."""
        if substitution is None:
            substitution = Substitution()
        
        # Apply current substitution
        term1 = substitution.apply_to_term(term1)
        term2 = substitution.apply_to_term(term2)
        
        # Same term
        if term1.name == term2.name and term1.term_type == term2.term_type and len(term1.arguments) == len(term2.arguments):
            if not term1.arguments:  # Constants
                return substitution
            
            # Unify arguments
            for arg1, arg2 in zip(term1.arguments, term2.arguments):
                substitution = self.unify_terms(arg1, arg2, substitution)
                if substitution is None:
                    return None
            return substitution
        
        # Variable unification
        if term1.is_variable():
            return self._unify_variable(term1.name, term2, substitution)
        elif term2.is_variable():
            return self._unify_variable(term2.name, term1, substitution)
        
        # Different terms
        return None
    
    def _unify_variable(self, var: str, term: Term, substitution: Substitution) -> Optional[Substitution]:
        """Unify a variable with a term."""
        if var in substitution.bindings:
            return self.unify_terms(substitution.bindings[var], term, substitution)
        
        if term.is_variable() and term.name in substitution.bindings:
            return self.unify_terms(Term(var, "variable"), substitution.bindings[term.name], substitution)
        
        # Occurs check
        if self._occurs_check(var, term):
            return None
        
        # Add binding
        new_substitution = Substitution(substitution.bindings.copy())
        new_substitution.bindings[var] = term
        return new_substitution
    
    def _occurs_check(self, var: str, term: Term) -> bool:
        """Check if variable occurs in term (prevents infinite structures)."""
        if term.is_variable() and term.name == var:
            return True
        
        for arg in term.arguments:
            if self._occurs_check(var, arg):
                return True
        
        return False
    
    def unify_predicates(self, pred1: Predicate, pred2: Predicate) -> Optional[Substitution]:
        """Unify two predicates."""
        if pred1.name != pred2.name or len(pred1.arguments) != len(pred2.arguments):
            return None
        
        if pred1.negated != pred2.negated:
            return None
        
        substitution = Substitution()
        for arg1, arg2 in zip(pred1.arguments, pred2.arguments):
            substitution = self.unify_terms(arg1, arg2, substitution)
            if substitution is None:
                return None
        
        return substitution


class ResolutionEngine:
    """Implements resolution theorem proving."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.unification_engine = UnificationEngine()
    
    def resolve(self, clause1: List[Predicate], clause2: List[Predicate]) -> List[List[Predicate]]:
        """Resolve two clauses."""
        resolvents = []
        
        for i, literal1 in enumerate(clause1):
            for j, literal2 in enumerate(clause2):
                # Try to resolve complementary literals
                if literal1.name == literal2.name and literal1.negated != literal2.negated:
                    substitution = self.unification_engine.unify_predicates(
                        literal1.negate() if literal1.negated else literal1,
                        literal2.negate() if literal2.negated else literal2
                    )
                    
                    if substitution is not None:
                        # Create resolvent
                        resolvent = []
                        
                        # Add remaining literals from clause1
                        for k, lit in enumerate(clause1):
                            if k != i:
                                new_lit = substitution.apply_to_predicate(lit)
                                resolvent.append(new_lit)
                        
                        # Add remaining literals from clause2
                        for k, lit in enumerate(clause2):
                            if k != j:
                                new_lit = substitution.apply_to_predicate(lit)
                                resolvent.append(new_lit)
                        
                        resolvents.append(resolvent)
        
        return resolvents
    
    def prove_by_resolution(self, premises: List[LogicalFormula], goal: LogicalFormula) -> Optional[Proof]:
        """Prove goal using resolution."""
        proof = Proof(
            proof_id=str(uuid.uuid4()),
            conclusion=goal,
            method="resolution"
        )
        
        # Convert to clause form
        clauses = []
        for premise in premises:
            if isinstance(premise.content, Predicate):
                clauses.append([premise.content])
        
        # Add negated goal
        if isinstance(goal.content, Predicate):
            negated_goal = goal.content.negate()
            clauses.append([negated_goal])
        
        proof.add_step("conversion", "Converted premises and negated goal to clause form", premises + [goal])
        
        # Resolution loop
        new_clauses = set()
        iteration = 0
        max_iterations = 1000
        
        while iteration < max_iterations:
            iteration += 1
            pairs_resolved = False
            
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    resolvents = self.resolve(clauses[i], clauses[j])
                    
                    for resolvent in resolvents:
                        if not resolvent:  # Empty clause found
                            proof.add_step("resolution", f"Derived empty clause from clauses {i} and {j}")
                            proof.add_step("conclusion", "Proof by contradiction complete")
                            return proof
                        
                        resolvent_tuple = tuple(str(lit) for lit in resolvent)
                        if resolvent_tuple not in new_clauses:
                            new_clauses.add(resolvent_tuple)
                            clauses.append(resolvent)
                            pairs_resolved = True
                            
                            proof.add_step("resolution", 
                                         f"Resolved clauses {i} and {j} to get {resolvent}")
            
            if not pairs_resolved:
                break
        
        proof.valid = False
        proof.add_step("failure", "Could not derive empty clause")
        return proof


class ForwardChainEngine:
    """Implements forward chaining inference."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.unification_engine = UnificationEngine()
    
    def forward_chain(self, facts: List[LogicalFormula], rules: List[LogicalRule], 
                     max_iterations: int = 100) -> ReasoningResult:
        """Perform forward chaining inference."""
        result = ReasoningResult(success=True)
        start_time = time.time()
        
        derived_facts = set(str(fact) for fact in facts)
        new_facts = facts.copy()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            facts_added_this_iteration = False
            
            for rule in rules:
                if not rule.enabled:
                    continue
                
                # Check if all premises are satisfied
                satisfying_substitutions = self._find_satisfying_substitutions(new_facts, rule.premises)
                
                for substitution in satisfying_substitutions:
                    # Apply substitution to conclusion
                    if isinstance(rule.conclusion.content, Predicate):
                        derived_predicate = substitution.apply_to_predicate(rule.conclusion.content)
                        derived_formula = LogicalFormula(
                            formula_id=str(uuid.uuid4()),
                            content=derived_predicate,
                            logic_type=rule.conclusion.logic_type,
                            source=f"rule_{rule.rule_id}"
                        )
                    else:
                        derived_formula = rule.conclusion
                    
                    derived_str = str(derived_formula)
                    if derived_str not in derived_facts:
                        derived_facts.add(derived_str)
                        new_facts.append(derived_formula)
                        result.add_conclusion(derived_formula)
                        facts_added_this_iteration = True
                        
                        rule.usage_count += 1
                        
                        result.reasoning_steps.append({
                            'iteration': iteration,
                            'rule_id': rule.rule_id,
                            'derived_fact': derived_str,
                            'substitution': substitution.bindings,
                            'timestamp': datetime.now(timezone.utc)
                        })
            
            if not facts_added_this_iteration:
                break
        
        result.execution_time = time.time() - start_time
        result.confidence = self._calculate_confidence(result.conclusions)
        
        return result
    
    def _find_satisfying_substitutions(self, facts: List[LogicalFormula], 
                                     premises: List[LogicalFormula]) -> List[Substitution]:
        """Find substitutions that satisfy all premises."""
        if not premises:
            return [Substitution()]
        
        return self._recursive_satisfaction(facts, premises, 0, Substitution())
    
    def _recursive_satisfaction(self, facts: List[LogicalFormula], premises: List[LogicalFormula],
                              premise_index: int, current_substitution: Substitution) -> List[Substitution]:
        """Recursively find satisfying substitutions."""
        if premise_index >= len(premises):
            return [current_substitution]
        
        satisfying_substitutions = []
        current_premise = premises[premise_index]
        
        for fact in facts:
            if isinstance(current_premise.content, Predicate) and isinstance(fact.content, Predicate):
                substitution = self.unification_engine.unify_predicates(
                    current_premise.content, fact.content
                )
                
                if substitution is not None:
                    combined_substitution = current_substitution.compose(substitution)
                    remaining_substitutions = self._recursive_satisfaction(
                        facts, premises, premise_index + 1, combined_substitution
                    )
                    satisfying_substitutions.extend(remaining_substitutions)
        
        return satisfying_substitutions
    
    def _calculate_confidence(self, conclusions: List[LogicalFormula]) -> float:
        """Calculate confidence in the reasoning result."""
        if not conclusions:
            return 0.0
        
        total_confidence = sum(conclusion.confidence for conclusion in conclusions)
        return min(1.0, total_confidence / len(conclusions))


class BackwardChainEngine:
    """Implements backward chaining inference."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.unification_engine = UnificationEngine()
    
    def backward_chain(self, facts: List[LogicalFormula], rules: List[LogicalRule], 
                      goal: LogicalFormula, max_depth: int = 10) -> ReasoningResult:
        """Perform backward chaining to prove a goal."""
        result = ReasoningResult(success=False)
        start_time = time.time()
        
        proof_found, substitutions = self._prove_goal(facts, rules, goal, max_depth, 0)
        
        if proof_found:
            result.success = True
            result.add_conclusion(goal)
            result.substitutions = substitutions
        
        result.execution_time = time.time() - start_time
        result.confidence = 1.0 if proof_found else 0.0
        
        return result
    
    def _prove_goal(self, facts: List[LogicalFormula], rules: List[LogicalRule],
                   goal: LogicalFormula, max_depth: int, current_depth: int) -> Tuple[bool, List[Substitution]]:
        """Recursively prove a goal."""
        if current_depth >= max_depth:
            return False, []
        
        # Check if goal is directly satisfied by facts
        for fact in facts:
            if isinstance(goal.content, Predicate) and isinstance(fact.content, Predicate):
                substitution = self.unification_engine.unify_predicates(goal.content, fact.content)
                if substitution is not None:
                    return True, [substitution]
        
        # Try to prove goal using rules
        for rule in rules:
            if not rule.enabled:
                continue
            
            if isinstance(rule.conclusion.content, Predicate) and isinstance(goal.content, Predicate):
                substitution = self.unification_engine.unify_predicates(
                    rule.conclusion.content, goal.content
                )
                
                if substitution is not None:
                    # Try to prove all premises
                    all_premises_proven = True
                    combined_substitutions = [substitution]
                    
                    for premise in rule.premises:
                        if isinstance(premise.content, Predicate):
                            substituted_premise = LogicalFormula(
                                formula_id=str(uuid.uuid4()),
                                content=substitution.apply_to_predicate(premise.content),
                                logic_type=premise.logic_type
                            )
                        else:
                            substituted_premise = premise
                        
                        premise_proven, premise_substitutions = self._prove_goal(
                            facts, rules, substituted_premise, max_depth, current_depth + 1
                        )
                        
                        if not premise_proven:
                            all_premises_proven = False
                            break
                        
                        # Combine substitutions
                        new_combined = []
                        for cs in combined_substitutions:
                            for ps in premise_substitutions:
                                new_combined.append(cs.compose(ps))
                        combined_substitutions = new_combined
                    
                    if all_premises_proven:
                        return True, combined_substitutions
        
        return False, []


class ConstraintSolver:
    """Solves constraint satisfaction problems."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def solve_csp(self, variables: List[str], domains: Dict[str, List[Any]], 
                  constraints: List[Callable], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Solve constraint satisfaction problem."""
        start_time = time.time()
        
        assignment = {}
        if self._backtrack_search(variables, domains, constraints, assignment, start_time, timeout):
            return assignment
        
        return None
    
    def _backtrack_search(self, variables: List[str], domains: Dict[str, List[Any]],
                         constraints: List[Callable], assignment: Dict[str, Any],
                         start_time: float, timeout: float) -> bool:
        """Backtracking search for CSP solution."""
        if time.time() - start_time > timeout:
            return False
        
        if len(assignment) == len(variables):
            return True  # Complete assignment found
        
        # Select unassigned variable
        var = self._select_unassigned_variable(variables, assignment)
        
        # Try each value in domain
        for value in domains[var]:
            if self._is_consistent(var, value, assignment, constraints):
                assignment[var] = value
                
                if self._backtrack_search(variables, domains, constraints, assignment, start_time, timeout):
                    return True
                
                del assignment[var]  # Backtrack
        
        return False
    
    def _select_unassigned_variable(self, variables: List[str], assignment: Dict[str, Any]) -> str:
        """Select next unassigned variable (MRV heuristic)."""
        unassigned = [var for var in variables if var not in assignment]
        return unassigned[0] if unassigned else ""
    
    def _is_consistent(self, var: str, value: Any, assignment: Dict[str, Any], 
                      constraints: List[Callable]) -> bool:
        """Check if assignment is consistent with constraints."""
        test_assignment = assignment.copy()
        test_assignment[var] = value
        
        for constraint in constraints:
            try:
                if not constraint(test_assignment):
                    return False
            except KeyError:
                # Constraint involves unassigned variables
                continue
        
        return True


class FuzzyLogicEngine:
    """Implements fuzzy logic reasoning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def fuzzy_and(self, a: float, b: float) -> float:
        """Fuzzy AND operation (minimum)."""
        return min(a, b)
    
    def fuzzy_or(self, a: float, b: float) -> float:
        """Fuzzy OR operation (maximum)."""
        return max(a, b)
    
    def fuzzy_not(self, a: float) -> float:
        """Fuzzy NOT operation."""
        return 1.0 - a
    
    def fuzzy_implies(self, a: float, b: float) -> float:
        """Fuzzy implication (Gödel implication)."""
        return 1.0 if a <= b else b
    
    def evaluate_fuzzy_rule(self, premises: List[float], conclusion_truth: float) -> float:
        """Evaluate fuzzy rule with given premise truth values."""
        if not premises:
            return conclusion_truth
        
        # Combine premises with AND
        premise_truth = premises[0]
        for truth in premises[1:]:
            premise_truth = self.fuzzy_and(premise_truth, truth)
        
        # Apply implication
        return self.fuzzy_implies(premise_truth, conclusion_truth)


class ExplanationGenerator:
    """Generates human-readable explanations for reasoning."""
    
    def __init__(self, language_chain: Optional[LanguageChain] = None):
        self.language_chain = language_chain
        self.logger = get_logger(__name__)
    
    async def explain_reasoning(self, result: ReasoningResult, context: Dict[str, Any] = None) -> str:
        """Generate explanation for reasoning result."""
        if not result.success or not result.conclusions:
            return "No conclusions could be derived from the given premises."
        
        explanation_parts = []
        
        # Explain conclusions
        if len(result.conclusions) == 1:
            explanation_parts.append(f"Based on the provided information, I can conclude that: {result.conclusions[0]}")
        else:
            explanation_parts.append("Based on the provided information, I can conclude the following:")
            for i, conclusion in enumerate(result.conclusions, 1):
                explanation_parts.append(f"{i}. {conclusion}")
        
        # Explain reasoning steps
        if result.reasoning_steps:
            explanation_parts.append("\nReasoning process:")
            for step in result.reasoning_steps:
                if 'rule_id' in step and 'derived_fact' in step:
                    explanation_parts.append(
                        f"- Applied rule {step['rule_id']} to derive: {step['derived_fact']}"
                    )
        
        # Add confidence information
        if result.confidence < 1.0:
            confidence_pct = int(result.confidence * 100)
            explanation_parts.append(f"\nConfidence level: {confidence_pct}%")
        
        explanation = "\n".join(explanation_parts)
        
        # Use language chain for natural language generation if available
        if self.language_chain:
            try:
                natural_explanation = await self.language_chain.generate_explanation(
                    explanation, context or {}
                )
                return natural_explanation
            except Exception as e:
                self.logger.warning(f"Failed to generate natural language explanation: {str(e)}")
        
        return explanation
    
    def explain_proof(self, proof: Proof) -> str:
        """Generate explanation for a proof."""
        if not proof.valid:
            return "The proof is invalid or incomplete."
        
        explanation_parts = [f"Proof of: {proof.conclusion}"]
        explanation_parts.append(f"Method: {proof.method}")
        explanation_parts.append("Steps:")
        
        for step in proof.steps:
            explanation_parts.append(f"{step['step_number']}. {step['description']}")
        
        return "\n".join(explanation_parts)


class KnowledgeBase:
    """Manages logical facts and rules."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager
        self.logger = get_logger(__name__)
        
        # In-memory storage
        self.facts: Dict[str, LogicalFormula] = {}
        self.rules: Dict[str, LogicalRule] = {}
        self.fact_index: Dict[str, Set[str]] = defaultdict(set)  # predicate_name -> fact_ids
        self.rule_index: Dict[str, Set[str]] = defaultdict(set)  # conclusion_predicate -> rule_ids
        
        # Statistics
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    async def add_fact(self, fact: LogicalFormula, persist: bool = True) -> None:
        """Add a fact to the knowledge base."""
        with self.lock:
            self.facts[fact.formula_id] = fact
            
            # Update index
            if isinstance(fact.content, Predicate):
                self.fact_index[fact.content.name].add(fact.formula_id)
            
            # Persist to memory if available
            if persist and self.memory_manager:
                try:
                    await self.memory_manager.store_semantic_memory(
                        content=str(fact),
                        memory_type="logical_fact",
                        metadata=fact.metadata
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to persist fact: {str(e)}")
    
    async def add_rule(self, rule: LogicalRule, persist: bool = True) -> None:
        """Add a rule to the knowledge base."""
        with self.lock:
            self.rules[rule.rule_id] = rule
            
            # Update index
            if isinstance(rule.conclusion.content, Predicate):
                self.rule_index[rule.conclusion.content.name].add(rule.rule_id)
            
            # Persist to memory if available
            if persist and self.memory_manager:
                try:
                    await self.memory_manager.store_semantic_memory(
                        content=str(rule),
                        memory_type="logical_rule",
                        metadata=rule.metadata
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to persist rule: {str(e)}")
    
    def get_facts_by_predicate(self, predicate_name: str) -> List[LogicalFormula]:
        """Get all facts with a specific predicate name."""
        with self.lock:
            fact_ids = self.fact_index.get(predicate_name, set())
            return [self.facts[fact_id] for fact_id in fact_ids if fact_id in self.facts]
    
    def get_rules_by_conclusion(self, predicate_name: str) -> List[LogicalRule]:
        """Get all rules that conclude a specific predicate."""
        with self.lock:
            rule_ids = self.rule_index.get(predicate_name, set())
            return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    def get_all_facts(self) -> List[LogicalFormula]:
        """Get all facts in the knowledge base."""
        with self.lock:
            return list(self.facts.values())
    
    def get_all_rules(self) -> List[LogicalRule]:
        """Get all rules in the knowledge base."""
        with self.lock:
            return list(self.rules.values())
    
    def remove_fact(self, fact_id: str) -> bool:
        """Remove a fact from the knowledge base."""
        with self.lock:
            if fact_id in self.facts:
                fact = self.facts[fact_id]
                del self.facts[fact_id]
                
                # Update index
                if isinstance(fact.content, Predicate):
                    self.fact_index[fact.content.name].discard(fact_id)
                
                return True
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the knowledge base."""
        with self.lock:
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                del self.rules[rule_id]
                
                # Update index
                if isinstance(rule.conclusion.content, Predicate):
                    self.rule_index[rule.conclusion.content.name].discard(rule_id)
                
                return True
            return False
    
    def query_facts(self, pattern: str, max_results: int = 100) -> List[LogicalFormula]:
        """Query facts using a pattern."""
        with self.lock:
            self.query_count += 1
            
            results = []
            for fact in self.facts.values():
                if pattern.lower() in str(fact).lower():
                    results.append(fact)
                    if len(results) >= max_results:
                        break
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        with self.lock:
            return {
                'total_facts': len(self.facts),
                'total_rules': len(self.rules),
                'query_count': self.query_count,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'fact_predicates': len(self.fact_index),
                'rule_conclusions': len(self.rule_index)
            }


class LogicEngine:
    """
    Advanced Logic Engine for the AI Assistant.
    
    This engine provides comprehensive logical reasoning capabilities including:
    - Propositional and predicate logic
    - Rule-based reasoning with forward/backward chaining
    - Resolution theorem proving
    - Constraint satisfaction problems
    - Fuzzy logic reasoning
    - Automated proof generation
    - Natural language explanation generation
    - Knowledge base management with persistence
    - Integration with memory and learning systems
    - Performance optimization and monitoring
    """
    
    def __init__(self, container: Container):
        """
        Initialize the logic engine.
        
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
        
        # Memory integration
        try:
            self.memory_manager = container.get(MemoryManager)
            self.context_manager = container.get(ContextManager)
            self.semantic_memory = container.get(SemanticMemory)
            self.working_memory = container.get(WorkingMemory)
        except Exception:
            self.memory_manager = None
            self.context_manager = None
            self.semantic_memory = None
            self.working_memory = None
        
        # Learning integration
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.continual_learner = None
            self.feedback_processor = None
        
        # Processing integration
        try:
            self.language_chain = container.get(LanguageChain)
            self.entity_extractor = container.get(EntityExtractor)
        except Exception:
            self.language_chain = None
            self.entity_extractor = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Initialize reasoning engines
        self.forward_chain_engine = ForwardChainEngine()
        self.backward_chain_engine = BackwardChainEngine()
        self.resolution_engine = ResolutionEngine()
        self.constraint_solver = ConstraintSolver()
        self.fuzzy_engine = FuzzyLogicEngine()
        
        # Knowledge management
        self.knowledge_base = KnowledgeBase(self.memory_manager)
        self.explanation_generator = ExplanationGenerator(self.language_chain)
        
        # Performance and caching
        self.query_cache: Dict[str, ReasoningResult] = {}
        self.proof_cache: Dict[str, Proof] = {}
        self.reasoning_stats: Dict[str, Any] = defaultdict(int)
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="logic_engine")
        self.reasoning_semaphore = asyncio.Semaphore(10)
        
        # Configuration
        self.default_timeout = self.config.get("logic.default_timeout", 30.0)
        self.max_reasoning_depth = self.config.get("logic.max_reasoning_depth", 10)
        self.enable_caching = self.config.get("logic.enable_caching", True)
        self.enable_learning = self.config.get("logic.enable_learning", True)
        self.cache_ttl = self.config.get("logic.cache_ttl", 3600)
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("logic_engine", self._health_check_callback)
        
        # Load built-in operators and rules
        self._setup_logical_operators()
        self._load_builtin_rules()
        
        self.logger.info("LogicEngine initialized successfully")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register logic engine metrics
            self.metrics.register_counter("logic_queries_total")
            self.metrics.register_counter("logic_inferences_total")
            self.metrics.register_counter("logic_proofs_generated")
            self.metrics.register_histogram("logic_reasoning_duration_seconds")
            self.metrics.register_gauge("logic_knowledge_base_size")
            self.metrics.register_counter("logic_cache_hits")
            self.metrics.register_counter("logic_cache_misses")
            self.metrics.register_counter("logic_errors_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _setup_logical_operators(self) -> None:
        """Setup standard logical operators."""
        self.logical_operators = {
            '∧': LogicalOperator('∧', 'and', 2, 3, 'left', lambda a, b: TruthValue.TRUE if a == TruthValue.TRUE and b == TruthValue.TRUE else TruthValue.FALSE),
            '∨': LogicalOperator('∨', 'or', 2, 2, 'left', lambda a, b: TruthValue.TRUE if a == TruthValue.TRUE or b == TruthValue.TRUE else TruthValue.FALSE),
            '¬': LogicalOperator('¬', 'not', 1, 4, 'right', lambda a: TruthValue.TRUE if a == TruthValue.FALSE else TruthValue.FALSE),
            '→': LogicalOperator('→', 'implies', 2, 1, 'right', lambda a, b: TruthValue.FALSE if a == TruthValue.TRUE and b == TruthValue.FALSE else TruthValue.TRUE),
            '↔': LogicalOperator('↔', 'iff', 2, 1, 'left', lambda a, b: TruthValue.TRUE if a == b else TruthValue.FALSE),
            '∀': LogicalOperator('∀', 'forall', 1, 5, 'right'),
            '∃': LogicalOperator('∃', 'exists', 1, 5, 'right')
        }

    def _load_builtin_rules(self) -> None:
        """Load built-in logical rules."""
        # Modus ponens: P → Q, P ⊢ Q
        modus_ponens = LogicalRule(
            rule_id="modus_ponens",
            premises=[
                LogicalFormula("mp_premise1", "P → Q"),
                LogicalFormula("mp_premise2", "P")
            ],
            conclusion=LogicalFormula("mp_conclusion", "Q"),
            rule_type=RuleType.IMPLICATION,
            metadata={"builtin": True, "description": "Modus ponens inference rule"}
        )
        
        # Modus tollens: P → Q, ¬Q ⊢ ¬P
        modus_tollens = LogicalRule(
            rule_id="modus_tollens",
            premises=[
                LogicalFormula("mt_premise1", "P → Q"),
                LogicalFormula("mt_premise2", "¬Q")
            ],
            conclusion=LogicalFormula("mt_conclusion", "¬P"),
            rule_type=RuleType.IMPLICATION,
            metadata={"builtin": True, "description": "Modus tollens inference rule"}
        )
        
        # Store built-in rules
        asyncio.create_task(self.knowledge_base.add_rule(modus_ponens, persist=False))
        asyncio.create_task(self.knowledge_base.add_rule(modus_tollens, persist=False))

    async def initialize(self) -> None:
        """Initialize the logic engine."""
        try:
            # Load persisted knowledge base if available
            await self._load_persisted_knowledge()
            
            # Start background tasks
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("LogicEngine initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LogicEngine: {str(e)}")
            raise LogicError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Memory events
        self.event_bus.subscribe("memory_updated", self._handle_memory_update)
        
        # Learning events
        if self.enable_learning:
            self.event_bus.subscribe("feedback_received", self._handle_feedback)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def _load_persisted_knowledge(self) -> None:
        """Load knowledge base from persistent storage."""
        if not self.memory_manager:
            return
        
        try:
            # Load facts
            fact_memories = await self.memory_manager.retrieve_memories(
                memory_type="logical_fact",
                limit=10000
            )
            
            for memory in fact_memories:
                try:
                    # Parse fact from memory content
                    fact = self._parse_fact_from_string(memory.get('content', ''))
                    if fact:
                        await self.knowledge_base.add_fact(fact, persist=False)
                except Exception as e:
                    self.logger.warning(f"Failed to load fact from memory: {str(e)}")
            
            # Load rules
            rule_memories = await self.memory_manager.retrieve_memories(
                memory_type="logical_rule",
                limit=1000
            )
            
            for memory in rule_memories:
                try:
                    # Parse rule from memory content
                    rule = self._parse_rule_from_string(memory.get('content', ''))
                    if rule:
                        await self.knowledge_base.add_rule(rule, persist=False)
                except Exception as e:
                    self.logger.warning(f"Failed to load rule from memory: {str(e)}")
            
            self.logger.info(f"Loaded knowledge base: {len(self.knowledge_base.facts)} facts, {len(self.knowledge_base.rules)} rules")
            
        except Exception as e:
            self.logger.warning(f"Failed to load persisted knowledge: {str(e)}")

    def _parse_fact_from_string(self, fact_str: str) -> Optional[LogicalFormula]:
        """Parse a logical fact from string representation."""
        try:
            # Simplified parsing - in practice would use a proper parser
            if '(' in fact_str and ')' in fact_str:
                # Predicate format: predicate_name(args)
                name = fact_str.split('(')[0].strip()
                args_str = fact_str.split('(')[1].split(')')[0]
                
                if args_str:
                    arg_names = [arg.strip() for arg in args_str.split(',')]
                    args = [Term(arg_name) for arg_name in arg_names]
                else:
                    args = []
                
                predicate = Predicate(name=name, arguments=args)
                
                return LogicalFormula(
                    formula_id=str(uuid.uuid4()),
                    content=predicate,
                    logic_type=LogicType.PREDICATE
                )
            else:
                # Propositional format
                return LogicalFormula(
                    formula_id=str(uuid.uuid4()),
                    content=fact_str,
                    logic_type=LogicType.PROPOSITIONAL
                )
        except Exception:
            return None

    def _parse_rule_from_string(self, rule_str: str) -> Optional[LogicalRule]:
        """Parse a logical rule from string representation."""
        try:
            # Simplified parsing for implication rules
            if '→' in rule_str:
                parts = rule_str.split('→')
                if len(parts) == 2:
                    premise_str = parts[0].strip()
                    conclusion_str = parts[1].strip()
                    
                    # Parse premise (may have multiple conjuncts)
                    if '∧' in premise_str:
                        premise_parts = premise_str.split('∧')
                        premises = [self._parse_fact_from_string(p.strip()) for p in premise_parts]
                        premises = [p for p in premises if p]
                    else:
                        premise = self._parse_fact_from_string(premise_str)
                        premises = [premise] if premise else []
                    
                    # Parse conclusion
                    conclusion = self._parse_fact_from_string(conclusion_str)
                    
                    if premises and conclusion:
                        return LogicalRule(
                            rule_id=str(uuid.uuid4()),
                            premises=premises,
                            conclusion=conclusion,
                            rule_type=RuleType.IMPLICATION
                        )
        except Exception:
            return None

    @handle_exceptions
    async def reason(
        self,
        facts: Optional[List[LogicalFormula]] = None,
        rules: Optional[List[LogicalRule]] = None,
        query: Optional[Union[LogicalFormula, str]] = None,
        mode: ReasoningMode = ReasoningMode.FORWARD_CHAINING,
        timeout: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Perform logical reasoning.
        
        Args:
            facts: List of facts to reason with
            rules: List of rules to apply
            query: Optional query to prove/answer
            mode: Reasoning mode to use
            timeout: Timeout in seconds
            context: Additional context for reasoning
            
        Returns:
            Reasoning result
        """
        async with self.reasoning_semaphore:
            start_time = time.time()
            timeout = timeout or self.default_timeout
            context = context or {}
            
            # Use knowledge base facts/rules if not provided
            if facts is None:
                facts = self.knowledge_base.get_all_facts()
            if rules is None:
                rules = self.knowledge_base.get_all_rules()
            
            # Create cache key
            cache_key = self._create_cache_key(facts, rules, query, mode)
            
            # Check cache
            if self.enable_caching and cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if time.time() - cached_result.metadata.get('cache_time', 0) < self.cache_ttl:
                    self.metrics.increment("logic_cache_hits")
                    return cached_result
                else:
                    del self.query_cache[cache_key]
            
            try:
                with self.tracer.trace("logic_reasoning") as span:
                    span.set_attributes({
                        "reasoning_mode": mode.value,
                        "facts_count": len(facts),
                        "rules_count": len(rules),
                        "has_query": query is not None
                    })
                    
                    # Emit reasoning started event
                    await self.event_bus.emit(LogicReasoningStarted(
                        reasoning_mode=mode.value,
                        facts_count=len(facts),
                        rules_count=len(rules),
                        context=context
                    ))
                    
                    # Perform reasoning based on mode
                    if mode == ReasoningMode.FORWARD_CHAINING:
                        result = self.forward_chain_engine.forward_chain(facts, rules)
                    elif mode == ReasoningMode.BACKWARD_CHAINING and query:
                        if isinstance(query, str):
                            query_formula = self._parse_fact_from_string(query)
                        else:
                            query_formula = query
                        
                        if query_formula:
                            result = self.backward_chain_engine.backward_chain(
                                facts, rules, query_formula, self.max_reasoning_depth
                            )
                        else:
                            result = ReasoningResult(success=False)
                    elif mode == ReasoningMode.RESOLUTION and query:
                        if isinstance(query, str):
                            query_formula = self._parse_fact_from_string(query)
                        else:
                            query_formula = query
                        
                        if query_formula:
                            proof = self.resolution_engine.prove_by_resolution(facts, query_formula)
                            result = ReasoningResult(success=proof.valid if proof else False)
                            if proof:
                                result.proofs.append(proof)
                                result.add_conclusion(query_formula, proof)
                        else:
                            result = ReasoningResult(success=False)
                    else:
                        # Default to forward chaining
                        result = self.forward_chain_engine.forward_chain(facts, rules)
                    
                    # Add metadata
                    result.metadata['reasoning_mode'] = mode.value
                    result.metadata['cache_time'] = time.time()
                    result.metadata['context'] = context
                    
                    # Generate explanation if requested
                    if context.get('generate_explanation', False):
                        result.explanation = await self.explanation_generator.explain_reasoning(result, context)
                    
                    # Cache result
                    if self.enable_caching:
                        self.query_cache[cache_key] = result
                        self.metrics.increment("logic_cache_misses")
                    
                    # Update statistics
                    self.reasoning_stats['total_inferences'] += len(result.conclusions)
                    self.reasoning_stats['total_queries'] += 1
                    
                    # Update metrics
                    self.metrics.increment("logic_queries_total")
                    self.metrics.increment("logic_inferences_total", len(result.conclusions))
                    self.metrics.record("logic_reasoning_duration_seconds", result.execution_time)
                    
                    return result
                    
            except Exception as e:
                    self.logger.error(f"Error in logical reasoning: {str(e)}")
                    raise
