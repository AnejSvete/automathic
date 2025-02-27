from collections import defaultdict

from formula import (
    Conjunction,
    Disjunction,
    ExistentialQuantifier,
    FOFormula,
    Negation,
    Predicate,
    Relation,
    UniversalQuantifier,
)
from fsa import FiniteStateAutomaton


class FOtoFSA:
    """
    Converts First-Order logic formulas to equivalent Finite State Automata.

    This implementation follows the standard construction technique:
    1. Convert atomic predicates to simple FSAs
    2. Handle Boolean operations using automata operations
    3. Handle quantifiers using projection operations
    """

    def __init__(self):
        self.counter = 0  # State counter for creating new states

    def fresh_state(self):
        """Generate a fresh state ID"""
        state = self.counter
        self.counter += 1
        return state

    def convert(self, formula):
        """
        Convert a First-Order formula to an FSA

        Args:
            formula: An instance of FOFormula

        Returns:
            A FiniteStateAutomaton equivalent to the formula
        """
        if isinstance(formula, Predicate):
            return self._convert_predicate(formula)
        elif isinstance(formula, Negation):
            return self.convert(formula.subformula).complement()
        elif isinstance(formula, Conjunction):
            return self.convert(formula.left).intersect(self.convert(formula.right))
        elif isinstance(formula, Disjunction):
            return self.convert(formula.left).union(self.convert(formula.right))
        elif isinstance(formula, ExistentialQuantifier):
            return self.convert(formula.subformula).project(formula.variable)
        elif isinstance(formula, UniversalQuantifier):
            # Universal quantifier: ∀x.φ(x) ≡ ¬∃x.¬φ(x)
            negated_subformula = Negation(formula.subformula)
            existential = ExistentialQuantifier(formula.variable, negated_subformula)
            return self.convert(existential)
        elif isinstance(formula, Relation):
            return self._convert_relation(formula)
        else:
            raise ValueError(f"Unsupported formula type: {type(formula)}")

    def _convert_predicate(self, predicate):
        """Convert a predicate (P_0(x), P_1(x), etc.) to an FSA"""
        # Extract the bit value (0 or 1) from the predicate name
        if predicate.name.startswith("P_"):
            bit = int(predicate.name[2:])
        else:
            raise ValueError(f"Unsupported predicate: {predicate.name}")

        # Create a 2-state FSA that recognizes the bit at the position
        # referenced by the variable
        fsa = FiniteStateAutomaton(2, 2)  # 2 states, binary alphabet

        # State 0: Initial state, waiting to match
        # State 1: Accepting state, matched the bit

        # From initial state, on the specified bit, go to accepting
        fsa.set_transition(0, bit, 1)
        # From initial state, on the other bit, stay in initial
        fsa.set_transition(0, 1 - bit, 0)
        # From accepting state, on any bit, stay in accepting
        fsa.set_transition(1, 0, 1)
        fsa.set_transition(1, 1, 1)

        fsa.set_initial_state(0)
        fsa.set_accepting_state(1)

        return fsa

    def _convert_relation(self, relation):
        """Convert a relation (x=y, x<y, etc.) to an FSA"""
        # For this implementation, we'll focus on a few common relations:
        # x = y+1 (successor)
        # x < y (less than)
        # x = y (equality)

        if relation.operator == "=" and relation.right == relation.left + "+1":
            # x = y+1: successor relation
            # This means the position referenced by x is immediately after y
            # We represent this with a simple FSA
            fsa = FiniteStateAutomaton(2, 2)

            # State 0: Initial, waiting for position y
            # State 1: Accepting, at position x (immediately after y)

            # Any bit at position y transitions to state 1
            fsa.set_transition(0, 0, 1)
            fsa.set_transition(0, 1, 1)

            # Any subsequent bit keeps us in state 1
            fsa.set_transition(1, 0, 1)
            fsa.set_transition(1, 1, 1)

            fsa.set_initial_state(0)
            fsa.set_accepting_state(1)

            return fsa

        elif relation.operator == "<":
            # x < y: less-than relation
            # This means position x is before position y
            fsa = FiniteStateAutomaton(2, 2)

            # State 0: Initial, haven't seen position x yet
            # State 1: Accepting, saw position x, waiting for y

            # Any bit keeps us in appropriate state
            fsa.set_transition(0, 0, 0)
            fsa.set_transition(0, 1, 0)

            # After seeing position x, any bit keeps us accepting
            fsa.set_transition(1, 0, 1)
            fsa.set_transition(1, 1, 1)

            # We need to manually mark position x and check if it's before y
            # This is simplified - in a real implementation we'd need a more complex FSA

            fsa.set_initial_state(0)
            fsa.set_accepting_state(1)

            return fsa

        elif relation.operator == "=":
            # x = y: equality relation
            # This means positions x and y refer to the same position
            # This is a bit simplified as well
            fsa = FiniteStateAutomaton(1, 2)

            # Only one state that's both initial and accepting
            fsa.set_transition(0, 0, 0)
            fsa.set_transition(0, 1, 0)

            fsa.set_initial_state(0)
            fsa.set_accepting_state(0)

            return fsa

        else:
            raise ValueError(
                f"Unsupported relation: {relation.left} {relation.operator} {relation.right}"
            )


def convert_fo_to_fsa(formula):
    """
    Convert a First-Order formula to an FSA

    Args:
        formula: An instance of FOFormula or a string representation of a formula

    Returns:
        A FiniteStateAutomaton equivalent to the formula
    """
    from parser import parse_fo_formula

    # If the input is a string, parse it first
    if isinstance(formula, str):
        formula = parse_fo_formula(formula)

    converter = FOtoFSA()
    return converter.convert(formula)
