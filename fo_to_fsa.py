from itertools import product

from formula import (
    Conjunction,
    Disjunction,
    ExistentialQuantifier,
    Negation,
    Predicate,
    Relation,
    SymbolPredicate,
    UniversalQuantifier,
)
from fsa import FiniteStateAutomaton, State


class FOtoFSA:
    """
    Converts First-Order logic formulas to equivalent Finite State Automata
    following Straubing's construction.

    This implementation handles the two-sorted view of words:
    - Positions (first-order variables)
    - Labels (predicates on positions)

    And works with arbitrary alphabets using Q_a predicates.
    """

    def __init__(self, alphabet=None):
        """
        Initialize the converter.

        Args:
            alphabet: List of symbols in the alphabet.
                     If None, defaults to {'a', 'b'} for a binary alphabet
        """
        self.counter = 0

        # Define the alphabet
        if alphabet is None:
            # Default binary alphabet
            self.alphabet = ["a", "b"]
        else:
            self.alphabet = alphabet

    def _powerset(self, V):
        """
        Generate the powerset of a set V (all possible subsets).

        Args:
            V: A set of elements

        Returns:
            A list of all possible subsets of V
        """
        result = [[]]  # Start with the empty set
        for element in V:
            # For each element, add it to all existing subsets
            result.extend([subset + [element] for subset in result])

        # Convert lists back to frozensets for immutability and hashability
        return [tuple(subset) for subset in result]

    def convert(self, formula):
        """
        Convert a First-Order formula to an FSA

        Args:
            formula: An instance of FOFormula

        Returns:
            A FiniteStateAutomaton equivalent to the formula
        """
        V = formula.get_variables()
        V = sorted(V)  # Sort the variables for deterministic order

        V_alphabet = list(product(self.alphabet, self._powerset(V)))

        V_structure_fsa = self._construct_V_structure_fsa(V, V_alphabet)

        # Apply Straubing's construction recursively
        formula_fsa = self._convert(formula, V, V_alphabet)

        # Intersect the structure FSA with the formula FSA
        # fsa = V_structure_fsa.intersect(formula_fsa)

        # Always trim and minimize the resulting automaton
        # fsa = fsa.trim().minimize()
        fsa = formula_fsa.trim().minimize()

        if formula.is_sentence():
            return self._project(fsa)
        else:
            return fsa

    def _project(self, fsa):
        """Project the automaton transitions to alphabet symbols only

        Args:
            fsa: The automaton to project
        """
        # Create a new FSA with the same alphabet but no variables
        proj_fsa = FiniteStateAutomaton(fsa.num_states, self.alphabet)

        # Set initial state
        if fsa.initial_state is not None:
            proj_fsa.set_initial_state(fsa.initial_state.id)

        # Process transitions
        for (src_state, old_symbol), dst_state in fsa.transitions.items():
            symbol, _ = old_symbol

            # Create new symbol without the variable
            new_symbol = symbol
            proj_fsa.set_transition(src_state.id, new_symbol, dst_state.id)

        # Set accepting states
        for state in fsa.accepting_states:
            proj_fsa.set_accepting_state(state.id)

        return proj_fsa.trim().minimize()

    def _construct_V_structure_fsa(self, V, V_alphabet):
        """Constructs the FSA that checks if a string is a valid V-structure.

        Args:
            V: Set of free variables in the formula
        """
        # Create an FSA with 2^|V| states
        fsa = FiniteStateAutomaton(2 ** len(V), V_alphabet)
        P_V = self._powerset(V)
        v2idx = {v: i for i, v in enumerate(P_V)}

        # State empty set is initial state
        fsa.set_initial_state(v2idx[()])

        # For every subset of V and every symbol in the subset, only advance
        # to the next subset if the symbol has not been seen in the string yet
        for symbol, subset in fsa.alphabet:
            outgoing_subsets = self._powerset(set(V) - set(subset))
            for out_subset in outgoing_subsets:
                next_subset = tuple(sorted(set(subset + out_subset)))
                fsa.set_transition(
                    v2idx[subset], (symbol, out_subset), v2idx[next_subset]
                )

            fsa.set_transition(v2idx[subset], (symbol, ()), v2idx[subset])

        # All states are accepting states
        for state in range(fsa.num_states):
            fsa.set_accepting_state(state)

        return fsa.trim().minimize()

    def _convert(self, formula, V, alphabet):
        """Internal recursive conversion method"""
        if isinstance(formula, Predicate):
            return self._convert_predicate(formula, alphabet)
        elif isinstance(formula, Relation):
            return self._convert_relation(formula, alphabet)
        elif isinstance(formula, Negation):
            return self._convert(formula.subformula, V, alphabet).complement()
        elif isinstance(formula, Conjunction):
            return self._convert(formula.left, V, alphabet).intersect(
                self._convert(formula.right, V, alphabet)
            )
        elif isinstance(formula, Disjunction):
            return self._convert(formula.left, V, alphabet).union(
                self._convert(formula.right, V, alphabet)
            )
        elif isinstance(formula, ExistentialQuantifier):
            # For existential quantification, we project out the variable
            return self._remove_variable(
                self._convert(formula.subformula, V, alphabet), V, formula.variable
            )
        elif isinstance(formula, UniversalQuantifier):
            # For universal quantification, we use the equivalence: ∀x.φ(x) ≡ ¬∃x.¬φ(x)
            negated_sub = Negation(formula.subformula)
            existential = ExistentialQuantifier(formula.variable, negated_sub)
            return self._convert(existential, V, alphabet)
        else:
            raise ValueError(f"Unsupported formula type: {type(formula).__name__}")

    def _convert_predicate(self, predicate, alphabet):
        """Convert a position predicate to an FSA"""
        # We only support Q_a predicates (testing for specific alphabet symbols)
        if isinstance(predicate, SymbolPredicate):
            symbol = predicate.symbol  # Extract the symbol (e.g., "a" from "Q_a")
            variable = predicate.variable  # Extract the variable (e.g., "x" from "Q_a")

            # Check if the symbol is in our alphabet
            if symbol not in self.alphabet:
                raise ValueError(
                    f"Symbol '{symbol}' not found in alphabet: {self.alphabet}"
                )

            # Create an automaton with 2 states:
            # State 0: Initial state, waiting to see the symbol
            # State 1: Accepting state, saw the symbol
            fsa = FiniteStateAutomaton(2, alphabet)

            # Set initial state
            fsa.set_initial_state(0)

            # For each symbol in the alphabet
            for alph_symbol in fsa.alphabet:
                if alph_symbol[0] == symbol and variable in alph_symbol[1]:
                    # When we see the matching symbol, go to accepting state
                    fsa.set_transition(0, alph_symbol, 1)
                else:
                    # When we see any other symbol, stay in initial state
                    fsa.set_transition(0, alph_symbol, 0)

                # Once in accepting state, stay there regardless of symbol
                fsa.set_transition(1, alph_symbol, 1)

            # Mark the accepting state
            fsa.set_accepting_state(1)

            return fsa.trim().minimize()
        else:
            raise ValueError(
                f"Unsupported predicate: {predicate.name}. Only Q_a predicates are supported."
            )

    def _convert_relation(self, relation, alphabet):
        """Convert a numerical relation to an FSA following Straubing's construction"""
        left, op, right = relation.left, relation.operator, relation.right

        # Equality: x = y
        if op == "=" and not (right.endswith("+1") or left.endswith("+1")):
            return self._relation_eq(left, right, alphabet)

        # Successor: x = y+1
        elif op == "=" and right.endswith("+1"):
            base_var = right[:-2]  # Remove "+1"
            return self._relation_succ(left, base_var, alphabet)

        # Predecessor: y = x+1 (rewritten as x = y-1 or x+1 = y)
        elif op == "=" and left.endswith("+1"):
            base_var = left[:-2]  # Remove "+1"
            return self._relation_succ(right, base_var, alphabet)

        # Less than: x < y
        elif op == "<":
            return self._relation_lt(left, right, alphabet)

        # Greater than: x > y (rewritten as y < x)
        elif op == ">":
            return self._relation_lt(right, left, alphabet)

        # Less than or equal: x ≤ y (rewritten as ¬(y < x))
        elif op == "<=":
            return self._relation_lt(right, left, alphabet).complement()

        # Greater than or equal: x ≥ y (rewritten as ¬(x < y))
        elif op == ">=":
            return self._relation_lt(left, right, alphabet).complement()

        else:
            raise ValueError(f"Unsupported relation: {left} {op} {right}")

    def _relation_eq(self, var1, var2, alphabet):
        """x = y: both variables refer to the same position"""
        # The equality relation is satisfied by all strings
        fsa = FiniteStateAutomaton(1, self.alphabet)

        # For any symbol, stay in the accepting state
        for symbol in self.alphabet:
            fsa.set_transition(0, symbol, 0)

        fsa.set_initial_state(0)
        fsa.set_accepting_state(0)
        return fsa.trim().minimize()

    def _relation_succ(self, var1, var2, alphabet):
        """x = y+1: var1 is immediately after var2"""
        fsa = FiniteStateAutomaton(2, self.alphabet)

        # State 0: Initial state, haven't seen position var2 yet
        # State 1: Accepting state, we're at position var1 (immediately after var2)

        # Any symbol at position var2 transitions to state 1 (next position)
        for symbol in self.alphabet:
            fsa.set_transition(0, symbol, 1)
            # Any symbol after accepting state keeps us in accepting state
            fsa.set_transition(1, symbol, 1)

        fsa.set_initial_state(0)
        fsa.set_accepting_state(1)
        return fsa.trim().minimize()

    def _relation_lt(self, var1, var2, alphabet):
        """x < y: var1 position is strictly less than var2 position"""
        fsa = FiniteStateAutomaton(3, alphabet)

        # State 0: Initial state, haven't seen either position
        # State 1: Seen position var1, waiting for var2
        # State 2: Accepting state, seen both positions in correct order

        # For every symbol in the alphabet
        for V_symbol in fsa.alphabet:
            _, vars_at_pos = V_symbol

            if var1 in vars_at_pos and var2 not in vars_at_pos:
                # Transition to state 1 when var1 is seen and var2 is not
                fsa.set_transition(0, V_symbol, 1)
            elif var2 in vars_at_pos and var1 not in vars_at_pos:
                # Transition to state 2 when var2 is seen after var1
                fsa.set_transition(1, V_symbol, 2)
            elif var1 not in vars_at_pos and var2 not in vars_at_pos:
                # Stay in the same state if neither var1 nor var2 is seen
                fsa.set_transition(0, V_symbol, 0)
                fsa.set_transition(1, V_symbol, 1)
                fsa.set_transition(2, V_symbol, 2)

        # Set initial state
        fsa.set_initial_state(0)
        # Set accepting state
        fsa.set_accepting_state(2)

        return fsa.trim().minimize()

    def _remove_variable(self, fsa, V, variable):
        """
        Implement projection to handle existential quantification following Straubing's approach.

        In this approach, we don't use epsilon transitions, but instead create a modified FSA
        that accepts the projection of the language recognized by the input FSA.

        Args:
            fsa: The automaton to project
            V: List of variables
            variable: The variable to project out

        Returns:
            A new FSA that recognizes the projection
        """
        # Create new alphabet without the projected variable
        V_new = sorted(set(V) - {variable})
        alphabet_new = list(product(self.alphabet, self._powerset(V_new)))

        # Create a new FSA for the result with the new alphabet
        # Each state in the new FSA will track:
        # 1. A state from the original FSA
        # 2. A bit indicating whether the variable has been seen (0=no, 1=yes)
        result = FiniteStateAutomaton(fsa.num_states * 2, alphabet_new)

        # Create a mapping from (original_state, seen_bit) to new state
        p2idx = {p: i for i, p in enumerate(product(range(fsa.num_states), [0, 1]))}

        print(f"Mapping: {p2idx}")

        # Set initial state - we start with the original initial state and variable not seen
        if fsa.initial_state is not None:
            result.set_initial_state(p2idx[(fsa.initial_state.id, 0)])
            print(f"Initial state: {fsa.initial_state.id} with variable not seen")

        # Process transitions from the original FSA
        for (src_state, old_symbol), dst_state in fsa.transitions.items():
            symbol, vars_at_pos = old_symbol

            print()
            print(f"Transition: {src_state.id} -> {dst_state.id} with {old_symbol}")
            print()

            # Create new symbol without the projected variable
            new_vars = tuple(sorted([v for v in vars_at_pos if v != variable]))
            new_symbol = (symbol, new_vars)

            if variable in vars_at_pos:
                # This is a position where the variable occurs
                # Add transition that "records" seeing the variable
                result.set_transition(
                    p2idx[(src_state.id, 0)],  # From state where variable not seen
                    new_symbol,  # New symbol (without the variable)
                    p2idx[(dst_state.id, 1)],  # To state where variable is seen
                )

                print(
                    f"Transition: {src_state.id} -> {dst_state.id} with {new_symbol} seen"
                )

                # !This seems to be in conflict with Straubing's construction,
                # !but it seems necessary
                # Also maintain existing transitions when variable was already seen
                result.set_transition(
                    p2idx[
                        (src_state.id, 1)
                    ],  # From state where variable was already seen
                    new_symbol,  # New symbol
                    p2idx[(dst_state.id, 1)],  # To state where variable is still seen
                )

                print(
                    f"Transition: {src_state.id} -> {dst_state.id} with {new_symbol} already seen"
                )
            else:
                # This is a position where the variable doesn't occur
                # Just copy the transitions, preserving the "seen" bit
                for u in range(2):
                    result.set_transition(
                        p2idx[(src_state.id, u)],  # From state where variable not seen
                        new_symbol,  # New symbol
                        p2idx[
                            (dst_state.id, u)
                        ],  # To state where variable still not seen
                    )

                    print(
                        f"Transition: {(src_state.id, u)}[{p2idx[(src_state.id, u)]}] -> {(dst_state.id, u)}[{p2idx[(dst_state.id, u)]}] with {new_symbol} not seen"
                    )

        # Set accepting states - a state is accepting if it corresponds to an accepting
        # state in the original FSA and the variable has been seen
        for state in fsa.accepting_states:
            result.set_accepting_state(p2idx[(state.id, 1)])
            print(f"Accepting state: {state.id} with variable seen")

        print()
        print()
        print()

        # Remove any unnecessary states and transitions
        return result.trim().minimize()


def convert_fo_to_fsa(formula, alphabet=None):
    """
    Convert a First-Order formula to an FSA following Straubing's construction.

    Args:
        formula: An FOFormula instance or formula string
        alphabet: List of symbols in the alphabet. If None, defaults to the
            alphabet of symbols in the formula's predicates.

    Returns:
        A FiniteStateAutomaton equivalent to the formula
    """
    from parser import parse_fo_formula

    # Parse string formulas
    if isinstance(formula, str):
        formula = parse_fo_formula(formula)

    # Convert to FSA
    converter = FOtoFSA(formula.get_alphabet() if alphabet is None else alphabet)
    automaton = converter.convert(formula)

    return automaton
