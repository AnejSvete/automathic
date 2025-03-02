from itertools import product

from automathic.fo.formula import (
    Conjunction,
    ExistentialQuantifier,
    Negation,
    Predicate,
    Relation,
    SymbolPredicate,
)
from automathic.fsa.fsa import FiniteStateAutomaton, State


class FOtoFSA:
    """
    Converts First-Order logic formulas to equivalent Finite State Automata
    following Straubing's construction.

    This implementation handles the two-sorted view of words:
    - Positions (first-order variables)
    - Labels (predicates on positions)

    And works with arbitrary alphabets using symbol predicates (Qa).

    Straubing's construction builds automata compositionally from atomic formulas
    and preserves the semantics of logical operations through automata operations.
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
        Generate the powerset of a set V (all possible subsets) in a deterministic order.
        Used to represent all possible variable assignments at each position.

        Args:
            V: A set of elements (variables)

        Returns:
            A list of all possible subsets of V
        """
        # Convert input to a sorted list to ensure consistent ordering
        elements = sorted(V)

        result = [[]]  # Start with the empty set
        for element in elements:
            # For each element, add it to all existing subsets
            result.extend([subset + [element] for subset in result])

        # Convert lists to tuples for immutability and hashability
        return [tuple(subset) for subset in result]

    def convert(self, formula):
        """
        Convert a First-Order formula to an FSA

        Args:
            formula: An instance of FOFormula

        Returns:
            A FiniteStateAutomaton equivalent to the formula
        """
        # First ensure the formula is in FO[<] form (only existential quantifiers,
        # conjunctions, negations, and < relations)
        formula = formula.to_fo_less()

        # Get free variables in sorted order for consistent processing
        V = sorted(formula.get_free_variables())

        # Construct the enriched alphabet: pairs of (symbol, variable_set)
        # Each pair represents a position in the word with its symbol and variables
        V_alphabet = list(product(self.alphabet, self._powerset(V)))

        # Apply Straubing's construction recursively
        fsa = self._convert(formula, V, V_alphabet)

        # For sentences (formulas with no free variables), project to the original alphabet
        if formula.is_sentence():
            return self._project(fsa)
        else:
            return fsa

    def _project(self, fsa):
        """
        Project the automaton transitions to alphabet symbols only, removing variable information.
        This is used for sentence formulas where we only care about acceptance, not variable assignments.

        Args:
            fsa: The automaton to project

        Returns:
            A projected FSA with the original alphabet
        """
        # Create a new FSA with the same alphabet but no variables
        proj_fsa = FiniteStateAutomaton(fsa.num_states, self.alphabet)

        # Set up states with same origins
        for i, state in enumerate(fsa.states):
            if state is not None:
                # Maintain the origin information
                proj_fsa.states[i] = State(i, origin=state.origin)

        # Set initial state
        if fsa.initial_state is not None:
            proj_fsa.set_initial_state(fsa.initial_state.id)

        # Process transitions
        for (src_state, old_symbol), dst_state in fsa.transitions.items():
            # Extract just the symbol part from (symbol, variables)
            symbol, _ = old_symbol

            # Create new transition with just the symbol
            proj_fsa.set_transition(src_state.id, symbol, dst_state.id)

        # Set accepting states
        for state in fsa.accepting_states:
            proj_fsa.set_accepting_state(state.id)

        # Return trimmed FSA (remove unreachable states)
        return proj_fsa.trim()

    def _construct_V_structure_fsa(self, V, V_alphabet):
        """
        Constructs the FSA that checks if a string is a valid V-structure.

        A valid V-structure assigns each variable to exactly one position,
        with disjoint sets of variables at each position. This is a fundamental
        constraint for representing variable assignments in the automaton.

        Args:
            V: Set of free variables in the formula
            V_alphabet: The enriched alphabet (symbols × variable sets)

        Returns:
            An FSA accepting only valid variable assignments
        """
        # Create an FSA with 2^|V| states (each state tracks seen variables)
        fsa = FiniteStateAutomaton(2 ** len(V), V_alphabet)
        P_V = self._powerset(V)
        v2idx = {v: i for i, v in enumerate(P_V)}

        # State empty set is initial state (no variables seen yet)
        fsa.set_initial_state(v2idx[()])

        # For each state (representing variables we've seen so far)
        for seen_vars in P_V:
            seen_vars_set = set(seen_vars)

            # For each possible input symbol
            for symbol, vars_at_pos in V_alphabet:
                vars_at_pos_set = set(vars_at_pos)

                # Check that vars_at_pos is disjoint from seen_vars (no variable appears twice)
                if len(seen_vars_set.intersection(vars_at_pos_set)) == 0:
                    # Calculate new set of seen variables
                    new_seen_vars = tuple(sorted(seen_vars_set.union(vars_at_pos_set)))
                    # Add transition
                    fsa.set_transition(
                        v2idx[seen_vars], (symbol, vars_at_pos), v2idx[new_seen_vars]
                    )

        # Only the state where all variables have been seen is accepting
        # This ensures every variable appears exactly once
        fsa.set_accepting_state(v2idx[tuple(sorted(V))])

        return fsa.trim().minimize()

    def _ensure_V_structure(self, fsa, V):
        """
        Ensures the FSA obeys the V-structure constraint by intersection.
        This is used after every operation to maintain the variable assignment invariant.

        Args:
            fsa: The FSA to constrain
            V: List of free variables

        Returns:
            An FSA that accepts only valid variable assignments
        """
        V = sorted(V)  # Sort the variables for deterministic order

        # Construct the enriched alphabet
        V_alphabet = list(product(self.alphabet, self._powerset(V)))

        # Construct the V-structure FSA
        V_structure_fsa = self._construct_V_structure_fsa(V, V_alphabet)

        # Intersect the structure FSA with the input FSA
        # This ensures we only keep runs with valid variable assignments
        return V_structure_fsa.intersect(fsa).trim().minimize()

    def _convert(self, formula, V, V_alphabet):
        """
        Internal recursive conversion method that builds automata compositionally.

        Args:
            formula: The formula to convert
            V: List of free variables
            V_alphabet: The enriched alphabet

        Returns:
            An FSA equivalent to the formula
        """
        if isinstance(formula, Predicate):
            return self._convert_predicate(formula, V_alphabet, V)
        elif isinstance(formula, Relation):
            return self._convert_relation(formula, V_alphabet, V)
        elif isinstance(formula, Negation):
            # For negation: complement the automaton and ensure V-structure
            return self._ensure_V_structure(
                self._convert(formula.subformula, V, V_alphabet).complement(), V
            )
        elif isinstance(formula, Conjunction):
            # For conjunction: intersect the automata of both subformulas
            left_fsa = self._convert(formula.left, V, V_alphabet)
            right_fsa = self._convert(formula.right, V, V_alphabet)
            intersection_fsa = left_fsa.intersect(right_fsa)
            return self._ensure_V_structure(intersection_fsa, V)
        elif isinstance(formula, ExistentialQuantifier):
            # For existential quantifier: add the new variable to the set
            _V = V + [formula.variable]
            _V_alphabet = list(product(self.alphabet, self._powerset(_V)))

            # Convert the subformula with the extended variable set
            subformula_fsa = self._convert(formula.subformula, _V, _V_alphabet)

            # Then remove the quantified variable to get the final automaton
            return self._remove_variable(subformula_fsa, V_alphabet, formula.variable)
        else:
            raise ValueError(f"Unsupported formula type: {type(formula).__name__}")

    def _convert_predicate(self, predicate, V_alphabet, V):
        """
        Convert a position predicate to an FSA.
        Handles symbol predicates like Qa(x) which test if position x has symbol a.

        Args:
            predicate: The predicate to convert
            V_alphabet: The enriched alphabet
            V: List of free variables

        Returns:
            An FSA accepting words where the predicate holds
        """
        # We only support symbol predicates (testing for specific alphabet symbols)
        if isinstance(predicate, SymbolPredicate):
            symbol = predicate.symbol  # Extract the symbol (e.g., "a" from "Qa")
            variable = predicate.variable  # Extract the variable (e.g., "x" from "Qa")

            # Check if the symbol is in our alphabet
            if symbol not in self.alphabet:
                raise ValueError(
                    f"Symbol '{symbol}' not found in alphabet: {self.alphabet}"
                )

            # Create an automaton with 2 states:
            # State 0: Initial state, waiting to see the symbol
            # State 1: Accepting state, saw the symbol
            fsa = FiniteStateAutomaton(2, V_alphabet)

            # Set initial state
            fsa.set_initial_state(0)

            # For each symbol-variable pair in the enriched alphabet
            for alph_symbol in fsa.alphabet:
                symbol_at_pos, vars_at_pos = alph_symbol

                if symbol_at_pos == symbol and variable in vars_at_pos:
                    # When we see the matching symbol with our variable, go to accepting state
                    fsa.set_transition(0, alph_symbol, 1)
                else:
                    # When we see any other symbol, stay in initial state
                    fsa.set_transition(0, alph_symbol, 0)

                # Once in accepting state, stay there regardless of symbol
                # This implements "we saw the symbol at the variable position at least once"
                fsa.set_transition(1, alph_symbol, 1)

            # Mark the accepting state
            fsa.set_accepting_state(1)

            # Ensure the FSA obeys variable assignment constraints
            return self._ensure_V_structure(fsa, V)
        else:
            raise ValueError(
                f"Unsupported predicate: {predicate.name}. Only symbol predicates are supported."
            )

    def _convert_relation(self, relation, V_alphabet, V):
        """
        Convert a numerical relation to an FSA following Straubing's construction.
        Currently only supports the < relation (strict ordering).

        Args:
            relation: The relation to convert
            V_alphabet: The enriched alphabet
            V: List of free variables

        Returns:
            An FSA accepting words where the relation holds
        """
        left, op, right = relation.left, relation.operator, relation.right

        # Less than: x < y
        if op == "<":
            return self._relation_lt(left, right, V_alphabet, V)
        else:
            raise ValueError(f"Unsupported relation: {left} {op} {right}")

    def _relation_lt(self, var1, var2, V_alphabet, V):
        """
        Builds an FSA for the relation x < y (var1 position is strictly less than var2 position).

        Args:
            var1: First variable
            var2: Second variable
            V_alphabet: The enriched alphabet
            V: List of free variables

        Returns:
            An FSA accepting words where var1 appears before var2
        """
        fsa = FiniteStateAutomaton(3, V_alphabet)

        # State 0: Initial state, haven't seen either position
        # State 1: Seen position var1, waiting for var2
        # State 2: Accepting state, seen both positions in correct order

        # For every symbol in the enriched alphabet
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

        # Ensure the FSA obeys variable assignment constraints
        return self._ensure_V_structure(fsa, V)

    def _remove_variable(self, fsa, V_alphabet, variable):
        """
        Remove a quantified variable by creating an NFA with 'seen' and 'unseen' states.
        This implements existential quantification through non-determinism.

        Args:
            fsa: The FSA for the subformula
            V_alphabet: The target alphabet (without the quantified variable)
            variable: The variable to remove

        Returns:
            An FSA equivalent to existential quantification over the variable
        """
        from automathic.fsa.nfa import NonDeterministicFSA

        # Create mapping from pairs (original_state, seen_bit) to new state IDs
        p2idx = {}  # (state_id, seen_bit) -> new_state_id
        idx = 0

        for state_id in range(fsa.num_states):
            if fsa.states[state_id] is not None:
                # Each state gets two copies - one for 'seen=0' and one for 'seen=1'
                # This tracks whether we've seen the quantified variable
                p2idx[(state_id, 0)] = idx
                idx += 1
                p2idx[(state_id, 1)] = idx
                idx += 1

        # Create a new NFA with double the number of states
        result = NonDeterministicFSA(idx, V_alphabet)

        # Copy state origins and set labels
        for (orig_id, seen_bit), new_id in p2idx.items():
            if fsa.states[orig_id] is not None:
                origin = fsa.states[orig_id].origin
                label = (
                    f"{fsa.states[orig_id].label}, seen={seen_bit}"
                    if fsa.states[orig_id].label
                    else None
                )
                result.states[new_id] = State(new_id, origin=origin, label=label)

        # Set initial state - the 'unseen' version of the original initial state
        if fsa.initial_state is not None:
            initial_id = p2idx[(fsa.initial_state.id, 0)]
            result.set_initial_state(initial_id)

        # Process transitions
        for (src_state, old_symbol), dst_state in fsa.transitions.items():
            symbol, vars_at_pos = old_symbol

            # Create new symbol without the projected variable
            new_vars = tuple(sorted([v for v in vars_at_pos if v != variable]))
            new_symbol = (symbol, new_vars)

            # Check if the variable is present in this transition
            var_present = variable in vars_at_pos

            # Get state IDs for both 'seen=0' and 'seen=1' versions
            unseen_src_id = p2idx[(src_state.id, 0)]
            seen_src_id = p2idx[(src_state.id, 1)]

            # The destination depends on whether the variable is present
            if var_present:
                # If the variable is present, the destination should be 'seen=1'
                unseen_dst_id = p2idx[(dst_state.id, 1)]  # unseen -> seen
                seen_dst_id = p2idx[(dst_state.id, 1)]  # seen -> seen
            else:
                # If the variable is not present, keep the 'seen' bit unchanged
                unseen_dst_id = p2idx[(dst_state.id, 0)]  # unseen -> unseen
                seen_dst_id = p2idx[(dst_state.id, 1)]  # seen -> seen

            # Add transitions
            result.set_transition(unseen_src_id, new_symbol, unseen_dst_id)
            result.set_transition(seen_src_id, new_symbol, seen_dst_id)

            if var_present:
                # Non-deterministic transition - can choose to make this the position
                # for the quantified variable
                result.set_transition(unseen_src_id, new_symbol, seen_dst_id)

        # Set accepting states - a state is accepting if it corresponds to an accepting
        # state in the original FSA and the quantified variable has been seen
        for state in fsa.accepting_states:
            accept_id = p2idx[(state.id, 1)]
            result.set_accepting_state(accept_id)

        # Convert back to a deterministic FSA and minimize
        return result.trim().determinize().minimize()


def convert_fo_to_fsa(formula, alphabet=None):
    """
    Convert a First-Order formula to an FSA following Straubing's construction.

    This is the main entry point function for formula-to-automaton conversion.

    Args:
        formula: An FOFormula instance or formula string
        alphabet: List of symbols in the alphabet. If None, defaults to the
            alphabet of symbols in the formula's predicates.

    Returns:
        A FiniteStateAutomaton equivalent to the formula
    """
    from automathic.fo.parser import parse_fo_formula

    # Parse string formulas
    if isinstance(formula, str):
        formula = parse_fo_formula(formula)

    # Extract alphabet from the formula if not provided
    derived_alphabet = formula.get_alphabet() if alphabet is None else alphabet

    # Convert to FSA
    converter = FOtoFSA(derived_alphabet)
    automaton = converter.convert(formula)

    return automaton
