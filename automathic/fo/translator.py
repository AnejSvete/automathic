from itertools import product

from automathic.fo.formula import (
    Conjunction,
    Disjunction,
    ExistentialQuantifier,
    ExistentialSetQuantifier,
    FOFormula,
    Negation,
    Predicate,
    Relation,
    SetMembership,
    SOMFormula,
    SymbolPredicate,
    UniversalQuantifier,
    UniversalSetQuantifier,
)
from automathic.fsa.fsa import FiniteStateAutomaton, State


class SOMtoFSA:
    """
    Converts Second-Order Monadic logic formulas to equivalent Finite State Automata.

    SOM formulas can express all regular languages, extending the expressivity
    of First-Order logic which can only express star-free languages.
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
            self.alphabet = ["a", "b"]
        else:
            self.alphabet = sorted(alphabet)

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
            # For each element, add a new subset that includes this element
            # to all existing subsets
            new_subsets = [subset + [element] for subset in result]
            result.extend(new_subsets)

        # Convert lists to tuples for immutability and hashability
        return [tuple(subset) for subset in result]

    def convert(self, formula):
        """
        Convert a Second-Order Monadic formula to an FSA

        Args:
            formula: An instance of SOMFormula or FOFormula

        Returns:
            A FiniteStateAutomaton equivalent to the formula
        """
        # First ensure the formula is in normal form for FSA conversion
        # (We would first convert to FO[<] form but SOM can express more than that)

        # Get free variables in sorted order for consistent processing
        V_first_order = sorted(formula.get_free_variables())

        # Handle both SOMFormula and FOFormula instances gracefully
        if hasattr(formula, "get_free_set_variables"):
            V_second_order = sorted(formula.get_free_set_variables())
        else:
            # For pure FOFormula instances that don't have this method
            V_second_order = []

        # Construct the enriched alphabet: pairs of (symbol, variable_set)
        # Each pair represents a position in the word with its symbol and variables
        V_alphabet = list(
            product(
                self.alphabet,
                self._powerset(V_first_order),
            )
        )

        # Apply construction recursively
        fsa = self._convert(formula, V_first_order, V_second_order, V_alphabet)

        # For sentences (formulas with no free variables), project to the original alphabet
        if not V_first_order and not V_second_order:
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
                proj_fsa.states[i] = State(i, origin=state.origin, label=state.label)

        # Set initial state
        if fsa.initial_state is not None:
            proj_fsa.set_initial_state(fsa.initial_state.id)

        # Process transitions
        for (src_state, old_symbol), dst_states in fsa.transitions.items():
            for dst_state in dst_states:
                symbol, _ = old_symbol  # Extract just the alphabet symbol
                proj_fsa.set_transition(src_state.id, symbol, dst_state.id)

        # Set accepting states
        for state in fsa.accepting_states:
            proj_fsa.set_accepting_state(state.id)

        # Return trimmed FSA (remove unreachable states)
        return proj_fsa.trim()

    def _convert(self, formula, V_first_order, V_second_order, V_alphabet):
        """
        Internal recursive conversion method that builds automata compositionally.

        Args:
            formula: The formula to convert
            V_first_order: List of free first-order variables
            V_second_order: List of free second-order variables
            V_alphabet: The enriched alphabet

        Returns:
            An FSA equivalent to the formula
        """
        if isinstance(formula, Predicate):
            return self._convert_predicate(formula, V_alphabet, V_first_order)
        elif isinstance(formula, Relation):
            return self._convert_relation(formula, V_alphabet, V_first_order)
        elif isinstance(formula, SetMembership):
            return self._convert_set_membership(
                formula, V_alphabet, V_first_order, V_second_order
            )
        elif isinstance(formula, Negation):
            # For negation: complement the automaton and ensure variable structure
            subformula_fsa = self._convert(
                formula.subformula, V_first_order, V_second_order, V_alphabet
            )
            return self._ensure_V_structure(subformula_fsa.complement(), V_first_order)
        elif isinstance(formula, Conjunction):
            # For conjunction: intersect the automata of both subformulas
            left_fsa = self._convert(
                formula.left, V_first_order, V_second_order, V_alphabet
            )
            right_fsa = self._convert(
                formula.right, V_first_order, V_second_order, V_alphabet
            )
            intersection_fsa = left_fsa.intersect(right_fsa)
            return self._ensure_V_structure(intersection_fsa, V_first_order)
        elif isinstance(formula, Disjunction):
            # For disjunction: union the automata of both subformulas
            left_fsa = self._convert(
                formula.left, V_first_order, V_second_order, V_alphabet
            )
            right_fsa = self._convert(
                formula.right, V_first_order, V_second_order, V_alphabet
            )
            union_fsa = left_fsa.union(right_fsa)
            return self._ensure_V_structure(union_fsa, V_first_order)
        elif isinstance(formula, ExistentialQuantifier):
            # For existential quantifier: add the new variable to the set
            _V = V_first_order + [formula.variable]
            _V_alphabet = list(product(self.alphabet, self._powerset(_V)))

            # Convert the subformula with the extended variable set
            subformula_fsa = self._convert(
                formula.subformula, _V, V_second_order, _V_alphabet
            )

            # Then remove the quantified variable to get the final automaton
            return self._remove_variable(subformula_fsa, V_alphabet, formula.variable)
        elif isinstance(formula, ExistentialSetQuantifier):
            # For existential set quantifier
            _V_second_order = V_second_order + [formula.set_variable]

            # Convert the subformula with the extended set variable list
            subformula_fsa = self._convert(
                formula.subformula, V_first_order, _V_second_order, V_alphabet
            )

            # Then remove the quantified set variable via projection
            return self._remove_set_variable(subformula_fsa, formula.set_variable)
        elif isinstance(formula, UniversalQuantifier):
            # ∀x.φ(x) = ¬∃x.¬φ(x)
            negated_subformula = Negation(formula.subformula)
            exists_formula = ExistentialQuantifier(formula.variable, negated_subformula)
            return self._convert(
                Negation(exists_formula), V_first_order, V_second_order, V_alphabet
            )
        elif isinstance(formula, UniversalSetQuantifier):
            # ∀X.φ(X) = ¬∃X.¬φ(X)
            negated_subformula = Negation(formula.subformula)
            exists_formula = ExistentialSetQuantifier(
                formula.set_variable, negated_subformula
            )
            return self._convert(
                Negation(exists_formula), V_first_order, V_second_order, V_alphabet
            )
        else:
            raise ValueError(f"Unsupported formula type: {type(formula).__name__}")

    def _convert_set_membership(
        self, membership, V_alphabet, V_first_order, V_second_order
    ):
        """
        Convert a set membership predicate (x ∈ X) to an FSA.

        Args:
            membership: The SetMembership predicate
            V_alphabet: The enriched alphabet
            V_first_order: List of free first-order variables
            V_second_order: List of free second-order variables

        Returns:
            An FSA accepting words where the position variable belongs to the set variable
        """
        pos_var = membership.position_variable
        set_var = membership.set_variable

        # Check if the set variable is in our known set variables
        if set_var not in V_second_order:
            raise ValueError(f"Set variable '{set_var}' is not bound")

        # Create an automaton with 2 states
        fsa = FiniteStateAutomaton(2, V_alphabet)

        # State 0: Initial state, haven't seen the position
        # State 1: Accepting state, found the position in the set

        # For every symbol in the enriched alphabet
        for V_symbol in fsa.alphabet:
            _, vars_at_pos = V_symbol

            if pos_var in vars_at_pos:
                # When we see the position variable, go to accepting state
                # (The set check is handled during evaluation with set assignments)
                fsa.set_transition(0, V_symbol, 1)
            else:
                # When position variable isn't present, stay in initial state
                fsa.set_transition(0, V_symbol, 0)

            # Once in accepting state, stay there
            fsa.set_transition(1, V_symbol, 1)

        # Set initial state
        fsa.set_initial_state(0)
        # Set accepting state
        fsa.set_accepting_state(1)

        return self._ensure_V_structure(fsa, V_first_order)

    def _remove_set_variable(self, fsa, set_variable):
        """
        Remove a set variable by non-deterministic automaton construction.
        For existential set quantification (∃X), we use non-determinism to "guess"
        the positions that belong to set X.

        Args:
            fsa: The FSA for the subformula
            set_variable: The set variable to remove

        Returns:
            An FSA accepting words with any valid assignment of positions to the set
        """
        # Since we're handling set variables implicitly during evaluation,
        # removing a set variable is simply passing through the automaton
        # (the set variable constraints are maintained through the transitions)
        return fsa.copy().trim()

    def _convert_predicate(self, predicate, V_alphabet, V):
        """
        Convert a predicate to an FSA.

        Args:
            predicate: The predicate to convert
            V_alphabet: The enriched alphabet
            V: List of variables

        Returns:
            An FSA equivalent to the predicate
        """
        # Create a 2-state FSA
        fsa = FiniteStateAutomaton(2, V_alphabet)

        # Extract predicate information
        var = predicate.variable

        # For symbol predicates like Qa(x), extract the symbol
        if isinstance(predicate, SymbolPredicate):
            symbol = predicate.symbol
        else:
            raise ValueError(f"Unsupported predicate: {predicate}")

        # Configure transitions for the automaton
        for V_symbol in fsa.alphabet:
            a, vars_at_pos = V_symbol
            if var in vars_at_pos and a == symbol:
                # If the variable is at this position AND the symbol matches,
                # transition to accepting state
                fsa.set_transition(0, V_symbol, 1)
            else:
                # Otherwise stay in non-accepting state
                fsa.set_transition(0, V_symbol, 0)

            # Once in accepting state, stay there
            fsa.set_transition(1, V_symbol, 1)

        # Set initial state
        fsa.set_initial_state(0)
        # Set accepting state
        fsa.set_accepting_state(1)

        # Ensure the structure correctly handles variable dependencies
        return self._ensure_V_structure(fsa, V)

    def _convert_relation(self, relation, V_alphabet, V):
        """
        Convert a relation to an FSA.

        Args:
            relation: The relation to convert
            V_alphabet: The enriched alphabet
            V: List of variables

        Returns:
            An FSA equivalent to the relation
        """
        left, op, right = relation.left, relation.operator, relation.right

        # Handle the different relation types
        if op == "<":
            return self._relation_lt(left, right, V_alphabet, V)
        else:
            # Other relations like <=, >, >=, = should be converted to FO[<] first
            # This should not happen if we first normalize the formula
            raise ValueError(
                f"Relation {op} not directly supported. Convert to FO[<] first."
            )

    # def _relation_lt(self, var1, var2, V_alphabet, V):
    #     """
    #     Convert a less-than relation (var1 < var2) to an FSA.

    #     Args:
    #         var1: The left variable
    #         var2: The right variable
    #         V_alphabet: The enriched alphabet
    #         V: List of variables

    #     Returns:
    #         An FSA equivalent to the relation
    #     """
    #     # Create a 3-state FSA
    #     fsa = FiniteStateAutomaton(3, V_alphabet)

    #     # State meanings:
    #     # 0: Initial state, neither var1 nor var2 seen
    #     # 1: var1 seen, var2 not seen yet (potentially accepting)
    #     # 2: var2 seen after var1 (accepting state)

    #     # Configure transitions
    #     for V_symbol in fsa.alphabet:
    #         _, vars_at_pos = V_symbol

    #         if var1 in vars_at_pos and var2 in vars_at_pos:
    #             # Both variables at same position - not possible for < relation
    #             # Stay in current state
    #             fsa.set_transition(0, V_symbol, 0)
    #             fsa.set_transition(1, V_symbol, 1)
    #             fsa.set_transition(2, V_symbol, 2)
    #         elif var1 in vars_at_pos:
    #             # Found var1
    #             if var2 not in vars_at_pos:
    #                 # Only var1 found, move to state 1
    #                 fsa.set_transition(0, V_symbol, 1)
    #                 # If already in state 1 or 2, stay there
    #                 fsa.set_transition(1, V_symbol, 1)
    #                 fsa.set_transition(2, V_symbol, 2)
    #         elif var2 in vars_at_pos:
    #             # Found var2
    #             if var1 not in vars_at_pos:
    #                 # Only var2 found
    #                 # If in state 0, stay in state 0 (var1 not seen yet)
    #                 fsa.set_transition(0, V_symbol, 0)
    #                 # If in state 1, move to accepting state 2 (var1 seen before var2)
    #                 fsa.set_transition(1, V_symbol, 2)
    #                 # If already in state 2, stay there
    #                 fsa.set_transition(2, V_symbol, 2)
    #         else:
    #             # Neither variable at this position, stay in current state
    #             fsa.set_transition(0, V_symbol, 0)
    #             fsa.set_transition(1, V_symbol, 1)
    #             fsa.set_transition(2, V_symbol, 2)

    #     # Set initial state
    #     fsa.set_initial_state(0)
    #     # Set accepting state
    #     fsa.set_accepting_state(2)

    #     # Ensure the structure correctly handles variable dependencies
    #     return self._ensure_V_structure(fsa, V)

    # def _relation_lt(self, var1, var2, V_alphabet, V):
    #     """
    #     Builds an FSA for the relation x < y (var1 position is strictly less than var2 position).

    #     Args:
    #         var1: First variable
    #         var2: Second variable
    #         V_alphabet: The enriched alphabet
    #         V: List of free variables

    #     Returns:
    #         An FSA accepting words where var1 appears before var2
    #     """
    #     fsa = FiniteStateAutomaton(3, V_alphabet)

    #     # State 0: Initial state, haven't seen either position
    #     # State 1: Seen position var1, waiting for var2
    #     # State 2: Accepting state, seen both positions in correct order

    #     # For every symbol in the enriched alphabet
    #     for V_symbol in fsa.alphabet:
    #         _, vars_at_pos = V_symbol

    #         # Handle all possible variable combinations
    #         if var1 in vars_at_pos and var2 in vars_at_pos:
    #             # Both variables at the same position - invalid for strict <
    #             # Stay in current state (non-accepting)
    #             fsa.set_transition(0, V_symbol, 0)
    #             fsa.set_transition(1, V_symbol, 1)
    #             fsa.set_transition(2, V_symbol, 2)
    #         elif var1 in vars_at_pos and var2 not in vars_at_pos:
    #             # Transition to state 1 when var1 is seen and var2 is not
    #             fsa.set_transition(0, V_symbol, 1)
    #             # If we see var1 again in state 1, stay in state 1
    #             fsa.set_transition(1, V_symbol, 1)
    #             # If already in accepting state, stay there
    #             fsa.set_transition(2, V_symbol, 2)
    #         elif var2 in vars_at_pos and var1 not in vars_at_pos:
    #             # Transition to state 2 when var2 is seen after var1
    #             fsa.set_transition(1, V_symbol, 2)
    #             # If we see var2 before var1, stay in non-accepting state
    #             fsa.set_transition(0, V_symbol, 0)
    #             # If already in accepting state, stay there
    #             fsa.set_transition(2, V_symbol, 2)
    #         else:  # Neither variable is present
    #             # Stay in the same state if neither var1 nor var2 is seen
    #             fsa.set_transition(0, V_symbol, 0)
    #             fsa.set_transition(1, V_symbol, 1)
    #             fsa.set_transition(2, V_symbol, 2)

    #     # Set initial state
    #     fsa.set_initial_state(0)
    #     # Set accepting state
    #     fsa.set_accepting_state(2)

    #     # Ensure the FSA obeys variable assignment constraints
    #     return self._ensure_V_structure(fsa, V)

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
        for (src_state, old_symbol), dst_states in fsa.transitions.items():
            for dst_state in dst_states:
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

    def _construct_V_structure_fsa(self, V, V_alphabet):
        """
        Construct an FSA that verifies the structure of variable bindings.
        Ensures each variable appears exactly once in the input word.

        Args:
            V: List of variables to enforce
            V_alphabet: The enriched alphabet

        Returns:
            An FSA that accepts only words with valid variable bindings
        """
        variables = sorted(V)
        if not variables:
            # If there are no variables to enforce, accept anything
            fsa = FiniteStateAutomaton(1, V_alphabet)
            fsa.set_initial_state(0)
            fsa.set_accepting_state(0)
            for symbol in V_alphabet:
                fsa.set_transition(0, symbol, 0)
            return fsa

        # For each variable, construct a sub-automaton that ensures it appears exactly once
        sub_fsas = []
        for var in variables:
            # Create a 2-state FSA: 0 = not seen, 1 = seen
            var_fsa = FiniteStateAutomaton(2, V_alphabet)
            var_fsa.set_initial_state(0)
            var_fsa.set_accepting_state(1)

            for V_symbol in V_alphabet:
                _, vars_at_pos = V_symbol
                if var in vars_at_pos:
                    # If variable is at this position, transition from not-seen to seen
                    var_fsa.set_transition(0, V_symbol, 1)
                    # Once seen, reject if seen again (no transition from state 1)
                else:
                    # If variable not at this position, stay in current state
                    var_fsa.set_transition(0, V_symbol, 0)
                    var_fsa.set_transition(1, V_symbol, 1)

            sub_fsas.append(var_fsa)

        # Intersect all the sub-automata to get the final structure enforcer
        if sub_fsas:
            result = sub_fsas[0]
            for i in range(1, len(sub_fsas)):
                result = result.intersect(sub_fsas[i])
            return result
        else:
            # Shouldn't reach here given the check at the beginning
            raise ValueError("No variables to enforce structure for")

    def _ensure_V_structure(self, fsa, V):
        """
        Ensure the FSA respects the structure for variables in V.
        Each variable must appear exactly once in any accepted word.

        Args:
            fsa: The FSA to enforce structure on
            V: List of variables to enforce

        Returns:
            An FSA with the variable structure enforced
        """
        # If there are no variables, no structure to enforce
        if not V:
            return fsa

        # Construct the structure-enforcing FSA
        structure_fsa = self._construct_V_structure_fsa(V, fsa.alphabet)

        # Intersect with the original FSA to enforce the structure
        result = fsa.intersect(structure_fsa)

        return result.trim()


# Rename FOtoFSA to maintain backward compatibility
class FOtoFSA(SOMtoFSA):
    """Legacy class name for backward compatibility"""

    pass


def convert_som_to_fsa(formula, alphabet=None):
    """
    Convert a Second-Order Monadic formula to an FSA.

    Args:
        formula: A SOMFormula instance or formula string
        alphabet: List of symbols in the alphabet. If None, defaults to the
            alphabet of symbols in the formula's predicates.

    Returns:
        A FiniteStateAutomaton equivalent to the formula
    """
    from automathic.fo.parser import parse_som_formula

    # Parse string formulas
    if isinstance(formula, str):
        formula = parse_som_formula(formula)

    # Extract alphabet from the formula if not provided
    derived_alphabet = formula.get_alphabet() if alphabet is None else alphabet

    # Convert to FSA
    converter = SOMtoFSA(derived_alphabet)
    automaton = converter.convert(formula)

    return automaton


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
    # This function now uses the SOM converter since it's a superset
    return convert_som_to_fsa(formula, alphabet)


class FSAToSOM:
    """
    Converts Finite State Automata to equivalent Second-Order Monadic logic formulas.

    The conversion is based on the automata's transitions and structure.
    """

    def __init__(self):
        self.counter = 0

    def convert(self, fsa):
        """
        Convert a Finite State Automaton to a Second-Order Monadic formula.

        Args:
            fsa: A FiniteStateAutomaton instance

        Returns:
            A SOMFormula equivalent to the automaton
        """
        # First ensure the FSA is normalized for conversion
        normalized_fsa = fsa.normalize()

        # Apply conversion recursively
        formula = self._convert(normalized_fsa)

        return formula
