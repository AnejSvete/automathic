from itertools import product

from automathic.fsa.fsa import FiniteStateAutomaton, State
from automathic.som.formula import (
    Conjunction,
    Disjunction,
    ExistentialQuantifier,
    ExistentialSetQuantifier,
    FOFormula,
    Implication,
    Negation,
    Predicate,
    Relation,
    SetMembership,
    SOMFormula,
    SymbolPredicate,
    UniversalQuantifier,
    UniversalSetQuantifier,
)


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
        V1 = sorted(formula.get_free_variables())

        # Handle both SOMFormula and FOFormula instances gracefully
        if hasattr(formula, "get_free_set_variables"):
            V2 = sorted(formula.get_free_set_variables())
        else:
            # For pure FOFormula instances that don't have this method
            V2 = []

        # Construct the enriched alphabet: pairs of (symbol, variable_set)
        # Each pair represents a position in the word with its symbol and variables
        V_alphabet = list(
            product(
                self.alphabet,
                self._powerset(V1),
                self._powerset(V2),
            )
        )

        # Apply construction recursively
        fsa = self._convert(formula, V1, V2, V_alphabet)

        # For sentences (formulas with no free variables), project to the original alphabet
        if not V1 and not V2:
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
                symbol, _, _ = old_symbol  # Extract just the alphabet symbol
                proj_fsa.set_transition(src_state.id, symbol, dst_state.id)

        # Set accepting states
        for state in fsa.accepting_states:
            proj_fsa.set_accepting_state(state.id)

        # Return trimmed FSA (remove unreachable states)
        return proj_fsa.trim()

    def _convert(self, formula, V1, V2, V_alphabet):
        """
        Internal recursive conversion method that builds automata compositionally.

        Args:
            formula: The formula to convert
            V1: List of free first-order variables
            V2: List of free second-order variables
            V_alphabet: The enriched alphabet

        Returns:
            An FSA equivalent to the formula
        """
        if isinstance(formula, Predicate):
            return self._convert_predicate(formula, V_alphabet, V1)
        elif isinstance(formula, Relation):
            return self._convert_relation(formula, V_alphabet, V1)
        elif isinstance(formula, SetMembership):
            return self._convert_set_membership(formula, V_alphabet, V1, V2)
        elif isinstance(formula, Negation):
            # For negation: complement the automaton and ensure variable structure
            subformula_fsa = self._convert(formula.subformula, V1, V2, V_alphabet)
            return self._ensure_V_structure(subformula_fsa.complement(), V1)
        elif isinstance(formula, Conjunction):
            # For conjunction: intersect the automata of both subformulas
            left_fsa = self._convert(formula.left, V1, V2, V_alphabet)
            right_fsa = self._convert(formula.right, V1, V2, V_alphabet)
            intersection_fsa = left_fsa.intersect(right_fsa)
            return self._ensure_V_structure(intersection_fsa, V1)
        elif isinstance(formula, Disjunction):
            # For disjunction: union the automata of both subformulas
            left_fsa = self._convert(formula.left, V1, V2, V_alphabet)
            right_fsa = self._convert(formula.right, V1, V2, V_alphabet)
            union_fsa = left_fsa.union(right_fsa)
            return self._ensure_V_structure(union_fsa, V1)
        elif isinstance(formula, ExistentialQuantifier):
            # For existential quantifier: add the new variable to the set
            _V1 = V1 + [formula.variable]
            _V_alphabet = list(
                product(self.alphabet, self._powerset(_V1), self._powerset(V2))
            )

            # Convert the subformula with the extended variable set
            subformula_fsa = self._convert(formula.subformula, _V1, V2, _V_alphabet)

            # Then remove the quantified variable to get the final automaton
            return self._remove_variable(subformula_fsa, V_alphabet, formula.variable)
        elif isinstance(formula, ExistentialSetQuantifier):
            # For existential set quantifier
            _V2 = V2 + [formula.set_variable]
            _V_alphabet = list(
                product(self.alphabet, self._powerset(V1), self._powerset(_V2))
            )

            # Convert the subformula with the extended set variable list
            subformula_fsa = self._convert(formula.subformula, V1, _V2, _V_alphabet)

            # Then remove the quantified set variable via projection
            return self._remove_set_variable(
                subformula_fsa, V_alphabet, formula.set_variable
            )
        elif isinstance(formula, UniversalQuantifier):
            # ∀x.φ(x) = ¬∃x.¬φ(x)
            negated_subformula = Negation(formula.subformula)
            exists_formula = ExistentialQuantifier(formula.variable, negated_subformula)
            return self._convert(Negation(exists_formula), V1, V2, V_alphabet)
        elif isinstance(formula, UniversalSetQuantifier):
            # ∀X.φ(X) = ¬∃X.¬φ(X)
            negated_subformula = Negation(formula.subformula)
            exists_formula = ExistentialSetQuantifier(
                formula.set_variable, negated_subformula
            )
            return self._convert(Negation(exists_formula), V1, V2, V_alphabet)
        else:
            raise ValueError(f"Unsupported formula type: {type(formula).__name__}")

    def _convert_set_membership(self, membership, V_alphabet, V1, V2):
        """
        Convert a set membership predicate (x ∈ X) to an FSA.

        Args:
            membership: The SetMembership predicate
            V_alphabet: The enriched alphabet
            V1: List of free first-order variables
            V2: List of free second-order variables

        Returns:
            An FSA accepting words where the position variable belongs to the set variable
        """

        # Check if the set variable is in our known set variables
        if membership.set_variable not in V2:
            raise ValueError(f"Set variable '{membership}' is not bound")

        # Create a 2-state FSA: 0 = not seen, 1 = seen
        fsa = FiniteStateAutomaton(2, V_alphabet)
        fsa.set_initial_state(0)
        fsa.set_accepting_state(1)

        for V_symbol in V_alphabet:
            symbol, vars_at_pos, set_vars_at_pos = V_symbol
            if (
                membership.position_variable in vars_at_pos
                and membership.set_variable in set_vars_at_pos
            ):
                # If variable is at this position, transition from not-seen to seen
                fsa.set_transition(0, V_symbol, 1)
                # Once seen, reject if seen again (no transition from state 1)
            else:
                # If variable not at this position, stay in current state
                fsa.set_transition(0, V_symbol, 0)
                fsa.set_transition(1, V_symbol, 1)

        return self._ensure_V_structure(fsa, V1, V2)

    def _remove_set_variable(self, fsa, V_alphabet, set_variable):
        """
        Remove a set variable by non-deterministic automaton construction.
        For existential set quantification (∃X), we use non-determinism to "guess"
        the positions that belong to set X.

        Args:
            fsa: The FSA for the subformula
            V_alphabet: The target alphabet (without the quantified variable)
            set_variable: The set variable to remove

        Returns:
            An FSA accepting words with any valid assignment of positions to the set
        """
        from automathic.fsa.nfa import NonDeterministicFSA

        # Create a new NFA with double the number of states
        result = NonDeterministicFSA(fsa.num_states, V_alphabet)

        # Set initial state
        if fsa.initial_state is not None:
            result.set_initial_state(fsa.initial_state)

        # Process transitions
        for (src_state, old_symbol), dst_states in fsa.transitions.items():
            for dst_state in dst_states:
                symbol, vars_at_pos, set_vars_at_pos = old_symbol

                # Create new symbol without the projected variable
                new_set_vars = tuple(
                    sorted([v for v in set_vars_at_pos if v != set_variable])
                )
                new_symbol = (symbol, vars_at_pos, new_set_vars)

                # Add transitions
                result.set_transition(src_state, new_symbol, dst_state)

        # Set accepting states
        for state in fsa.accepting_states:
            result.set_accepting_state(state)

        # Convert back to a deterministic FSA and minimize
        return result.trim().determinize().minimize()

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
            a, vars_at_pos, set_vars_at_pos = V_symbol
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
        elif op == "=":
            return self._relation_eq(left, right, V_alphabet, V)
        else:
            # Other relations like <=, >, >=, = should be converted to FO[<] first
            # This should not happen if we first normalize the formula
            raise ValueError(
                f"Relation {op} not directly supported. Convert to FO[<] first."
            )

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
            _, vars_at_pos, set_vars_at_pos = V_symbol

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

    def _relation_eq(self, var1, var2, V_alphabet, V):
        """
        Builds an FSA for the relation x = y (var1 position the same as var2 position).

        Args:
            var1: First variable
            var2: Second variable
            V_alphabet: The enriched alphabet
            V: List of free variables

        Returns:
            An FSA accepting words where var1 and var2 appear together
        """
        fsa = FiniteStateAutomaton(2, V_alphabet)

        # State 0: Initial state, haven't seen either position
        # State 1: Seen position var1 and var2

        # For every symbol in the enriched alphabet
        for V_symbol in fsa.alphabet:
            _, vars_at_pos, set_vars_at_pos = V_symbol

            if var1 in vars_at_pos and var2 in vars_at_pos:
                # Transition to state 1 when var1 and var2 are seen
                fsa.set_transition(0, V_symbol, 1)
            else:
                # Stay in the same state
                fsa.set_transition(0, V_symbol, 0)
                fsa.set_transition(1, V_symbol, 1)

        # Set initial state
        fsa.set_initial_state(0)
        # Set accepting state
        fsa.set_accepting_state(1)

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
                symbol, vars_at_pos, set_vars_at_pos = old_symbol

                # Create new symbol without the projected variable
                new_vars = tuple(sorted([v for v in vars_at_pos if v != variable]))
                new_symbol = (symbol, new_vars, set_vars_at_pos)

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

    def _construct_V_structure_fsa(self, V1, V2, V_alphabet):
        """
        Construct an FSA that ensures each first-order variable appears exactly once.

        Args:
            V1: List of first-order variables
            V2: List of second-order variables
            V_alphabet: The enriched alphabet

        Returns:
            An FSA that enforces first-order variable constraints
        """
        # For each variable, construct a sub-automaton that ensures it appears exactly once
        sub_fsas = []
        for var in V1:
            # Create a 2-state FSA: 0 = not seen, 1 = seen
            var_fsa = FiniteStateAutomaton(2, V_alphabet)
            var_fsa.set_initial_state(0)
            var_fsa.set_accepting_state(1)

            for V_symbol in V_alphabet:
                symbol, vars_at_pos, set_vars_at_pos = V_symbol
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
            # If no first-order variables, create an automaton that accepts anything
            fsa = FiniteStateAutomaton(1, V_alphabet)
            fsa.set_initial_state(0)
            fsa.set_accepting_state(0)
            for symbol in V_alphabet:
                fsa.set_transition(0, symbol, 0)
            return fsa

    def _ensure_V_structure(self, fsa, V1, V2=None):
        """
        Ensure the FSA respects the structure for variables in V.
        Each first-order variable must appear exactly once in any accepted word,
        and second-order variables must satisfy membership constraints.

        Args:
            fsa: The FSA to enforce structure on
            V1: The list of free first-order variables
            V2: The list of free second-order variables (default: None)

        Returns:
            An FSA with the variable structure enforced
        """
        # If V2 is not provided, default to empty list
        if V2 is None:
            V2 = []

        # If there are no variables, no structure to enforce
        if not V1 and not V2:
            return fsa

        # Construct the structure-enforcing FSA
        structure_fsa = self._construct_V_structure_fsa(V1, V2, fsa.alphabet)

        # Intersect with the original FSA to enforce the structure
        result = fsa.intersect(structure_fsa)

        return result.trim()


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
    from automathic.som.parser import parse_som_formula

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
        fsa = fsa.trim().minimize()

        # Check if the automaton accepts the empty string
        # (initial state is also an accepting state)
        accepts_empty = (
            fsa.initial_state is not None and fsa.initial_state in fsa.accepting_states
        )

        # Create set variable quantifiers for each state
        state_vars = [f"X{state.id}" for state in fsa.states if state is not None]

        # Build the core formula components
        φ_1 = self._construct_φ_1(fsa)  # Every position belongs to exactly one state
        φ_2 = self._construct_φ_2(fsa)  # States are mutually exclusive
        φ_3 = self._construct_φ_3(fsa)  # Initial state constraint
        φ_4 = self._construct_φ_4(fsa)  # Transition constraints
        φ_5 = self._construct_φ_5(fsa)  # Accepting state constraint

        # Combine all constraints with conjunction
        formula_body = Conjunction(
            Conjunction(Conjunction(φ_1, φ_2), Conjunction(φ_3, φ_4)), φ_5
        )
        # formula_body = Conjunction(Conjunction(φ_1, Conjunction(φ_3, φ_4)), φ_5)

        # Create a formula that's true for the empty string
        # The formula ∀x.(x ≠ x) is only true for the empty string
        empty_string_formula = UniversalQuantifier(
            "x0", Negation(Relation("x0", "=", "x0"))
        )
        nonepty_string_formula = ExistentialQuantifier("x0", Relation("x0", "=", "x0"))

        # If the automaton accepts the empty string, we should include it in our formula
        if accepts_empty:
            # The formula is: (empty string formula) OR (non-empty string formula)
            formula_body = Disjunction(empty_string_formula, formula_body)
        else:
            # If the automaton doesn't accept the empty string, we only need the non-empty formula
            formula_body = Conjunction(nonepty_string_formula, formula_body)

        # Wrap with existential quantifiers for all state sets
        result = formula_body
        for state_var in state_vars:
            result = ExistentialSetQuantifier(state_var, result)

        return result

    def _construct_φ_1(self, fsa):
        """
        Construct the formula that ensures every position belongs to exactly one state.
        This is achieved by stating that for each position, it belongs to at least one state.

        Combined with φ_2 (mutual exclusion), this ensures exactly one state per position.
        """
        # Get all valid states in the FSA
        valid_states = [state for state in fsa.states if state is not None]

        # Handle empty automaton or no states
        if not valid_states:
            # No states means the language is empty
            # Return a contradiction formula
            return Conjunction(
                SetMembership("x1", "X0"), Negation(SetMembership("x1", "X0"))
            )

        # Create the set membership predicates for each state
        state_sets = [SetMembership("x1", f"X{state.id}") for state in valid_states]

        # If there's only one state, the disjunction is just that state
        if len(state_sets) == 1:
            disjunction = state_sets[0]
        else:
            # Create a disjunction of all state memberships
            disjunction = Disjunction(state_sets[0], state_sets[1])
            for i in range(2, len(state_sets)):
                disjunction = Disjunction(disjunction, state_sets[i])

        # For all positions x1, x1 belongs to at least one state
        return UniversalQuantifier("x1", disjunction)

    def _construct_φ_2(self, fsa):
        """
        Construct the formula that ensures states are mutually exclusive.
        This means that for each position, it belongs to at most one state.

        Combined with φ_1, this ensures exactly one state per position.
        """
        # Get all valid states in the FSA
        valid_states = [state for state in fsa.states if state is not None]

        # If no states or only one state, no need for mutual exclusion
        if len(valid_states) <= 1:
            # Return a tautology formula - always true
            return UniversalQuantifier("x2", Relation("x2", "=", "x2"))

        # Create the set membership predicates for each state
        state_sets = [SetMembership("x2", f"X{state.id}") for state in valid_states]

        # Build a conjunction of all pairwise exclusions
        exclusions = []
        for i in range(0, len(state_sets)):
            for j in range(i + 1, len(state_sets)):
                # A position cannot be in both state i and state j
                exclusions.append(Negation(Conjunction(state_sets[i], state_sets[j])))

        # Create the full conjunction of all exclusions
        result = exclusions[0]
        for i in range(1, len(exclusions)):
            result = Conjunction(result, exclusions[i])

        # For all positions x2, mutual exclusion of states holds
        return UniversalQuantifier("x2", result)

    def _construct_φ_3(self, fsa):
        """
        Construct the formula that ensures the initial position belongs to the initial state.
        """
        if fsa.initial_state is None:
            # If no initial state, return a contradiction
            return Conjunction(
                SetMembership("x3", "X0"), Negation(SetMembership("x3", "X0"))
            )

        # Formula: first position is in initial state
        # First position is the one with no predecessor
        first_position = UniversalQuantifier(
            "x4",
            Relation("x4", ">=", "x3"),
        )

        return UniversalQuantifier(
            "x3",
            Implication(
                first_position, SetMembership("x3", f"X{fsa.initial_state.id}")
            ),
        )

    def _construct_φ_4(self, fsa):
        """
        Construct the formula that ensures proper transitions between states.
        This formula properly constrains both allowed and forbidden transitions.
        """

        pair_formulas = []
        # Process each pair of states in the FSA
        for i in range(0, len(fsa.states)):
            for j in range(0, len(fsa.states)):
                # Build a formula for this state that covers all possible symbols

                transitions = fsa.get_transitions_between(i, j)

                # Symbol predicate for this position
                if transitions:
                    # Create a disjunction of all valid next states
                    symbol_preds = []
                    for symbol in transitions:
                        symbol_preds.append(SymbolPredicate("Q", "x5", symbol))

                    # If only one next state, use it directly
                    if len(symbol_preds) == 1:
                        symbol_pred = symbol_preds[0]
                    else:
                        # Create a disjunction of all next states
                        symbol_pred = Disjunction(symbol_preds[0], symbol_preds[1])
                        for k in range(2, len(symbol_preds)):
                            symbol_pred = Disjunction(symbol_pred, symbol_preds[k])

                    # For this symbol: if y has symbol, then y must be in one of valid next states
                    pair_formulas.append(
                        Implication(
                            Conjunction(
                                SetMembership("x5", f"X{i}"),
                                SetMembership("x6", f"X{j}"),
                            ),
                            symbol_pred,
                        )
                    )
                else:
                    # For symbols with no valid transitions, they cannot appear next
                    # Add a contradiction for this state
                    pair_formulas.append(
                        Implication(
                            Conjunction(
                                SetMembership("x5", f"X{i}"),
                                SetMembership("x6", f"X{j}"),
                            ),
                            Conjunction(
                                Negation(SetMembership("x5", f"X{j}")),
                                SetMembership("x5", f"X{j}"),
                            ),
                        )
                    )

        # Combine all symbol cases with conjunction
        if pair_formulas:
            combined_pairs = pair_formulas[0]
            for case in pair_formulas[1:]:
                combined_pairs = Conjunction(combined_pairs, case)
        else:
            # If no states with transitions, use a tautology
            return UniversalQuantifier(
                "x5",
                Disjunction(
                    SetMembership("x5", "X0"), Negation(SetMembership("x5", "X0"))
                ),
            )

        # Quantify over both positions
        return UniversalQuantifier(
            "x5",
            UniversalQuantifier(
                "x6",
                Implication(
                    Relation("x6", "=", "x5+1"),
                    combined_pairs,
                ),
            ),
        )

    def _construct_φ_5(self, fsa):
        """
        Construct the formula that ensures accepting states.
        This formula expresses: the last position must be in an accepting state.
        """
        accepting_transitions = []

        for i in range(0, len(fsa.states)):
            state_transitions = []
            for symbol, dst_state in fsa.get_transitions(i):
                if dst_state in fsa.accepting_states:
                    state_transitions.append(SymbolPredicate("Q", "x7", symbol))

            state_disjunction = None
            if state_transitions:
                # If there are accepting transitions, create a disjunction
                state_disjunction = state_transitions[0]

                for k in range(1, len(state_transitions)):
                    state_disjunction = Disjunction(
                        state_disjunction, state_transitions[k]
                    )

            else:
                # If no accepting transitions, add a contradiction
                state_disjunction = Conjunction(
                    Negation(SetMembership("x7", f"X{i}")),
                    SetMembership("x7", f"X{i}"),
                )
            accepting_transitions.append(
                Implication(SetMembership("x7", f"X{i}"), state_disjunction)
            )

        # If no accepting states, the language is empty
        if not accepting_transitions:
            return Conjunction(
                SetMembership("x7", "X0"), Negation(SetMembership("x7", "X0"))
            )

        # Create a conjunction of all states
        if len(accepting_transitions) == 1:
            accepting_formula = accepting_transitions[0]
        else:
            accepting_formula = Conjunction(
                accepting_transitions[0], accepting_transitions[1]
            )
            for i in range(2, len(accepting_transitions)):
                accepting_formula = Conjunction(
                    accepting_formula, accepting_transitions[i]
                )

        # The last position is the one with no successor
        last_position = UniversalQuantifier("x8", Relation("x8", "<=", "x7"))

        return UniversalQuantifier("x7", Implication(last_position, accepting_formula))


def convert_fsa_to_som(fsa):
    """
    Convert a Finite State Automaton to a Second-Order Monadic formula.

    Args:
        fsa: A FiniteStateAutomaton instance

    Returns:
        A SOMFormula equivalent to the automaton
    """
    converter = FSAToSOM()
    formula = converter.convert(fsa)

    return formula
