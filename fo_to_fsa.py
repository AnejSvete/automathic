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
        fsa = V_structure_fsa.intersect(formula_fsa)

        # Always trim and minimize the resulting automaton
        fsa = fsa.trim().minimize()
        # fsa = formula_fsa.trim().minimize()

        if formula.is_sentence():
            return self._project(fsa)
        else:
            return fsa, formula_fsa, V_structure_fsa

    def _project(self, fsa):
        """Project the automaton transitions to alphabet symbols only

        Args:
            fsa: The automaton to project
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
            symbol, _ = old_symbol

            # Create new symbol without the variable
            new_symbol = symbol
            proj_fsa.set_transition(src_state.id, new_symbol, dst_state.id)

        # Set accepting states
        for state in fsa.accepting_states:
            proj_fsa.set_accepting_state(state.id)

        return proj_fsa.trim()  # Not minimizing to preserve state structure

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
            # print("DEBUG: Negation")
            # X = self._convert(formula.subformula, V, alphabet)
            # print(X.ascii())
            return self._convert(formula.subformula, V, alphabet).complement()
            # return X.complement()
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
            raise ValueError("Universal quantification is not supported")
            # # For universal quantification, we use the equivalence: ∀x.φ(x) ≡ ¬∃x.¬φ(x)
            # negated_sub = Negation(formula.subformula)
            # existential = ExistentialQuantifier(formula.variable, negated_sub)
            # return self._convert(existential, V, alphabet)
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
        Remove variable by creating an NFSA with 'seen' and 'unseen' states
        that allows non-deterministic transitions. The NFSA will be determinized
        later in the process.
        """
        from itertools import product

        from nfa import NonDeterministicFSA

        print(f"DEBUG: Starting _remove_variable with variable={variable}")
        print(
            f"DEBUG: Original FSA has {fsa.num_states} states and {len(fsa.transitions)} transitions"
        )
        print(f"DEBUG: Original alphabet size: {len(fsa.alphabet)}")

        # Print all original transitions for reference
        print("\nORIGINAL TRANSITIONS:")
        for (src, symbol), dst in fsa.transitions.items():
            print(f"  {src.label} --{symbol}--> {dst.label}")

        # Create new alphabet without the projected variable
        V_new = sorted(set(V) - {variable})
        alphabet_new = list(product(self.alphabet, self._powerset(V_new)))

        print(f"\nDEBUG: New alphabet without {variable}: size={len(alphabet_new)}")
        print(f"DEBUG: Sample of new alphabet: {alphabet_new[:3]}...")

        # Create mapping from pairs (original_state, seen_bit) to new state IDs
        p2idx = {}  # (state_id, seen_bit) -> new_state_id
        idx = 0

        for state_id in range(fsa.num_states):
            if fsa.states[state_id] is not None:
                # Each state gets two copies - one for 'seen=0' and one for 'seen=1'
                p2idx[(state_id, 0)] = idx
                idx += 1
                p2idx[(state_id, 1)] = idx
                idx += 1

        print(f"\nDEBUG: Created mapping with {len(p2idx)} state pairs")
        print(f"DEBUG: Mapping sample: {list(p2idx.items())[:3]}...")

        # Create a new NFSA with double the number of states
        result = NonDeterministicFSA(idx, alphabet_new)

        # Copy state origins and set labels
        for (orig_id, seen_bit), new_id in p2idx.items():
            if fsa.states[orig_id] is not None:
                origin = fsa.states[orig_id].origin
                # Add 'seen' indicator to state label
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
            print(f"DEBUG: Set initial state to {initial_id}")
        else:
            print("DEBUG: Original FSA has no initial state!")

        # Process transitions
        transition_count = 0
        added_transitions = []  # Track transitions that were added
        skipped_transitions = []  # Track transitions that were skipped
        attempted_transitions = []  # Track all transitions we attempt to add

        print("\nPROCESSING TRANSITIONS:")
        for (src_state, old_symbol), dst_state in fsa.transitions.items():
            symbol, vars_at_pos = old_symbol
            print(
                f"\nProcessing: {src_state.label} --({symbol},{vars_at_pos})--> {dst_state.label}"
            )

            # Create new symbol without the projected variable
            new_vars = tuple(sorted([v for v in vars_at_pos if v != variable]))
            new_symbol = (symbol, new_vars)
            print(f"  New symbol: {new_symbol}")

            # Check if the variable is present in this transition
            var_present = variable in vars_at_pos
            print(
                f"  Variable {variable} is {'present' if var_present else 'NOT present'} in this transition"
            )

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

            # Track what we're trying to add
            attempted_transitions.append(
                (unseen_src_id, new_symbol, unseen_dst_id, "unseen->unseen/seen")
            )
            attempted_transitions.append(
                (seen_src_id, new_symbol, seen_dst_id, "seen->seen")
            )

            try:
                # For the unseen bit (0)
                print(
                    f"  Adding: {unseen_src_id} --{new_symbol}--> {unseen_dst_id} (unseen->{('seen' if var_present else 'unseen')})"
                )
                print(
                    f"  Adding: {result.states[unseen_src_id].label} --{new_symbol}--> {result.states[unseen_dst_id].label} (unseen->{('seen' if var_present else 'unseen')})"
                )
                result.set_transition(
                    unseen_src_id,  # From state with seen=0
                    new_symbol,  # New symbol
                    unseen_dst_id,  # To state with seen=0 or seen=1
                )
                transition_count += 1
                added_transitions.append(
                    (unseen_src_id, new_symbol, unseen_dst_id, "unseen->unseen/seen")
                )

                # For the seen bit (1)
                print(
                    f"  Adding: {seen_src_id} --{new_symbol}--> {seen_dst_id} (seen->seen)"
                )
                print(
                    f"  Adding: {result.states[seen_src_id].label} --{new_symbol}--> {result.states[seen_dst_id].label} (seen->seen)"
                )
                result.set_transition(
                    seen_src_id,  # From state with seen=1
                    new_symbol,  # New symbol
                    seen_dst_id,  # To state with seen=1
                )
                transition_count += 1
                added_transitions.append(
                    (seen_src_id, new_symbol, seen_dst_id, "seen->seen")
                )

                # Add non-deterministic transitions
                if var_present:
                    # When variable is present, also add non-deterministic transition to allow
                    # other possible destinations for the same source and symbol
                    for (other_src, other_symbol), other_dst in fsa.transitions.items():
                        if other_src.id == src_state.id and other_symbol[0] == symbol:
                            # Skip the exact same transition
                            if (
                                other_dst.id == dst_state.id
                                and other_symbol[1] == vars_at_pos
                            ):
                                continue

                            # Create the new symbol without the variable
                            other_new_vars = tuple(
                                sorted([v for v in other_symbol[1] if v != variable])
                            )
                            other_new_symbol = (other_symbol[0], other_new_vars)

                            # If this is the same symbol we're currently processing
                            if other_new_symbol == new_symbol:
                                # Determine if this other transition involves the variable
                                other_var_present = variable in other_symbol[1]

                                # Determine destination state ID
                                if other_var_present:
                                    other_unseen_dst_id = p2idx[
                                        (other_dst.id, 1)
                                    ]  # unseen -> seen
                                    other_seen_dst_id = p2idx[
                                        (other_dst.id, 1)
                                    ]  # seen -> seen
                                else:
                                    other_unseen_dst_id = p2idx[
                                        (other_dst.id, 0)
                                    ]  # unseen -> unseen
                                    other_seen_dst_id = p2idx[
                                        (other_dst.id, 1)
                                    ]  # seen -> seen

                                # Add non-deterministic transitions
                                print(
                                    f"  Adding non-deterministic: {unseen_src_id} --{new_symbol}--> {other_unseen_dst_id}"
                                )
                                result.set_transition(
                                    unseen_src_id, new_symbol, other_unseen_dst_id
                                )
                                transition_count += 1
                                added_transitions.append(
                                    (
                                        unseen_src_id,
                                        new_symbol,
                                        other_unseen_dst_id,
                                        "non-deterministic",
                                    )
                                )

                                print(
                                    f"  Adding non-deterministic: {seen_src_id} --{new_symbol}--> {other_seen_dst_id}"
                                )
                                result.set_transition(
                                    seen_src_id, new_symbol, other_seen_dst_id
                                )
                                transition_count += 1
                                added_transitions.append(
                                    (
                                        seen_src_id,
                                        new_symbol,
                                        other_seen_dst_id,
                                        "non-deterministic",
                                    )
                                )

            except Exception as e:
                print(f"  ERROR setting transitions: {e}")
                skipped_transitions.append(
                    ((src_state.id, old_symbol, dst_state.id), str(e))
                )

        print(f"\nDEBUG: Added {transition_count} transitions to result FSA")

        # Set accepting states - a state is accepting if it corresponds to an accepting
        # state in the original FSA and the variable has been seen
        accepting_count = 0
        print("\nSETTING ACCEPTING STATES:")
        for state in fsa.accepting_states:
            try:
                accept_id = p2idx[(state.id, 1)]
                print(
                    f"  Setting accepting state: {accept_id} (from original {state.id})"
                )
                result.set_accepting_state(accept_id)
                accepting_count += 1
            except Exception as e:
                print(f"  ERROR setting accepting state: {e}")
                print(f"  Failed for state={state.id}")

        print(f"\nDEBUG: Set {accepting_count} accepting states")

        # Summary before trimming
        print("\n===== TRANSITION SUMMARY =====")
        print(f"Attempted: {len(attempted_transitions)} transitions")
        print(f"Added: {len(added_transitions)} transitions")
        print(f"Skipped: {len(skipped_transitions)} transitions")

        # Check for transitions in attempted but not in added
        missed_transitions = [
            t
            for t in attempted_transitions
            if (t[0], t[1], t[2]) not in [(a[0], a[1], a[2]) for a in added_transitions]
        ]
        if missed_transitions:
            print("\nMISSED TRANSITIONS (attempted but not added):")
            for t in missed_transitions:
                print(f"  {t[0]} --{t[1]}--> {t[2]} ({t[3]})")

        # Before trimming, check if the result has initial and accepting states
        if result.initial_state is None:
            print("\nWARNING: Result FSA has no initial state!")

        if not result.accepting_states:
            print("\nWARNING: Result FSA has no accepting states!")

        # Print all transitions in result FSA
        print("\nRESULT FSA TRANSITIONS:")
        for (src, symbol), dst_set in result.transitions.items():
            for dst in dst_set:
                print(f"  {src.label} --{symbol}--> {dst.label}")

        # Check if there's a path from initial state to an accepting state
        if result.initial_state:
            path_found = False
            visited = set()
            frontier = [(result.initial_state.id, [])]  # (state_id, path)

            print("\nPATH ANALYSIS:")
            while frontier and not path_found:
                state_id, path = frontier.pop(0)
                if state_id in visited:
                    continue

                visited.add(state_id)
                state = result.states[state_id]

                if state in result.accepting_states:
                    path_found = True
                    print(f"OK: Found path(s) from initial to accepting states")
                    break

                for (src, sym), dst_set in result.transitions.items():
                    if src.id == state_id:
                        for dst in dst_set:
                            if dst.id not in visited:
                                new_path = path + [(sym, dst.id)]
                                frontier.append((dst.id, new_path))

            if not path_found:
                print("WARNING: No path found from initial to accepting states!")

        # Remove any unnecessary states and transitions
        print("\nSTARTING TRIM OPERATION...")
        trimmed_result = result.trim()
        print(
            f"DEBUG: After trimming: {trimmed_result.num_states} states, {len(trimmed_result.transitions)} transitions"
        )

        # Check if the trim made the FSA empty
        if trimmed_result.num_states == 0:
            print("CRITICAL ERROR: Trimming resulted in an empty FSA!")
            print("This likely means there's no path from initial to accepting states.")
            return result  # Return the untrimmed FSA for debugging

        print("\nRESULT FSA AFTER TRIMMING:")
        for (src, symbol), dst_set in trimmed_result.transitions.items():
            for dst in dst_set:
                print(f"  {src.label} --{symbol}--> {dst.label}")

        fsa = trimmed_result.determinize().minimize()

        return fsa


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
