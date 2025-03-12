from itertools import product

from automathic.fsa.fsa import FiniteStateAutomaton, State
from automathic.ltl.formula import (
    BooleanConstant,
    Conjunction,
    Disjunction,
    Eventually,
    Globally,
    LTLFormula,
    Negation,
    Past,
    Proposition,
    Since,
    SymbolPredicate,
    Until,
)


class LTLtoFSA:
    """
    Converts Linear Temporal Logic formulas to equivalent Finite State Automata.

    This implementation follows a compositional approach where automata are built
    for subformulas and then combined according to LTL semantics. The construction
    handles both future-oriented operators (Until, Eventually) and past-oriented
    operators (Since, Past).
    """

    def __init__(self, alphabet=None):
        """
        Initialize the converter.

        Args:
            alphabet: List of symbols in the alphabet.
                     If None, defaults to the alphabet derived from the formula
        """
        self.counter = 0
        self.alphabet = alphabet

    def convert(self, formula):
        """
        Convert an LTL formula to an FSA

        Args:
            formula: An instance of LTLFormula

        Returns:
            A FiniteStateAutomaton equivalent to the formula
        """
        # Extract the alphabet from the formula if not provided
        if self.alphabet is None:
            self.alphabet = self._extract_alphabet(formula)
            if not self.alphabet:
                self.alphabet = ["a", "b"]  # Default alphabet if none found

        # Apply construction recursively
        fsa = self._convert(formula)

        return fsa.trim().minimize()

    def _extract_alphabet(self, formula):
        """
        Extract the alphabet from a formula by finding all symbol predicates.

        Args:
            formula: The LTL formula

        Returns:
            A list of unique symbols used in the formula
        """
        props = formula.get_propositions()
        symbols = []

        for prop in props:
            if prop.startswith("Q"):
                symbols.append(prop[1:])  # Extract the symbol from SymbolPredicate

        return sorted(set(symbols))

    def _convert(self, formula):
        """
        Internal recursive conversion method that builds automata compositionally.

        Args:
            formula: The formula to convert

        Returns:
            An FSA equivalent to the formula
        """
        if isinstance(formula, SymbolPredicate):
            return self._convert_symbol_predicate(formula)
        elif isinstance(formula, BooleanConstant):
            return self._convert_boolean_constant(formula)
        elif isinstance(formula, Negation):
            # For negation: complement the automaton
            subformula_fsa = self._convert(formula.subformula)
            return subformula_fsa.complement()
        elif isinstance(formula, Conjunction):
            # For conjunction: intersect the automata of both subformulas
            left_fsa = self._convert(formula.left)
            right_fsa = self._convert(formula.right)
            return left_fsa.intersect(right_fsa)
        elif isinstance(formula, Disjunction):
            # For disjunction: union the automata of both subformulas
            left_fsa = self._convert(formula.left)
            right_fsa = self._convert(formula.right)
            return left_fsa.union(right_fsa)
        elif isinstance(formula, Past):
            # For past operator (P): convert to an automaton where the property held at some point in the past
            return self._convert_past(formula)
        elif isinstance(formula, Eventually):
            # For eventually operator (F): convert to an automaton where the property will hold at some point in the future
            return self._convert_eventually(formula)
        elif isinstance(formula, Since):
            # For since operator (φ1 S φ2): convert to an automaton where φ2 held at some point in the past,
            # and φ1 has held continuously since then
            return self._convert_since(formula)
        elif isinstance(formula, Until):
            # For until operator (φ1 U φ2): convert to an automaton where φ1 holds continuously until φ2 becomes true
            return self._convert_until(formula)
        else:
            raise ValueError(f"Unsupported formula type: {type(formula).__name__}")

    def _convert_symbol_predicate(self, predicate):
        """
        Convert a symbol predicate (Qa) to an FSA that accepts only when the current position has symbol 'a'.

        Args:
            predicate: The SymbolPredicate

        Returns:
            An FSA accepting strings with the specified symbol
        """
        # Create a simple 2-state FSA
        fsa = FiniteStateAutomaton(2, self.alphabet)
        fsa.set_initial_state(0)
        fsa.set_accepting_state(1)

        # For the symbol in the predicate, transition from state 0 to state 1
        fsa.set_transition(0, predicate.symbol, 1)

        # Stay in state 1 for any further symbols (accept suffix)
        for symbol in self.alphabet:
            fsa.set_transition(1, symbol, 1)

        return fsa

    def _convert_boolean_constant(self, constant):
        """
        Convert a boolean constant to an FSA.

        Args:
            constant: The BooleanConstant

        Returns:
            An FSA that either accepts everything (True) or nothing (False)
        """
        # Create a 1-state FSA
        fsa = FiniteStateAutomaton(1, self.alphabet)
        fsa.set_initial_state(0)

        # If the constant is True, the state is accepting
        if constant.value:
            fsa.set_accepting_state(0)

        # Self-loops for all symbols
        for symbol in self.alphabet:
            fsa.set_transition(0, symbol, 0)

        return fsa

    def _convert_past(self, formula):
        """
        Convert a Past formula (P φ) to an FSA.

        Args:
            formula: The Past formula

        Returns:
            An FSA accepting strings where φ held at some point in the past
        """
        # Convert the subformula
        subformula_fsa = self._convert(formula.subformula)

        # Create a new FSA with both "not seen" and "seen" states
        num_states = subformula_fsa.num_states * 2
        past_fsa = FiniteStateAutomaton(num_states, self.alphabet)

        # Map original states to new "not seen" and "seen" states
        state_map = {}
        for i in range(subformula_fsa.num_states):
            if subformula_fsa.states[i] is not None:
                # "Not seen" state
                state_map[(i, 0)] = i
                # "Seen" state
                state_map[(i, 1)] = i + subformula_fsa.num_states

                # Create the states in the new FSA
                past_fsa.states[i] = State(i, origin=f"{i}_not_seen")
                past_fsa.states[i + subformula_fsa.num_states] = State(
                    i + subformula_fsa.num_states, origin=f"{i}_seen"
                )

        # Set initial state (in "not seen" part)
        if subformula_fsa.initial_state is not None:
            past_fsa.set_initial_state(state_map[(subformula_fsa.initial_state.id, 0)])

        # Set accepting states (all "seen" states plus accepting "not seen" states)
        for state in subformula_fsa.accepting_states:
            # The "seen" state is always accepting
            past_fsa.set_accepting_state(state_map[(state.id, 1)])
            # The "not seen" state is accepting if the subformula accepts at this position
            past_fsa.set_accepting_state(state_map[(state.id, 0)])

        # Set transitions
        for (src_id, symbol), dst_ids in subformula_fsa.transitions.items():
            for dst_id in dst_ids:
                # "Not seen" to "not seen" transition if the destination is not accepting
                if dst_id not in [
                    state.id for state in subformula_fsa.accepting_states
                ]:
                    past_fsa.set_transition(
                        state_map[(src_id.id, 0)], symbol, state_map[(dst_id.id, 0)]
                    )
                # "Not seen" to "seen" transition if the destination is accepting
                else:
                    past_fsa.set_transition(
                        state_map[(src_id.id, 0)], symbol, state_map[(dst_id.id, 1)]
                    )

                # "Seen" to "seen" transition (once seen, always seen)
                past_fsa.set_transition(
                    state_map[(src_id.id, 1)], symbol, state_map[(dst_id.id, 1)]
                )

        return past_fsa.trim()

    def _convert_eventually(self, formula):
        """
        Convert an Eventually formula (F φ) to an FSA.

        Args:
            formula: The Eventually formula

        Returns:
            An FSA accepting strings where φ eventually holds
        """
        # Convert the subformula
        subformula_fsa = self._convert(formula.subformula)

        # Create a new FSA with "waiting" and "satisfied" states
        num_states = subformula_fsa.num_states * 2
        eventually_fsa = FiniteStateAutomaton(num_states, self.alphabet)

        # Map original states to new states
        state_map = {}
        for i in range(subformula_fsa.num_states):
            if subformula_fsa.states[i] is not None:
                # "Waiting" state
                state_map[(i, 0)] = i
                # "Satisfied" state
                state_map[(i, 1)] = i + subformula_fsa.num_states

                # Create the states in the new FSA
                eventually_fsa.states[i] = State(i, origin=f"{i}_waiting")
                eventually_fsa.states[i + subformula_fsa.num_states] = State(
                    i + subformula_fsa.num_states, origin=f"{i}_satisfied"
                )

        # Set initial state (in "waiting" part)
        if subformula_fsa.initial_state is not None:
            eventually_fsa.set_initial_state(
                state_map[(subformula_fsa.initial_state.id, 0)]
            )

        # Set accepting states (only "satisfied" states are accepting)
        for state in subformula_fsa.accepting_states:
            eventually_fsa.set_accepting_state(state_map[(state.id, 1)])

        # Set transitions
        for (src_id, symbol), dst_ids in subformula_fsa.transitions.items():
            for dst_id in dst_ids:
                # "Waiting" to "waiting" transition
                eventually_fsa.set_transition(
                    state_map[(src_id.id, 0)], symbol, state_map[(dst_id.id, 0)]
                )

                # "Waiting" to "satisfied" transition if the destination is accepting in the subformula
                if dst_id in subformula_fsa.accepting_states:
                    eventually_fsa.set_transition(
                        state_map[(src_id.id, 0)], symbol, state_map[(dst_id.id, 1)]
                    )

                # "Satisfied" to "satisfied" transition (once satisfied, always satisfied)
                eventually_fsa.set_transition(
                    state_map[(src_id.id, 1)], symbol, state_map[(dst_id.id, 1)]
                )

        return eventually_fsa.trim()

    def _convert_since(self, formula):
        """
        Convert a Since formula (φ1 S φ2) to an FSA.

        Args:
            formula: The Since formula

        Returns:
            An FSA accepting strings where φ2 held at some point in the past,
            and φ1 has held continuously since then
        """
        # Convert the subformulas
        left_fsa = self._convert(formula.left)  # φ1
        right_fsa = self._convert(formula.right)  # φ2

        # Create a product automaton with states for tracking "seen φ2" and "maintained φ1"
        num_states = left_fsa.num_states * right_fsa.num_states * 2
        since_fsa = FiniteStateAutomaton(num_states, self.alphabet)

        # Map product states to new state IDs
        state_map = {}
        idx = 0
        for left_state in range(left_fsa.num_states):
            for right_state in range(right_fsa.num_states):
                if (
                    left_fsa.states[left_state] is not None
                    and right_fsa.states[right_state] is not None
                ):
                    # State 0: Not satisfied yet
                    # State 1: φ2 has occurred and φ1 has held since then
                    state_map[(left_state, right_state, 0)] = idx
                    idx += 1
                    state_map[(left_state, right_state, 1)] = idx
                    idx += 1

                    # Create the states in the new FSA
                    origin = f"({left_state},{right_state},not_satisfied)"
                    since_fsa.states[state_map[(left_state, right_state, 0)]] = State(
                        state_map[(left_state, right_state, 0)], origin=origin
                    )

                    origin = f"({left_state},{right_state},satisfied)"
                    since_fsa.states[state_map[(left_state, right_state, 1)]] = State(
                        state_map[(left_state, right_state, 1)], origin=origin
                    )

        # Set initial state
        if left_fsa.initial_state is not None and right_fsa.initial_state is not None:
            since_fsa.set_initial_state(
                state_map[(left_fsa.initial_state.id, right_fsa.initial_state.id, 0)]
            )

        # Build transitions
        for symbol in self.alphabet:
            for left_src, right_src, satisfied in state_map:
                # Get the destination states in the original automata
                left_dests = left_fsa.get_transition_targets(left_src, symbol)
                right_dests = right_fsa.get_transition_targets(right_src, symbol)

                if not left_dests or not right_dests:
                    continue

                for left_dest in left_dests:
                    for right_dest in right_dests:
                        # Check if φ2 is satisfied at the current position
                        right_accepts = right_dest in [
                            s.id for s in right_fsa.accepting_states
                        ]

                        # Check if φ1 is satisfied at the current position
                        left_accepts = left_dest in [
                            s.id for s in left_fsa.accepting_states
                        ]

                        if satisfied == 0:  # Not satisfied yet
                            if right_accepts:
                                # If φ2 is satisfied, transition to satisfied state
                                since_fsa.set_transition(
                                    state_map[(left_src, right_src, 0)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 1)],
                                )
                            else:
                                # Otherwise, stay in not-satisfied state
                                since_fsa.set_transition(
                                    state_map[(left_src, right_src, 0)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 0)],
                                )
                        else:  # Already satisfied
                            if left_accepts:
                                # If φ1 holds, remain in satisfied state
                                since_fsa.set_transition(
                                    state_map[(left_src, right_src, 1)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 1)],
                                )
                            elif right_accepts:
                                # If φ2 holds again, remain in satisfied state (reset the "since" point)
                                since_fsa.set_transition(
                                    state_map[(left_src, right_src, 1)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 1)],
                                )
                            else:
                                # If neither φ1 nor φ2 holds, go back to not-satisfied state
                                since_fsa.set_transition(
                                    state_map[(left_src, right_src, 1)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 0)],
                                )

        # Set accepting states (all satisfied states)
        for left_state, right_state, satisfied in state_map:
            if satisfied == 1:
                since_fsa.set_accepting_state(
                    state_map[(left_state, right_state, satisfied)]
                )

        return since_fsa.trim()

    def _convert_until(self, formula):
        """
        Convert an Until formula (φ1 U φ2) to an FSA.

        Args:
            formula: The Until formula

        Returns:
            An FSA accepting strings where φ1 holds continuously until φ2 becomes true
        """
        # Convert the subformulas
        left_fsa = self._convert(formula.left)  # φ1
        right_fsa = self._convert(formula.right)  # φ2

        # Create a product automaton with states for tracking "waiting for φ2" and "found φ2"
        num_states = left_fsa.num_states * right_fsa.num_states * 2
        until_fsa = FiniteStateAutomaton(num_states, self.alphabet)

        # Map product states to new state IDs
        state_map = {}
        idx = 0
        for left_state in range(left_fsa.num_states):
            for right_state in range(right_fsa.num_states):
                if (
                    left_fsa.states[left_state] is not None
                    and right_fsa.states[right_state] is not None
                ):
                    # State 0: Waiting for φ2, requiring φ1 to hold
                    # State 1: φ2 has occurred, formula is satisfied
                    state_map[(left_state, right_state, 0)] = idx
                    idx += 1
                    state_map[(left_state, right_state, 1)] = idx
                    idx += 1

                    # Create the states in the new FSA
                    origin = f"({left_state},{right_state},waiting)"
                    until_fsa.states[state_map[(left_state, right_state, 0)]] = State(
                        state_map[(left_state, right_state, 0)], origin=origin
                    )

                    origin = f"({left_state},{right_state},satisfied)"
                    until_fsa.states[state_map[(left_state, right_state, 1)]] = State(
                        state_map[(left_state, right_state, 1)], origin=origin
                    )

        # Set initial state
        if left_fsa.initial_state is not None and right_fsa.initial_state is not None:
            until_fsa.set_initial_state(
                state_map[(left_fsa.initial_state.id, right_fsa.initial_state.id, 0)]
            )

        # Build transitions
        for symbol in self.alphabet:
            for left_src, right_src, satisfied in state_map:
                # Get the destination states in the original automata
                left_dests = left_fsa.get_transition_targets(left_src, symbol)
                right_dests = right_fsa.get_transition_targets(right_src, symbol)

                if not left_dests or not right_dests:
                    continue

                for left_dest in left_dests:
                    for right_dest in right_dests:
                        # Check if φ2 is satisfied at the current position
                        right_accepts = right_dest in [
                            s.id for s in right_fsa.accepting_states
                        ]

                        # Check if φ1 is satisfied at the current position
                        left_accepts = left_dest in [
                            s.id for s in left_fsa.accepting_states
                        ]

                        if satisfied == 0:  # Still waiting for φ2
                            if right_accepts:
                                # If φ2 is satisfied, transition to satisfied state
                                until_fsa.set_transition(
                                    state_map[(left_src, right_src, 0)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 1)],
                                )
                            elif left_accepts:
                                # If φ1 holds but not φ2, continue waiting
                                until_fsa.set_transition(
                                    state_map[(left_src, right_src, 0)],
                                    symbol,
                                    state_map[(left_dest, right_dest, 0)],
                                )
                                # Note: If neither holds, there's no transition (formula fails)
                        else:  # Already satisfied
                            # Once satisfied, the formula remains satisfied
                            until_fsa.set_transition(
                                state_map[(left_src, right_src, 1)],
                                symbol,
                                state_map[(left_dest, right_dest, 1)],
                            )

        # Set accepting states (all satisfied states)
        for left_state, right_state, satisfied in state_map:
            if satisfied == 1:
                until_fsa.set_accepting_state(
                    state_map[(left_state, right_state, satisfied)]
                )

        return until_fsa.trim()


def convert_ltl_to_fsa(formula, alphabet=None):
    """
    Convert an LTL formula to an FSA.

    Args:
        formula: An LTLFormula instance or formula string
        alphabet: List of symbols in the alphabet. If None, defaults to the
            alphabet of symbols in the formula's predicates.

    Returns:
        A FiniteStateAutomaton equivalent to the formula
    """
    from automathic.ltl.parser import parse_ltl

    # Parse string formulas
    if isinstance(formula, str):
        formula = parse_ltl(formula)

    # Convert to FSA
    converter = LTLtoFSA(alphabet)
    automaton = converter.convert(formula)

    return automaton
