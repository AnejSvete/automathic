from collections import defaultdict

from automathic.fsa.nfa import NonDeterministicFSA, State


class FiniteStateAutomaton(NonDeterministicFSA):
    """
    Deterministic Finite State Automaton (DFA) implementation.

    A DFA is a special case of NFA where:
    1. Each (state, symbol) pair has exactly one outgoing transition
    2. There are no epsilon (empty string) transitions

    This implementation extends the NonDeterministicFSA class but enforces the
    deterministic constraint by storing at most one destination state for each
    (state, symbol) pair.

    Theoretical Background:
    - DFAs recognize regular languages
    - For any NFA, there exists an equivalent DFA (potentially with more states)
    - DFAs offer efficient string recognition (linear in the length of the string)
    """

    def __init__(self, num_states, alphabet):
        """
        Initialize a deterministic FSA with a given number of states and alphabet.

        Args:
            num_states (int): Number of states in the automaton
            alphabet (list): List of symbols in the alphabet
        """
        super().__init__(num_states, alphabet)
        # Override the transitions data structure to enforce determinism
        # In a DFA, each (state, symbol) pair maps to exactly one state
        self.transitions = {}

    def set_transition(self, src_id, symbol, dst_id):
        """
        Set a transition in the DFA.

        For a DFA, each (state, symbol) pair can transition to at most one state.
        If a transition already exists for this pair, it will be overwritten.

        Args:
            src_id (int): Source state ID
            symbol: Alphabet symbol triggering the transition
            dst_id (int): Destination state ID
        """
        if src_id >= len(self.states) or dst_id >= len(self.states):
            raise ValueError("State index out of bounds")

        src_state = self.states[src_id]
        dst_state = self.states[dst_id]

        # For a DFA, overwrite any existing transition for this state-symbol pair
        self.transitions[(src_state, symbol)] = dst_state

    def transition(self, state, symbol):
        """
        Get the next state for a given state and symbol.

        Args:
            state: Either a state ID (int) or a State object
            symbol: An alphabet symbol

        Returns:
            The destination State object, or None if no transition exists
        """
        state_obj = self._get_state_obj(state)
        return self.transitions.get((state_obj, symbol), None)

    def accepts(self, input_string):
        """
        Check if the DFA accepts the given input string.

        A string is accepted if, starting from the initial state and following transitions
        for each symbol in the string, we end in an accepting state.

        Args:
            input_string (str): The string to check for acceptance

        Returns:
            bool: True if the string is accepted, False otherwise
        """
        if self.initial_state is None:
            return False

        state = self.initial_state
        for symbol in input_string:
            state = self.transition(state, symbol)
            if state is None:
                return False
        return state in self.accepting_states

    def minimize(self):
        """
        Create a new minimized FSA using Hopcroft's algorithm.

        Hopcroft's algorithm works by partitioning states into equivalence classes:
        1. Start with the partition [accepting states, non-accepting states]
        2. Repeatedly refine the partition by splitting groups based on transitions
        3. States in the same final group are equivalent and can be merged

        Returns:
            A new FiniteStateAutomaton with the minimal number of states that accepts
            the same language as the original automaton
        """
        # Start with the partition [accepting states, non-accepting states]
        partition = [
            self.accepting_states,
            set(self.states) - self.accepting_states,
        ]

        # Remove any empty partitions
        partition = [p for p in partition if p]

        # Repeatedly refine the partition until no more refinement is possible
        old_partition = []
        while old_partition != partition:
            old_partition = partition.copy()
            new_partition = []
            for group in partition:
                # Split each group based on transitions
                split_groups = defaultdict(set)
                for state in group:
                    # Create a signature for this state based on transitions to other groups
                    signature = []
                    for symbol in self.alphabet:
                        next_state = self.transition(state, symbol)
                        if next_state is None:
                            signature.append(-1)  # Special value for no transition
                        else:
                            # Find which group the next state belongs to
                            group_idx = next(
                                i for i, g in enumerate(partition) if next_state in g
                            )
                            signature.append(group_idx)

                    # Use the signature to group similar states
                    split_groups[tuple(signature)].add(state)

                # Add each split group to the new partition
                new_partition.extend(split_groups.values())

            partition = new_partition

        # If no states remain, return empty FSA
        if not partition:
            return FiniteStateAutomaton(0, self.alphabet)

        # Create new minimal FSA
        result = FiniteStateAutomaton(len(partition), self.alphabet)

        # Create a mapping from old states to new states
        state_map = {}
        for i, group in enumerate(partition):
            # Combine the origins of all states in the group
            combined_origin = tuple(
                str(origin) for state in group for origin in state.origin
            )

            # Create a new state with the combined origins
            result.states[i] = State(i, origin=combined_origin)

            # Map each state in the group to the new state ID
            for state in group:
                state_map[state] = i

        # Map initial state
        if self.initial_state:
            result.set_initial_state(state_map[self.initial_state])

        # Map accepting states
        for i, group in enumerate(partition):
            if any(state in self.accepting_states for state in group):
                result.set_accepting_state(i)

        # Map transitions
        for i, group in enumerate(partition):
            representative = min(group, key=lambda s: s.id)
            for symbol in self.alphabet:
                next_state = self.transition(representative, symbol)
                if next_state is not None:
                    next_group = state_map[next_state]
                    result.set_transition(i, symbol, next_group)

        return result

    def is_complete(self):
        """
        Check if the DFA is complete (has transitions for all symbols from all states).

        A complete DFA has a transition defined for every state-symbol pair.

        Returns:
            bool: True if the DFA is complete, False otherwise
        """
        for state_id in range(self.num_states):
            state = self.states[state_id]
            if state is None:
                continue
            for symbol in self.alphabet:
                if self.transition(state_id, symbol) is None:
                    return False
        return True

    def complete(self):
        """
        Make the DFA complete by adding a sink state if needed.

        A sink state is a non-accepting state that transitions to itself for all symbols.
        This is added to handle missing transitions in the original DFA.

        Returns:
            A new complete DFA
        """
        if self.is_complete():
            return self.copy()

        # Create a new FSA with an additional sink state
        result = FiniteStateAutomaton(self.num_states + 1, self.alphabet)
        sink_state = self.num_states  # The new sink state

        # Copy all existing transitions
        for state_id in range(self.num_states):
            for symbol in self.alphabet:
                next_state = self.transition(state_id, symbol)
                if next_state is not None:
                    # Use state ID consistently
                    result.set_transition(state_id, symbol, next_state.id)
                else:
                    result.set_transition(state_id, symbol, sink_state)

        # Add transitions from sink state to itself
        for symbol in self.alphabet:
            result.set_transition(sink_state, symbol, sink_state)

        # Copy initial state
        if self.initial_state is not None:
            result.set_initial_state(self.initial_state.id)

        # Copy accepting states
        for state in self.accepting_states:
            result.set_accepting_state(state.id)

        return result

    def complement(self):
        """
        Create a DFA that accepts exactly the strings rejected by this automaton.

        The complement operation:
        1. Makes the DFA complete (adding a sink state if needed)
        2. Flips the accepting/non-accepting status of all states

        Returns:
            A new DFA that accepts the complement language
        """
        # Make sure the DFA is complete
        complete_dfa = self.complete()

        # Create a new DFA with the same structure but complemented accepting states
        result = FiniteStateAutomaton(complete_dfa.num_states, complete_dfa.alphabet)

        # Copy all transitions
        for (src, symbol), dst in complete_dfa.transitions.items():
            result.set_transition(src.id, symbol, dst.id)

        # Copy initial state
        if complete_dfa.initial_state is not None:
            result.set_initial_state(complete_dfa.initial_state.id)

        # Complement accepting states
        for state in complete_dfa.states:
            if state not in complete_dfa.accepting_states:
                result.set_accepting_state(state.id)

        return result

    def copy(self):
        """
        Create a deep copy of this DFA.

        Returns:
            A new FiniteStateAutomaton identical to this one
        """
        result = FiniteStateAutomaton(self.num_states, self.alphabet)

        # Copy transitions
        for (src, symbol), dst in self.transitions.items():
            result.set_transition(src.id, symbol, dst.id)

        # Copy initial state
        if self.initial_state is not None:
            result.set_initial_state(self.initial_state.id)

        # Copy accepting states
        for state in self.accepting_states:
            result.set_accepting_state(state.id)

        return result
