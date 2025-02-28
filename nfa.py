from fsa import FiniteStateAutomaton, State


class NonDeterministicFSA(FiniteStateAutomaton):
    """
    Non-deterministic Finite State Automaton implementation.
    Extends the FiniteStateAutomaton class but allows multiple transitions
    for the same state and symbol.
    """

    def __init__(self, num_states, alphabet):
        """Initialize a non-deterministic FSA with a given number of states and alphabet."""
        super().__init__(num_states, alphabet)
        # Redefine transitions as a dictionary mapping (state, symbol) to a set of states
        self.transitions = {}

    def set_transition(self, src_id, symbol, dst_id):
        """
        Set a transition in the NFSA. For the same source state and symbol,
        multiple destination states are allowed.
        """
        if src_id >= len(self.states) or dst_id >= len(self.states):
            raise ValueError("State index out of bounds")

        src_state = self.states[src_id]
        dst_state = self.states[dst_id]

        # If this transition key doesn't exist yet, create a new set
        if (src_state, symbol) not in self.transitions:
            self.transitions[(src_state, symbol)] = set()

        # Add the destination state to the set
        self.transitions[(src_state, symbol)].add(dst_state)

    def get_transitions(self, state_id, symbol=None):
        """
        Get all transitions from a state, optionally filtered by symbol.
        Returns a list of (symbol, destination_state) tuples.
        """
        if state_id >= len(self.states):
            raise ValueError("State index out of bounds")

        state = self.states[state_id]
        transitions = []

        for (src, sym), destinations in self.transitions.items():
            if src == state and (symbol is None or sym == symbol):
                for dst in destinations:
                    transitions.append((sym, dst.id))

        return transitions

    def trim(self):
        """
        Remove unreachable states and states that cannot reach an accepting state.
        Return a new NFSA with only the necessary states.
        """
        # Implementation similar to FSA.trim() but adapted for non-deterministic transitions
        if self.initial_state is None:
            # No initial state, return empty automaton
            return NonDeterministicFSA(0, self.alphabet)

        # Find reachable states
        reachable = set()
        frontier = {self.initial_state.id}

        while frontier:
            state_id = frontier.pop()
            reachable.add(state_id)

            for (src, _), destinations in self.transitions.items():
                if src.id == state_id:
                    for dst in destinations:
                        if dst.id not in reachable:
                            frontier.add(dst.id)

        # Find states that can reach an accepting state
        can_accept = set(state.id for state in self.accepting_states)
        old_size = -1

        while len(can_accept) > old_size:
            old_size = len(can_accept)

            for (src, _), destinations in self.transitions.items():
                if any(dst.id in can_accept for dst in destinations):
                    can_accept.add(src.id)

        # Keep only states that are both reachable and can reach an accepting state
        keep_states = reachable.intersection(can_accept)

        # Create a new automaton with only the necessary states
        num_kept = len(keep_states)
        if num_kept == 0:
            return NonDeterministicFSA(0, self.alphabet)

        result = NonDeterministicFSA(num_kept, self.alphabet)

        # Create a mapping from old state IDs to new state IDs
        old_to_new = {}
        for i, old_id in enumerate(sorted(keep_states)):
            old_to_new[old_id] = i
            old_state = self.states[old_id]
            result.states[i] = State(i, origin=old_state.origin, label=old_state.label)

        # Set initial and accepting states
        if self.initial_state.id in keep_states:
            result.set_initial_state(old_to_new[self.initial_state.id])

        for state in self.accepting_states:
            if state.id in keep_states:
                result.set_accepting_state(old_to_new[state.id])

        # Copy transitions
        for (src, symbol), destinations in self.transitions.items():
            if src.id in keep_states:
                for dst in destinations:
                    if dst.id in keep_states:
                        result.set_transition(
                            old_to_new[src.id], symbol, old_to_new[dst.id]
                        )

        return result

    def determinize(self):
        """
        Convert this NFSA to a deterministic FSA.
        Returns a new FiniteStateAutomaton that accepts the same language.
        """
        from fsa import FiniteStateAutomaton

        if self.initial_state is None:
            return FiniteStateAutomaton(0, self.alphabet)

        # Map from sets of states to new state IDs
        subset_to_id = {}
        id_to_subset = {}

        # Start with the initial state
        initial_subset = frozenset([self.initial_state.id])
        subset_to_id[initial_subset] = 0
        id_to_subset[0] = initial_subset

        # Create a new deterministic FSA - set to max possible size to start
        max_possible_states = 2 ** len([s for s in self.states if s is not None])
        result = FiniteStateAutomaton(max_possible_states, self.alphabet)
        result.num_states = 1  # But only 1 state is actually defined initially

        # Set the origin for the initial state
        if len(initial_subset) == 1:
            origin = self.states[next(iter(initial_subset))].origin
            result.states[0] = State(0, origin=origin)
        else:
            origins = [
                self.states[s].origin
                for s in initial_subset
                if self.states[s].origin is not None
            ]
            if origins:
                # Combine origins if there are multiple
                result.states[0] = State(0, origin=tuple(origins))

        result.set_initial_state(0)

        # Process subsets of states
        todo = [initial_subset]
        next_id = 1

        while todo:
            current_subset = todo.pop()
            current_id = subset_to_id[current_subset]

            # For each symbol in the alphabet
            for symbol in self.alphabet:
                # Find all possible next states
                next_states = set()
                for state_id in current_subset:
                    for (src, sym), destinations in self.transitions.items():
                        if src.id == state_id and sym == symbol:
                            next_states.update(dst.id for dst in destinations)

                if not next_states:
                    continue

                next_subset = frozenset(next_states)

                # Create a new state if we haven't seen this subset before
                if next_subset not in subset_to_id:
                    subset_to_id[next_subset] = next_id
                    id_to_subset[next_id] = next_subset

                    # Create the state with appropriate origin
                    if len(next_subset) == 1:
                        origin = self.states[next(iter(next_subset))].origin
                        result.states[next_id] = State(next_id, origin=origin)
                    else:
                        origins = [
                            self.states[s].origin
                            for s in next_subset
                            if self.states[s].origin is not None
                        ]
                        if origins:
                            # Combine origins if there are multiple
                            result.states[next_id] = State(
                                next_id, origin=tuple(origins)
                            )

                    todo.append(next_subset)
                    result.num_states += 1  # Update the number of states
                    next_id += 1

                # Add the transition
                result.set_transition(current_id, symbol, subset_to_id[next_subset])

        # Set accepting states
        accepting_ids = set(state.id for state in self.accepting_states)
        for subset, state_id in subset_to_id.items():
            if any(s in accepting_ids for s in subset):
                result.set_accepting_state(state_id)

        # Trim the result FSA to the actual number of states used
        # Create a new FSA with just the states we've actually used
        final_result = FiniteStateAutomaton(result.num_states, self.alphabet)

        # Copy all the used states
        for i in range(result.num_states):
            final_result.states[i] = result.states[i]

        # Copy initial state
        if result.initial_state:
            final_result.set_initial_state(result.initial_state.id)

        # Copy accepting states
        for state in result.accepting_states:
            if state.id < result.num_states:
                final_result.set_accepting_state(state.id)

        # Copy transitions
        for (src, symbol), dst in result.transitions.items():
            if src.id < result.num_states and dst.id < result.num_states:
                final_result.set_transition(src.id, symbol, dst.id)

        return final_result
