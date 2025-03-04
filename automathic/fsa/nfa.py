class State:
    """
    Represents a state in a finite state automaton.

    Each state has:
    - A unique ID for identification
    - An origin field that tracks where this state came from (useful for operations like product construction)
    - An optional label for display purposes

    States can be compared by ID for equality and used as dictionary keys.
    """

    def __init__(self, id, origin=None, label=None):
        """
        Initialize a state.

        Args:
            id (int): Numeric ID for the state
            origin: Tuple representing the history/origin of the state
            label: Optional label for the state (defaults to str(id))
        """
        self.id = id
        # Initialize origin as a tuple containing just this state's ID if not provided
        self.origin = origin if origin is not None else (id,)
        # Use the label if provided, otherwise use the origin tuple representation
        self.label = label if label is not None else str(self.origin)

    def __eq__(self, other):
        """
        States are equal if they have the same ID. Also allows comparison with integers.
        """
        if isinstance(other, State):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        return False

    def __hash__(self):
        """
        Hash function based on state ID, allowing States to be used as dictionary keys.
        """
        return hash(self.id)

    def __repr__(self):
        return f"State({self.id}, origin={self.origin}, '{self.label}')"

    def __str__(self):
        return self.label


class NonDeterministicFSA:
    """
    Non-deterministic Finite State Automaton (NFA) implementation.

    An NFA allows:
    1. Multiple transitions from the same state on the same symbol
    2. Epsilon transitions (though not implemented in this version)

    Theoretical Background:
    - NFAs, like DFAs, recognize regular languages
    - NFAs can be more compact than equivalent DFAs
    - Every NFA can be converted to an equivalent DFA (via the subset construction)
    """

    def __init__(self, num_states, alphabet):
        """
        Initialize a non-deterministic FSA.

        Args:
            num_states (int): Number of states in the automaton
            alphabet: Either a list of symbols or an integer size
        """
        # Handle both alphabet list and alphabet size
        if isinstance(alphabet, list):
            self.alphabet = alphabet
            self.alphabet_size = len(alphabet)
        else:
            # For backward compatibility, if an integer is passed
            self.alphabet_size = alphabet
            self.alphabet = list(range(alphabet))

        self.num_symbols = len(self.alphabet)

        # Create state objects with origin tracking
        self.states = [State(i) for i in range(num_states)]
        self.num_states = num_states

        # Initialize transitions dictionary: {(state, symbol): set(next_states)}
        self.transitions = {}
        self.initial_state = None  # Single initial state
        self.accepting_states = set()

        # For visualization
        self.theme = "dark"  # Default theme

    def _get_state_obj(self, state_id):
        """
        Convert state ID to state object if needed.

        Args:
            state_id: Either a state ID (int) or a State object

        Returns:
            State: The corresponding State object

        Raises:
            ValueError: If state_id is invalid
        """
        if isinstance(state_id, State):
            return state_id
        elif isinstance(state_id, int) and 0 <= state_id < self.num_states:
            return self.states[state_id]
        else:
            raise ValueError(f"Invalid state: {state_id}")

    def set_state_label(self, state_id, label):
        """
        Set a human-readable label for a state.

        Args:
            state_id: State ID or object
            label: The new label string
        """
        state = self._get_state_obj(state_id)
        state.label = label

    def set_transition(self, src_id, symbol, dst_id):
        """
        Set a transition in the NFA.

        For NFAs, multiple destination states are allowed for the same (state, symbol) pair.

        Args:
            src_id (int): Source state ID
            symbol: Alphabet symbol triggering the transition
            dst_id (int): Destination state ID
        """

        src_state, dst_state = self._get_state_obj(src_id), self._get_state_obj(dst_id)

        # If this transition key doesn't exist yet, create a new set
        if (src_state, symbol) not in self.transitions:
            self.transitions[(src_state, symbol)] = set()

        # Add the destination state to the set
        self.transitions[(src_state, symbol)].add(dst_state)

    def set_initial_state(self, state):
        """
        Set the initial state.

        Args:
            state: Either a state ID or a State object
        """
        state_obj = self._get_state_obj(state)
        self.initial_state = state_obj

    def set_accepting_state(self, state):
        """
        Set a state as an accepting state.

        Args:
            state: Either a state ID or a State object
        """
        state_obj = self._get_state_obj(state)
        self.accepting_states.add(state_obj)

    def get_transitions(self, state_id, symbol=None):
        """
        Get all transitions from a state, optionally filtered by symbol.

        Args:
            state_id: Either a state ID (int) or a State object
            symbol: Optional symbol to filter by

        Returns:
            List of (symbol, destination_state_id) tuples

        Raises:
            ValueError: If state_id is invalid
        """
        # Handle both State objects and numeric IDs
        if isinstance(state_id, State):
            state = state_id
            state_id = state.id
        elif isinstance(state_id, int):
            if state_id >= len(self.states):
                raise ValueError("State index out of bounds")
            state = self.states[state_id]
        else:
            raise ValueError(f"Invalid state: {state_id}")

        transitions = []

        for (src, sym), destinations in self.transitions.items():
            if src == state and (symbol is None or sym == symbol):
                if isinstance(destinations, set):  # NFA case
                    for dst in destinations:
                        transitions.append((sym, dst.id))
                else:  # DFA case (single state)
                    transitions.append((sym, destinations.id))

        return transitions

    def accepts(self, input_string):
        """
        Check if the NFA accepts the given input string using a breadth-first search.

        An NFA accepts a string if, starting from the initial state and following all
        possible transitions for each symbol, at least one path leads to an accepting state.

        Args:
            input_string (str): The string to check for acceptance

        Returns:
            bool: True if the string is accepted, False otherwise
        """
        if self.initial_state is None:
            return False

        current_states = {self.initial_state}

        for symbol in input_string:
            if not current_states:
                return False

            next_states = set()
            for state in current_states:
                if (state, symbol) in self.transitions:
                    destinations = self.transitions[(state, symbol)]
                    if isinstance(destinations, set):  # NFA case
                        next_states.update(destinations)
                    else:  # DFA case (single state)
                        next_states.add(destinations)

            current_states = next_states

        # Check if any current state is accepting
        return any(state in self.accepting_states for state in current_states)

    def to_som(self):
        """Convert the FSA to a SOM formula."""
        from automathic.fo.translator import convert_fsa_to_som

        return convert_fsa_to_som(self)

    def trim(self):
        """
        Remove unreachable states and states that cannot reach an accepting state.

        This operation:
        1. Identifies states reachable from the initial state
        2. Identifies states that can reach an accepting state
        3. Keeps only states that satisfy both conditions

        Returns:
            A new NFA with only the necessary states
        """
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
                    if isinstance(destinations, set):  # NFA case
                        for dst in destinations:
                            if dst.id not in reachable:
                                frontier.add(dst.id)
                    else:  # DFA case (single state)
                        dst = destinations
                        if dst.id not in reachable:
                            frontier.add(dst.id)

        # Find states that can reach an accepting state
        can_accept = set(state.id for state in self.accepting_states)
        old_size = -1

        while len(can_accept) > old_size:
            old_size = len(can_accept)

            for (src, _), destinations in self.transitions.items():
                if isinstance(destinations, set):  # NFA case
                    if any(dst.id in can_accept for dst in destinations):
                        can_accept.add(src.id)
                else:  # DFA case (single state)
                    if destinations.id in can_accept:
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
                if isinstance(destinations, set):  # NFA case
                    for dst in destinations:
                        if dst.id in keep_states:
                            result.set_transition(
                                old_to_new[src.id], symbol, old_to_new[dst.id]
                            )
                else:  # DFA case (single state)
                    dst = destinations
                    if dst.id in keep_states:
                        result.set_transition(
                            old_to_new[src.id], symbol, old_to_new[dst.id]
                        )

        return result

    def determinize(self):
        """
        Convert this NFA to a deterministic FSA using the subset construction.

        The subset construction:
        1. Creates states in the new DFA corresponding to sets of states in the NFA
        2. The initial state of the DFA corresponds to the set containing the NFA's initial state
        3. A state in the DFA is accepting if any of the corresponding NFA states is accepting
        4. Transitions are computed by following all possible NFA transitions and grouping by symbol

        Returns:
            A new deterministic FSA that accepts the same language
        """
        from automathic.fsa.fsa import FiniteStateAutomaton

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

        # Process subsets of states using breadth-first search
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
                            if isinstance(destinations, set):
                                next_states.update(dst.id for dst in destinations)
                            else:
                                next_states.add(destinations.id)

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
        for (src, symbol), dsts in result.transitions.items():
            for dst in dsts:
                if src.id < result.num_states and dst.id < result.num_states:
                    final_result.set_transition(src.id, symbol, dst.id)

        return final_result

    def minimize(self):
        """
        Minimize this NFA by first determinizing it and then minimizing the resulting DFA.

        For NFAs, minimization requires first converting to a DFA.

        Returns:
            A minimized DFA that accepts the same language
        """
        # First convert the NFA to a DFA
        dfa = self.determinize()

        # Then minimize the DFA
        return dfa.minimize()

    def intersect(self, other):
        """
        Create a product automaton that accepts the intersection of the languages.

        The product construction:
        1. Creates states that are pairs of states from the two input automata
        2. A state (q1, q2) is accepting if both q1 and q2 are accepting
        3. Transitions are defined by following both automata simultaneously

        Args:
            other: Another NonDeterministicFSA or FiniteStateAutomaton

        Returns:
            A new automaton that accepts the intersection of the languages
        """
        from automathic.fsa.fsa import FiniteStateAutomaton

        # For NFAs, convert to DFAs first for more efficient intersection
        if not isinstance(self, FiniteStateAutomaton):
            self_dfa = self.determinize()
        else:
            self_dfa = self

        if not isinstance(other, FiniteStateAutomaton):
            other_dfa = other.determinize()
        else:
            other_dfa = other

        # Check if alphabets match
        if set(self_dfa.alphabet) != set(other_dfa.alphabet):
            raise ValueError("Automata must have the same alphabet for intersection")

        # Create a product automaton where states are pairs (q1, q2)
        num_states = self_dfa.num_states * other_dfa.num_states
        result = FiniteStateAutomaton(num_states, self_dfa.alphabet)

        # Initialize state mapping and state counter
        state_map = {}  # Maps (q1, q2) to new state ID
        counter = 0

        # Set up the initial state
        if self_dfa.initial_state is not None and other_dfa.initial_state is not None:
            state_pair = (self_dfa.initial_state.id, other_dfa.initial_state.id)
            state_map[state_pair] = counter

            # Create combined origin for the state
            origin1 = self_dfa.states[state_pair[0]].origin
            origin2 = other_dfa.states[state_pair[1]].origin
            combined_origin = (f"({origin1},{origin2})",)
            result.states[counter] = State(counter, origin=combined_origin)

            result.set_initial_state(counter)
            counter += 1

        # Process state pairs using a breadth-first approach
        queue = [state_pair for state_pair in state_map.keys()]
        processed = set()

        while queue:
            current_pair = queue.pop(0)
            if current_pair in processed:
                continue

            processed.add(current_pair)
            current_id = state_map[current_pair]
            q1, q2 = current_pair

            # For each symbol in the alphabet
            for symbol in self_dfa.alphabet:
                next_state1 = self_dfa.transition(q1, symbol)
                next_state2 = other_dfa.transition(q2, symbol)

                # Only create a transition if both automata can move
                if next_state1 is not None and next_state2 is not None:
                    next_pair = (next_state1.id, next_state2.id)

                    # Create a new state if we haven't seen this pair before
                    if next_pair not in state_map:
                        state_map[next_pair] = counter

                        # Create combined origin for the state
                        origin1 = self_dfa.states[next_pair[0]].origin
                        origin2 = other_dfa.states[next_pair[1]].origin
                        combined_origin = (f"({origin1},{origin2})",)
                        result.states[counter] = State(counter, origin=combined_origin)

                        counter += 1
                        queue.append(next_pair)

                    # Add the transition
                    result.set_transition(current_id, symbol, state_map[next_pair])

        # Set accepting states - a state is accepting if both component states are accepting
        for (q1, q2), state_id in state_map.items():
            if (
                self_dfa.states[q1] in self_dfa.accepting_states
                and other_dfa.states[q2] in other_dfa.accepting_states
            ):
                result.set_accepting_state(state_id)

        # Create a new FSA with just the states we've actually used
        final_result = FiniteStateAutomaton(counter, self_dfa.alphabet)

        # Copy all the used states
        for i in range(counter):
            final_result.states[i] = result.states[i]

        # Copy initial state
        if result.initial_state:
            final_result.set_initial_state(result.initial_state.id)

        # Copy accepting states
        for state in result.accepting_states:
            if state.id < counter:
                final_result.set_accepting_state(state.id)

        # Copy transitions
        for (src, symbol), dsts in result.transitions.items():
            for dst in dsts:
                if src.id < counter and dst.id < counter:
                    final_result.set_transition(src.id, symbol, dst.id)

        return final_result.trim()

    def complement(self):
        """
        Create an automaton that accepts exactly the strings rejected by self.
        For NFAs, this requires determinization first.

        Returns:
            A new FiniteStateAutomaton that accepts the complement language
        """
        # First convert the NFA to a DFA
        dfa = self.determinize()

        # Then complement the DFA
        return dfa.complement()

    def reindex(self):
        """
        Reindex the states so they are numbered sequentially starting from 0.
        This is useful after operations that may have created gaps in state IDs.

        Returns:
            A new automaton with reindexed states
        """
        # If no states, return an empty automaton
        if self.num_states == 0:
            return self.__class__(0, self.alphabet)

        # Create a new automaton
        result = self.__class__(self.num_states, self.alphabet)

        # Copy all states with sequential IDs
        old_to_new = {}
        new_id = 0

        for old_id, state in enumerate(self.states):
            if state is None:
                continue

            old_to_new[old_id] = new_id
            result.states[new_id] = State(new_id, origin=None, label=None)
            new_id += 1

        # Set initial state
        if self.initial_state:
            result.set_initial_state(old_to_new[self.initial_state.id])

        # Set accepting states
        for state in self.accepting_states:
            result.set_accepting_state(old_to_new[state.id])

        # Copy transitions
        for (src, symbol), dsts in self.transitions.items():
            for d in dsts:
                result.set_transition(old_to_new[src.id], symbol, old_to_new[d.id])

        # Update the number of states
        result.num_states = new_id

        return result

    def is_equivalent(self, other):
        """
        Check if this automaton accepts the same language as another automaton.

        Two automata are equivalent if they accept exactly the same strings.
        This is implemented by checking if their symmetric difference
        (union of A∩B̄ and Ā∩B) accepts any string.

        Args:
            other: Another NonDeterministicFSA or FiniteStateAutomaton

        Returns:
            bool: True if both automata accept exactly the same language
        """
        from automathic.fsa.fsa import FiniteStateAutomaton

        # Check if alphabets match
        if set(self.alphabet) != set(other.alphabet):
            return False

        # For NFAs, we need to determinize first for more efficient operations
        self_dfa = (
            self.determinize() if not isinstance(self, FiniteStateAutomaton) else self
        )
        other_dfa = (
            other.determinize()
            if not isinstance(other, FiniteStateAutomaton)
            else other
        )

        # Compute the automaton accepting the symmetric difference
        # (L(self) ∩ L(other)ᶜ) ∪ (L(self)ᶜ ∩ L(other))
        # This is equivalent to: (L(self) ∪ L(other)) - (L(self) ∩ L(other))

        # Get complements
        self_complement = self_dfa.complement()
        other_complement = other_dfa.complement()

        # Get (L(self) ∩ L(other)ᶜ) - automaton accepting strings in self but not in other
        diff_1 = self_dfa.intersect(other_complement)

        # Get (L(self)ᶜ ∩ L(other)) - automaton accepting strings in other but not in self
        diff_2 = self_complement.intersect(other_dfa)

        # Union of differences - implemented by first converting to NFAs, then combining transitions
        union = NonDeterministicFSA(
            diff_1.num_states + diff_2.num_states + 1, self.alphabet
        )

        # Create a new initial state
        union.set_initial_state(0)

        # Copy states from diff_1, shifting IDs by 1
        for i in range(diff_1.num_states):
            state_obj = diff_1.states[i]
            # Create a new state with the same properties but new ID
            new_id = i + 1
            union.states[new_id] = State(
                new_id, origin=state_obj.origin, label=state_obj.label
            )

            # If this was an accepting state in diff_1, make it accepting in the union
            if state_obj in diff_1.accepting_states:
                union.set_accepting_state(new_id)

            # Add epsilon transition from union's initial state to this state
            if diff_1.initial_state and diff_1.initial_state.id == i:
                # In NFAs, we add all transitions from initial state to the new state's transitions
                for symbol, next_id in diff_1.get_transitions(i):
                    union.set_transition(new_id, symbol, next_id + 1)

        # Copy states from diff_2, shifting IDs by diff_1.num_states + 1
        offset = diff_1.num_states + 1
        for i in range(diff_2.num_states):
            state_obj = diff_2.states[i]
            # Create a new state with the same properties but new ID
            new_id = i + offset
            union.states[new_id] = State(
                new_id, origin=state_obj.origin, label=state_obj.label
            )

            # If this was an accepting state in diff_2, make it accepting in the union
            if state_obj in diff_2.accepting_states:
                union.set_accepting_state(new_id)

            # Add epsilon transition from union's initial state to this state
            if diff_2.initial_state and diff_2.initial_state.id == i:
                # In NFAs, we add all transitions from initial state to the new state's transitions
                for symbol, next_id in diff_2.get_transitions(i):
                    union.set_transition(new_id, symbol, next_id + offset)

        # Add transitions from diff_1, shifting all state IDs by 1
        for (src, symbol), destinations in diff_1.transitions.items():
            if isinstance(destinations, set):  # NFA case
                for dst in destinations:
                    union.set_transition(src.id + 1, symbol, dst.id + 1)
            else:  # DFA case
                union.set_transition(src.id + 1, symbol, destinations.id + 1)

        # Add transitions from diff_2, shifting all state IDs by offset
        for (src, symbol), destinations in diff_2.transitions.items():
            if isinstance(destinations, set):  # NFA case
                for dst in destinations:
                    union.set_transition(src.id + offset, symbol, dst.id + offset)
            else:  # DFA case
                union.set_transition(src.id + offset, symbol, destinations.id + offset)

        # Add transitions from initial state to both automata's initial states
        if diff_1.initial_state is not None:
            for symbol, next_id in diff_1.get_transitions(diff_1.initial_state):
                union.set_transition(0, symbol, next_id + 1)

        if diff_2.initial_state is not None:
            for symbol, next_id in diff_2.get_transitions(diff_2.initial_state):
                union.set_transition(0, symbol, next_id + offset)

        # The automata are equivalent if their symmetric difference accepts no strings
        # This is true if the resulting automaton has no reachable accepting states
        minimized = union.trim()
        return len(minimized.accepting_states) == 0

    def union(self, other):
        """
        Create a new NFA that accepts the union of the languages of self and other.

        The union construction:
        1. Creates a new initial state with epsilon transitions to the initial states of both NFAs
        2. Combines the states and transitions of both NFAs

        Args:
            other: Another NonDeterministicFSA

        Returns:
            A new NonDeterministicFSA that accepts the union of the languages
        """
        if set(self.alphabet) != set(other.alphabet):
            raise ValueError("Automata must have the same alphabet for union")

        # Create a new NFA with combined states
        num_states = self.num_states + other.num_states + 1
        result = NonDeterministicFSA(num_states, self.alphabet)

        # Create a new initial state
        result.set_initial_state(0)

        # Copy states from self, shifting IDs by 1
        for i in range(self.num_states):
            state_obj = self.states[i]
            new_id = i + 1
            result.states[new_id] = State(
                new_id, origin=state_obj.origin, label=state_obj.label
            )
            if state_obj in self.accepting_states:
                result.set_accepting_state(new_id)

        # Copy states from other, shifting IDs by self.num_states + 1
        offset = self.num_states + 1
        for i in range(other.num_states):
            state_obj = other.states[i]
            new_id = i + offset
            result.states[new_id] = State(
                new_id, origin=state_obj.origin, label=state_obj.label
            )
            if state_obj in other.accepting_states:
                result.set_accepting_state(new_id)

        # Add epsilon transitions from the new initial state to the initial states of both NFAs
        if self.initial_state is not None:
            result.set_transition(0, "", self.initial_state.id + 1)
        if other.initial_state is not None:
            result.set_transition(0, "", other.initial_state.id + offset)

        # Copy transitions from self, shifting IDs by 1
        for (src, symbol), destinations in self.transitions.items():
            if isinstance(destinations, set):
                for dst in destinations:
                    result.set_transition(src.id + 1, symbol, dst.id + 1)
            else:
                result.set_transition(src.id + 1, symbol, destinations.id + 1)

        # Copy transitions from other, shifting IDs by offset
        for (src, symbol), destinations in other.transitions.items():
            if isinstance(destinations, set):
                for dst in destinations:
                    result.set_transition(src.id + offset, symbol, dst.id + offset)
            else:
                result.set_transition(src.id + offset, symbol, destinations.id + offset)

        return result

    def __str__(self):
        return self.ascii()

    def ascii(self):
        """Generate a text-based visualization of the NFA"""
        lines = []

        # Title
        if self.num_states == 0:
            lines.append("Empty NFA")
            return "\n".join(lines)

        lines.append(f"NFA with {self.num_states} states")

        # States
        lines.append("\nSTATES:")
        for q in range(self.num_states):
            state = self.states[q]
            if state is None:
                continue

            state_desc = f"{q}: {state.label}"
            if state == self.initial_state:
                state_desc += " (initial)"
            if state in self.accepting_states:
                state_desc += " (accepting)"
            lines.append(state_desc)

        # Transition table
        lines.append("\nTRANSITION TABLE:")
        max_sym_len = 25
        format_str = f" {{:^{max_sym_len}}} |"

        # Create header with properly formatted symbol strings
        header = "State                      |" + "".join(
            format_str.format(str(sym)) for sym in self.alphabet
        )
        lines.append(header)
        lines.append("-" * len(header))

        # Add transitions
        for q in range(self.num_states):
            state = self.states[q]
            if state is None:
                continue

            # Prefix for state (marking initial/accepting)
            prefix = ""
            if state == self.initial_state:
                prefix += ">"
            if state in self.accepting_states:
                prefix += "*"
            if state != self.initial_state and state not in self.accepting_states:
                prefix += " "

            row = f"{prefix}{state.label:3} |"

            # Add each transition - for NFA, can have multiple destinations
            for symbol in self.alphabet:
                destinations = set()
                if (state, symbol) in self.transitions:
                    destinations = self.transitions[(state, symbol)]

                if destinations:
                    dest_str = ",".join(str(dst.label) for dst in destinations)
                    if len(dest_str) > max_sym_len - 2:
                        dest_str = dest_str[: max_sym_len - 5] + "..."
                    cell = f" {dest_str:^{max_sym_len}} |"
                else:
                    cell = f" {'-':^{max_sym_len}} |"
                row += cell

            lines.append(row)

        return "\n".join(lines)

    def _repr_html_(self):
        """
        When returned from a Jupyter cell, this will generate the FSA visualization
        with distinct colors and thicker borders for accepting states
        """
        import json
        from collections import defaultdict
        from uuid import uuid4

        def _get_state_label(state):
            if len(state.origin) == 1:  # Single origin, use the label
                return str(state.id).replace('"', '\\"').replace("'", "")
            else:  # Multiple origins, use the origin tuple
                return str(state.label).replace('"', '\\"').replace("'", "")

        ret = []
        if self.num_states == 0:
            return "<code>Empty FSA</code>"

        if self.num_states > 64:
            return (
                "FSA too large to draw graphic, use fsa.ascii()<br />"
                + f"<code>FSA(states={self.num_states})</code>"
            )

        # Define color schemes based on theme
        if hasattr(self, "theme") and self.theme == "dark":
            colors = {
                "normal": "4c566a",  # Dark blue-gray
                "initial": "88c0d0",  # Bright blue
                "accepting": "bf616a",  # Soft red
                "initial_accepting": "ebcb8b",  # Gold
            }
            stroke_color = "rgb(192, 192, 192)"  # Light gray for dark theme
            text_color = "#ffffff"  # White text for dark theme
        else:
            colors = {
                "normal": "e9f0f7",  # Light blue
                "initial": "9fd3c7",  # Teal green
                "accepting": "ffcab0",  # Soft coral
                "initial_accepting": "ecdfc8",  # Beige
            }
            stroke_color = "#333"  # Dark gray for light theme
            text_color = "#000000"  # Black text for light theme

        # Add all states with explicit labels
        for q in self.states:
            if q is None:
                continue

            # Explicitly convert the label to string
            node_label = _get_state_label(q)

            # Determine node style - use stroke-width for accepting states
            if q == self.initial_state and q in self.accepting_states:
                color = colors["initial_accepting"]
                border = 3  # Thicker border for accepting states
            elif q == self.initial_state:
                color = colors["initial"]
                border = 1  # Normal border
            elif q in self.accepting_states:
                color = colors["accepting"]
                border = 3  # Thicker border for accepting states
            else:
                color = colors["normal"]
                border = 1  # Normal border

            # Create node with proper styling
            ret.append(
                f'g.setNode("{q.id}", {{ '
                f'label: "{node_label}", '
                f'shape: "circle", '
                f'style: "fill: #{color}; stroke: {stroke_color}; stroke-width: {border}px;" '
                f"}});\n"
            )

        # Add edges for transitions
        for q in self.states:
            if q is None:
                continue

            to = defaultdict(list)
            for symbol, next_state_id in self.get_transitions(q):
                # Since get_transitions now returns state IDs (integers), we need to convert back to State objects
                next_state = (
                    self.states[next_state_id]
                    if isinstance(next_state_id, int)
                    else next_state_id
                )
                to[next_state_id].append(str(symbol))

            for d, values in to.items():
                # Use state ID for edge definition
                dest_state = self.states[d] if isinstance(d, int) else d
                # if len(values) > 6:
                #     values = values[0:3] + [". . ."]
                edge_label = ", ".join(values)
                ret.append(
                    f'g.setEdge("{q.id}", "{dest_state.id}", {{ '
                    f'label: "{edge_label}", '  # Use simple quotes
                    f'arrowhead: "vee", '
                    f'style: "stroke: {stroke_color}; fill: none;", '
                    f'labelStyle: "fill: {stroke_color};", '
                    f'arrowheadStyle: "fill: {stroke_color}; stroke: {stroke_color};" '
                    f"}});\n"
                )

        # Add a special invisible node and edge for initial state marker
        if self.initial_state is not None:
            ini_id = self.initial_state.id
            ret.append(
                f'g.setNode("start", {{ '
                f'label: "", '
                f"width: 0, "
                f"height: 0, "
                f'style: "opacity: 0" '
                f"}});\n"
            )
            ret.append(
                f'g.setEdge("start", "{ini_id}", {{ '
                f'label: "", '
                f'arrowhead: "normal", '
                f'style: "stroke: {stroke_color}; fill: none;", '
                f'arrowheadStyle: "fill: {stroke_color}; stroke: {stroke_color};" '
                f"}});\n"
            )

        # If the machine is too big, don't attempt to display it
        if len(ret) > 256:
            return (
                "FSA too large to draw graphic, use fsa.ascii()<br />"
                + f"<code>FSA(states={self.num_states})</code>"
            )

        # Build the HTML with embedded JavaScript
        ret2 = [
            """
        <script>
        try {
            require.config({
                paths: {
                    "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
                    "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
                }
            });
        } catch (e) {
            ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
            "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(
                function (src) {
                    var tag = document.createElement('script');
                    tag.src = src;
                    document.body.appendChild(tag);
                }
            );
        }
        try {
            requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
            require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        """
        ]

        # Add theme-specific styles
        ret2.append(
            f"""
        <style>
        /* Import LaTeX-like font */
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600&display=swap');
        
        .node rect,
        .node circle,
        .node ellipse {{
            stroke: {stroke_color};
            /* Don't set stroke-width here, it's applied per-node */
        }}
        
        .edgePath path {{
            stroke: {stroke_color};
            fill: {stroke_color};
            stroke-width: 1.5px;
        }}
        
        /* LaTeX-like typography */
        .node text {{
            font-family: 'Source Serif Pro', 'Computer Modern', 'Latin Modern Math', serif;
            font-size: 14px;
            font-weight: normal;
            fill: {text_color} !important; /* Force text color */
        }}
        
        .edgeLabel text {{
            font-family: 'Source Serif Pro', 'Computer Modern', 'Latin Modern Math', serif;
            font-size: 12px;
            font-weight: normal;
            fill: {stroke_color} !important; /* Force edge label color */
        }}
        </style>
        """
        )

        obj = "fsa_" + str(uuid4()).replace("-", "_")
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
        <script>
        (function render_d3() {
            var d3, dagreD3;
            try {
                d3 = require('d3');
                dagreD3 = require('dagreD3');
            } catch (e) {
                if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined"){
                    d3 = window.d3;
                    dagreD3 = window.dagreD3;
                } else {
                    setTimeout(render_d3, 50);
                    return;
                }
            }
            
            // Create directed graph
            var g = new dagreD3.graphlib.Graph().setGraph({
                rankdir: 'LR',
                marginx: 20,
                marginy: 20,
                ranksep: 50,
                nodesep: 30
            });
        """
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            f"""
            var inner = svg.select("g");
            
            // Set up zoom support
            var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {{
                inner.attr("transform", d3.event.transform);
            }});
            svg.call(zoom);
            
            // Create the renderer
            var render = new dagreD3.render();
            
            // Render the graph
            render(inner, g);
            
            // Hide start node and apply styles after rendering
            if (g.hasNode("start")) {{
                d3.select(g.node("start").elem).style("opacity", "0");
            }}
            
            // Force labels to be visible with correct colors
            inner.selectAll("g.node text").style("fill", "{text_color}");
            inner.selectAll(".edgeLabel text").style("fill", "{stroke_color}");
            
            // Center the graph
            var initialScale = 0.75;
            svg.call(zoom.transform, d3.zoomIdentity
                .translate((svg.attr("width") - g.graph().width * initialScale) / 2, 20)
                .scale(initialScale));
                
            // Adjust SVG height to fit graph
            svg.attr('height', g.graph().height * initialScale + 40);
        }})();
        </script>
        """
        )

        return "".join(ret2)
