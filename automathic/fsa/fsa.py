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
        self.transitions[(src_state, symbol)] = {dst_state}

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
        dst = self.transitions.get((state_obj, symbol), None)
        return next(iter(dst)) if dst else None

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

        # Handle the case of empty FSAs
        if self.num_states == 0 or self.initial_state is None:
            # Create a single-state DFA that accepts all strings
            result = FiniteStateAutomaton(1, self.alphabet)
            result.set_initial_state(0)
            result.set_accepting_state(0)

            # Add transitions from the state to itself for all symbols
            for symbol in self.alphabet:
                result.set_transition(0, symbol, 0)

            return result

        # Make sure the DFA is complete
        complete_dfa = self.complete()

        # Create a new DFA with the same structure but complemented accepting states
        result = FiniteStateAutomaton(complete_dfa.num_states, complete_dfa.alphabet)

        # Copy all transitions
        for (src, symbol), dst in complete_dfa.transitions.items():
            result.set_transition(src.id, symbol, next(iter(dst)).id)

        # Copy initial state
        if complete_dfa.initial_state is not None:
            result.set_initial_state(complete_dfa.initial_state.id)

        # Complement accepting states
        for state in complete_dfa.states:
            if state not in complete_dfa.accepting_states:
                result.set_accepting_state(state.id)

        return result

    def cascade_product(self, other, feedback_function=None):
        """
        Compute the cascade product of this DFA with another DFA as used in Krohn-Rhodes decomposition.

        In the Krohn-Rhodes theory, a cascade product allows the second automaton's transitions
        to depend on both the input symbol and the current state of the first automaton.

        Args:
            other (FiniteStateAutomaton): The second automaton in the cascade
            feedback_function (callable, optional): A function that takes (q1, a) where q1 is a state
                from self and a is an input symbol, and returns a symbol for the second automaton.
                If None, the same input symbol is used for both automata.

        Returns:
            FiniteStateAutomaton: A new FSA representing the cascade product
        """
        # Create a new FSA with states representing pairs of states from both FSAs
        result = FiniteStateAutomaton(self.num_states * other.num_states, self.alphabet)

        # Use identity function if no feedback function is provided
        if feedback_function is None:
            feedback_function = lambda q, a: a

        # Map pairs of states to new state IDs
        state_map = {}
        for i in range(self.num_states):
            for j in range(other.num_states):
                new_id = i * other.num_states + j
                state_map[(i, j)] = new_id

                # Set the combined origins for debugging/visualization
                first_origin = self.states[i].origin
                second_origin = other.states[j].origin
                combined_origin = first_origin + second_origin
                result.states[new_id] = State(new_id, origin=combined_origin)

        # Set initial state
        if self.initial_state is not None and other.initial_state is not None:
            initial_pair = (self.initial_state.id, other.initial_state.id)
            result.set_initial_state(state_map[initial_pair])

        # Set accepting states (typically defined by the final state of the second automaton)
        for i in range(self.num_states):
            for j in range(other.num_states):
                if other.states[j] in other.accepting_states:
                    new_id = state_map[(i, j)]
                    result.set_accepting_state(new_id)

        # Set transitions - the key part of a cascade product
        for i in range(self.num_states):
            first_state = self.states[i]
            for j in range(other.num_states):
                second_state = other.states[j]
                for symbol in self.alphabet:
                    # First machine processes the input directly
                    next_first = self.transition(first_state, symbol)

                    if next_first is not None:
                        # Second machine's input depends on first machine's state and the input
                        modified_symbol = feedback_function(first_state, symbol)

                        # Check if the modified symbol is in the second automaton's alphabet
                        if modified_symbol in other.alphabet:
                            next_second = other.transition(
                                second_state, modified_symbol
                            )

                            if next_second is not None:
                                src_id = state_map[(i, j)]
                                dst_id = state_map[(next_first.id, next_second.id)]
                                result.set_transition(src_id, symbol, dst_id)

        return result

    def syntactic_monoid(self):
        """
        Constructs the syntactic monoid of this finite state automaton.

        The syntactic monoid represents transformations on the states induced by input strings:
        - Each element of the monoid is an equivalence class of strings that induce the same
        state transformation
        - The operation is function composition (corresponding to string concatenation)
        - The identity element is the empty string

        This implementation:
        1. Creates initial transformations for each alphabet symbol
        2. Iteratively composes them until closure is reached
        3. Returns a Monoid object from the algebra package

        Returns:
            A Monoid object representing the syntactic monoid of the FSA
        """
        from automathic.algebra.algebra import make_finite_algebra

        # Ensure the FSA is complete and minimal
        complete_fsa = self.minimize().complete()
        n_states = complete_fsa.num_states

        # Dictionary to store transformations (as tuples) and their representative strings
        transformations = {}

        # Start with the identity transformation (empty string)
        identity_transform = tuple(range(n_states))
        transformations[""] = identity_transform

        # Add transformations for each individual symbol
        for symbol in complete_fsa.alphabet:
            transformation = []
            for state_id in range(n_states):
                next_state = complete_fsa.transition(state_id, symbol)
                transformation.append(next_state.id)
            if tuple(transformation) not in transformations.values():
                transformations[symbol] = tuple(transformation)

        # Keep track of which transformations we've composed
        composed_pairs = set()

        # Compose transformations until we reach closure
        changed = True
        while changed:
            changed = False

            # Get all current transformations
            current_transforms = list(transformations.items())

            # Try to compose each pair of transformations
            for string1, transform1 in current_transforms:
                for string2, transform2 in current_transforms:
                    # Skip if we've already tried this composition
                    if (string1, string2) in composed_pairs:
                        continue

                    # Mark this pair as composed
                    composed_pairs.add((string1, string2))

                    # Concatenate the strings
                    new_string = string1 + string2

                    # Compose the transformations (t1 ∘ t2)
                    composed_transform = tuple(transform1[i] for i in transform2)

                    # If this produces a new transformation, add it
                    if composed_transform not in transformations.values():
                        # Find the shortest string that produces this transformation
                        for s, t in transformations.items():
                            if t == composed_transform and len(s) < len(new_string):
                                new_string = s
                                break

                        transformations[new_string] = composed_transform
                        changed = True

        # Get unique transformations and assign indices
        unique_transformations = {}
        for string, transform in transformations.items():
            # Only keep the shortest string for each unique transformation
            existing_transform = None
            for t, idx in unique_transformations.items():
                if t == transform:
                    existing_transform = t
                    break

            if existing_transform is None or len(string) < len(existing_transform[0]):
                if existing_transform is not None:
                    del unique_transformations[existing_transform]
                unique_transformations[(transform, string)] = len(
                    unique_transformations
                )

        # Create elements list using representative strings
        elements = [""] * len(unique_transformations)
        for (transform, string), idx in unique_transformations.items():
            elements[idx] = string if string else "ε"  # Use ε for empty string

        # Create the multiplication table
        n_elements = len(unique_transformations)
        mult_table = [[0 for _ in range(n_elements)] for _ in range(n_elements)]

        # Fill in the multiplication table
        for (t1, _), i in unique_transformations.items():
            for (t2, _), j in unique_transformations.items():
                # Compose the transformations
                composed = tuple(t1[state] for state in t2)

                # Find the index of the resulting transformation
                for (t, _), k in unique_transformations.items():
                    if t == composed:
                        mult_table[i][j] = k
                        break

        # Construct the monoid
        name = f"SyntacticMonoid({self.name if hasattr(self, 'name') else 'FSA'})"
        description = f"Syntactic monoid of the given finite state automaton"

        return make_finite_algebra(name, description, elements, mult_table)

    def copy(self):
        """
        Create a deep copy of this DFA.

        Returns:
            A new FiniteStateAutomaton identical to this one
        """
        result = FiniteStateAutomaton(self.num_states, self.alphabet)

        # Copy transitions
        for (src, symbol), dst in self.transitions.items():
            result.set_transition(src.id, symbol, next(iter(dst)).id)

        # Copy initial state
        if self.initial_state is not None:
            result.set_initial_state(self.initial_state.id)

        # Copy accepting states
        for state in self.accepting_states:
            result.set_accepting_state(state.id)

        return result
