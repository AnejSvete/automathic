from automathic.fsa.fsa import FiniteStateAutomaton


def create_substring_abc_automaton():
    """
    Creates an FSA that recognizes strings containing 'abc' as a substring.

    States:
    - 0: Initial state (haven't seen any pattern yet)
    - 1: Seen 'a'
    - 2: Seen 'ab'
    - 3: Seen 'abc' (accepting state)
    """
    # Create FSA with 4 states and alphabet {a,b,c}
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(4, alphabet)

    # Set up transitions
    # State 0 transitions
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(0, "a", 1)  # On input 'a', move to state 1
        else:
            fsa.set_transition(0, symbol, 0)  # On other inputs, stay in state 0

    # State 1 transitions
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(1, "b", 2)  # On input 'b', move to state 2
        elif symbol == "a":
            fsa.set_transition(1, "a", 1)  # On input 'a', stay in state 1
        else:
            fsa.set_transition(1, symbol, 0)  # On other inputs, back to state 0

    # State 2 transitions
    for symbol in alphabet:
        if symbol == "c":
            fsa.set_transition(2, "c", 3)  # On input 'c', move to accepting state 3
        elif symbol == "a":
            fsa.set_transition(2, "a", 1)  # On input 'a', go to state 1
        else:
            fsa.set_transition(2, symbol, 0)  # On other inputs, back to state 0

    # State 3 transitions (once in accepting state, stay there)
    for symbol in alphabet:
        fsa.set_transition(3, symbol, 3)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(3)

    return fsa


def create_end_with_ab_automaton():
    """
    Creates an FSA that recognizes strings ending with 'ab'.

    States:
    - 0: Initial state
    - 1: Seen 'a' as the potential start of 'ab'
    - 2: Seen 'ab' (accepting state)

    Returns:
        FiniteStateAutomaton: The automaton accepting strings ending with 'ab'
    """
    alphabet = ["a", "b"]
    fsa = FiniteStateAutomaton(3, alphabet)

    # Set transitions
    fsa.set_transition(0, "a", 1)  # On input 'a', move to state 1
    fsa.set_transition(0, "b", 0)  # On input 'b', stay in state 0

    fsa.set_transition(1, "a", 1)  # On input 'a', stay in state 1
    fsa.set_transition(1, "b", 2)  # On input 'b', move to accepting state 2

    fsa.set_transition(2, "a", 1)  # On input 'a', go to state 1
    fsa.set_transition(2, "b", 0)  # On input 'b', go to state 0

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(2)

    return fsa


def create_divisible_by_3_automaton():
    """
    Creates an FSA that recognizes binary strings representing numbers divisible by 3.

    States:
    - 0: Initial state (remainder 0)
    - 1: Remainder 1
    - 2: Remainder 2

    Returns:
        FiniteStateAutomaton: The automaton accepting binary strings divisible by 3
    """
    alphabet = ["0", "1"]
    fsa = FiniteStateAutomaton(3, alphabet)

    # Set transitions
    # State 0 (remainder 0)
    fsa.set_transition(0, "0", 0)  # 0 * 2 = 0 (mod 3)
    fsa.set_transition(0, "1", 1)  # 0 * 2 + 1 = 1 (mod 3)

    # State 1 (remainder 1)
    fsa.set_transition(1, "0", 2)  # 1 * 2 = 2 (mod 3)
    fsa.set_transition(1, "1", 0)  # 1 * 2 + 1 = 3 = 0 (mod 3)

    # State 2 (remainder 2)
    fsa.set_transition(2, "0", 1)  # 2 * 2 = 4 = 1 (mod 3)
    fsa.set_transition(2, "1", 2)  # 2 * 2 + 1 = 5 = 2 (mod 3)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)  # Only accept if remainder is 0

    return fsa


def create_contains_a_followed_by_b_automaton():
    """
    Creates an FSA that recognizes strings containing an 'a' followed by a 'b'
    (not necessarily immediately).

    States:
    - 0: Initial state (haven't seen 'a' yet)
    - 1: Seen at least one 'a', waiting for 'b'
    - 2: Seen 'a' followed by 'b' (accepting state)

    Returns:
        FiniteStateAutomaton: The automaton accepting strings with 'a' followed by 'b'
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(3, alphabet)

    # State 0 transitions
    fsa.set_transition(0, "a", 1)  # On seeing 'a', move to state 1
    fsa.set_transition(0, "b", 0)  # On seeing 'b', stay in state 0
    fsa.set_transition(0, "c", 0)  # On seeing 'c', stay in state 0

    # State 1 transitions
    fsa.set_transition(1, "a", 1)  # On seeing 'a', stay in state 1
    fsa.set_transition(1, "b", 2)  # On seeing 'b', move to accepting state 2
    fsa.set_transition(1, "c", 1)  # On seeing 'c', stay in state 1

    # State 2 transitions (once in accepting state, stay there)
    for symbol in alphabet:
        fsa.set_transition(2, symbol, 2)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(2)

    return fsa


def create_even_parity_automaton():
    """
    Creates an FSA that recognizes binary strings with even parity (even number of 1s).

    States:
    - 0: Initial state (even number of 1s seen so far)
    - 1: Odd number of 1s seen so far

    Returns:
        FiniteStateAutomaton: The automaton accepting strings with even parity
    """
    alphabet = ["0", "1"]
    fsa = FiniteStateAutomaton(2, alphabet)

    # State 0 transitions (even parity)
    fsa.set_transition(0, "0", 0)  # 0 doesn't change parity
    fsa.set_transition(0, "1", 1)  # 1 flips parity to odd

    # State 1 transitions (odd parity)
    fsa.set_transition(1, "0", 1)  # 0 doesn't change parity
    fsa.set_transition(1, "1", 0)  # 1 flips parity back to even

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)  # Only accept if we've seen an even number of 1s

    # Set state labels for clarity
    fsa.set_state_label(0, "Even")
    fsa.set_state_label(1, "Odd")

    return fsa


def create_even_length_automaton():
    """
    Creates an FSA that recognizes strings of even length over any alphabet.

    States:
    - 0: Initial state (even length seen so far - including empty string)
    - 1: Odd length seen so far

    Returns:
        FiniteStateAutomaton: The automaton accepting strings of even length
    """
    alphabet = ["a", "b", "c"]  # Can be any alphabet
    fsa = FiniteStateAutomaton(2, alphabet)

    # Each symbol read toggles between even and odd length
    for symbol in alphabet:
        # State 0 transitions (even length)
        fsa.set_transition(0, symbol, 1)  # Any symbol makes length odd

        # State 1 transitions (odd length)
        fsa.set_transition(1, symbol, 0)  # Any symbol makes length even again

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)  # Accept only strings of even length

    # Set state labels for clarity
    fsa.set_state_label(0, "Even")
    fsa.set_state_label(1, "Odd")

    return fsa


def create_even_number_of_as_automaton():
    """
    Creates an FSA that recognizes strings with an even number of 'a's.

    States:
    - 0: Initial state (even number of 'a's seen so far)
    - 1: Odd number of 'a's seen so far

    Returns:
        FiniteStateAutomaton: The automaton accepting strings with even number of 'a's
    """
    alphabet = ["a", "b", "c"]  # Can include any symbols besides 'a'
    fsa = FiniteStateAutomaton(2, alphabet)

    # State 0 transitions (even number of 'a's)
    fsa.set_transition(0, "a", 1)  # 'a' makes count odd
    fsa.set_transition(0, "b", 0)  # Other symbols don't affect count
    fsa.set_transition(0, "c", 0)

    # State 1 transitions (odd number of 'a's)
    fsa.set_transition(1, "a", 0)  # 'a' makes count even again
    fsa.set_transition(1, "b", 1)  # Other symbols don't affect count
    fsa.set_transition(1, "c", 1)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)  # Only accept if we've seen an even number of 'a's

    # Set state labels for clarity
    fsa.set_state_label(0, "Even-a")
    fsa.set_state_label(1, "Odd-a")

    return fsa
