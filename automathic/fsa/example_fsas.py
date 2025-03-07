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


def create_odd_parity_automaton():
    """
    Creates an FSA that recognizes binary strings with odd parity (odd number of 1s).

    States:
    - 0: Initial state (odd number of 1s seen so far)
    - 1: Odd number of 1s seen so far

    Returns:
        FiniteStateAutomaton: The automaton accepting strings with odd parity
    """
    alphabet = ["0", "1"]
    fsa = FiniteStateAutomaton(2, alphabet)

    # State 0 transitions (odd parity)
    fsa.set_transition(0, "0", 0)  # 0 doesn't change parity
    fsa.set_transition(0, "1", 1)  # 1 flips parity to odd

    # State 1 transitions (odd parity)
    fsa.set_transition(1, "0", 1)  # 0 doesn't change parity
    fsa.set_transition(1, "1", 0)  # 1 flips parity back to even

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(1)  # Only accept if we've seen an odd number of 1s

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


def create_sigma_star_automaton():
    """
    Creates an FSA that recognizes any string over the alphabet {a, b, c}.

    States:
    - 0: Initial state (can accept any string)

    Returns:
        FiniteStateAutomaton: The automaton accepting any string over {a, b, c}
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(1, alphabet)

    # Any symbol read keeps the automaton in the accepting state
    for symbol in alphabet:
        fsa.set_transition(0, symbol, 0)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)

    return fsa


def create_starts_with_a_and_ends_with_b_automaton():
    """
    Creates an FSA that recognizes strings starting with 'a' and ending with 'b'.
    Corresponds to formula φ_0.

    States:
    - 0: Initial state (start of string)
    - 1: Have seen 'a' as first symbol, currently in middle of string
    - 2: Have seen 'a' as first symbol, and ending with 'b' (accepting state)
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(3, alphabet)

    # State 0 transitions (initial state)
    fsa.set_transition(0, "a", 1)  # If first symbol is 'a', move to state 1
    for symbol in ["b", "c"]:
        # If first symbol is not 'a', string can't be accepted
        # We'll create transitions to make the automaton complete
        fsa.set_transition(0, symbol, 0)

    # State 1 transitions (seen 'a' at start, waiting for end)
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(1, "b", 2)  # If we see 'b', might be the end
        else:
            fsa.set_transition(1, symbol, 1)  # Otherwise, stay in middle state

    # State 2 transitions (have seen 'a' at start and 'b' at current position)
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(2, "b", 2)  # If we see another 'b', update last 'b'
        else:
            fsa.set_transition(
                2, symbol, 1
            )  # If we see non-'b', go back to middle state

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(2)

    # Set state labels
    fsa.set_state_label(0, "Start")
    fsa.set_state_label(1, "Seen 'a' first")
    fsa.set_state_label(2, "Ends with 'b'")

    return fsa


def create_starts_with_a_automaton():
    """
    Creates an FSA that recognizes strings starting with 'a'.
    Corresponds to formula φ_1.

    States:
    - 0: Initial state
    - 1: Seen 'a' as first symbol (accepting state)
    - 2: Didn't start with 'a' (sink state, non-accepting)
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(3, alphabet)

    # State 0 transitions (initial state)
    fsa.set_transition(0, "a", 1)  # If first symbol is 'a', accept
    for symbol in ["b", "c"]:
        fsa.set_transition(0, symbol, 2)  # If first symbol is not 'a', reject

    # State 1 transitions (accepting state - stay here)
    for symbol in alphabet:
        fsa.set_transition(1, symbol, 1)

    # State 2 transitions (sink state - stay here)
    for symbol in alphabet:
        fsa.set_transition(2, symbol, 2)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(1)

    # Set state labels
    fsa.set_state_label(0, "Start")
    fsa.set_state_label(1, "Starts with 'a'")
    fsa.set_state_label(2, "Not starting with 'a'")

    return fsa


def create_ends_with_b_automaton():
    """
    Creates an FSA that recognizes strings ending with 'b'.
    Corresponds to formula φ_2.

    States:
    - 0: Initial state (haven't seen 'b' as last symbol)
    - 1: Seen 'b' as potentially last symbol (accepting state)
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(2, alphabet)

    # State 0 transitions (haven't seen 'b' as last symbol)
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(0, "b", 1)  # If we see 'b', it might be the last
        else:
            fsa.set_transition(0, symbol, 0)  # For non-'b', stay in state 0

    # State 1 transitions (seen 'b' as potentially last symbol)
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(1, "b", 1)  # If we see another 'b', update last 'b'
        else:
            fsa.set_transition(1, symbol, 0)  # If we see non-'b', go back to state 0

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(1)

    # Set state labels
    fsa.set_state_label(0, "Not ending with 'b'")
    fsa.set_state_label(1, "Ending with 'b'")

    return fsa


def create_a_followed_by_b_automaton():
    """
    Creates an FSA that recognizes strings where every 'a' is immediately followed by 'b'.
    Corresponds to formula φ_3.

    States:
    - 0: Initial state (no 'a' seen, or all 'a's followed by 'b')
    - 1: Seen 'a', waiting for 'b' (non-accepting)
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(2, alphabet)

    # State 0 transitions (no 'a' seen or all 'a's followed by 'b')
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(0, "a", 1)  # If we see 'a', need 'b' next
        else:
            fsa.set_transition(0, symbol, 0)  # For non-'a', stay in state 0

    # State 1 transitions (seen 'a', waiting for 'b')
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(1, "b", 0)  # If we see 'b' after 'a', all good
        else:
            # If we see anything other than 'b', this violates the requirement
            # Since there's no transition defined, this will lead to rejection
            pass

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)

    # Set state labels
    fsa.set_state_label(0, "Valid")
    fsa.set_state_label(1, "Waiting for 'b'")

    return fsa


def create_no_aa_automaton():
    """
    Creates an FSA that recognizes strings where no 'a' is immediately followed by another 'a'.
    Corresponds to formula φ_4.

    States:
    - 0: Initial state (no 'a' seen yet, or last symbol wasn't 'a')
    - 1: Last symbol was 'a', cannot have another 'a' next
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(2, alphabet)

    # State 0 transitions (no 'a' seen yet or last symbol wasn't 'a')
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(0, "a", 1)  # If we see 'a', move to state 1
        else:
            fsa.set_transition(0, symbol, 0)  # For non-'a', stay in state 0

    # State 1 transitions (last symbol was 'a')
    for symbol in alphabet:
        if symbol == "a":
            # This would make 'aa', which is not allowed
            # No transition defined means rejection
            pass
        else:
            fsa.set_transition(1, symbol, 0)  # For non-'a', go back to state 0

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)
    fsa.set_accepting_state(1)

    # Set state labels
    fsa.set_state_label(0, "No 'a' or after non-'a'")
    fsa.set_state_label(1, "After 'a'")

    return fsa


def create_exactly_one_a_automaton():
    """
    Creates an FSA that recognizes strings containing exactly one 'a'.
    Corresponds to formula φ_5.

    States:
    - 0: Initial state (no 'a' seen yet)
    - 1: Seen exactly one 'a' (accepting state)
    - 2: Seen more than one 'a' (sink state, non-accepting)
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(3, alphabet)

    # State 0 transitions (no 'a' seen yet)
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(0, "a", 1)  # First 'a', move to accepting state
        else:
            fsa.set_transition(0, symbol, 0)  # For non-'a', stay in state 0

    # State 1 transitions (seen exactly one 'a')
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(1, "a", 2)  # Second 'a' seen, move to rejecting state
        else:
            fsa.set_transition(1, symbol, 1)  # For non-'a', stay in accepting state

    # State 2 transitions (seen more than one 'a')
    for symbol in alphabet:
        fsa.set_transition(2, symbol, 2)  # Stay in rejecting state

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(1)

    # Set state labels
    fsa.set_state_label(0, "No 'a' seen")
    fsa.set_state_label(1, "Exactly one 'a'")
    fsa.set_state_label(2, "More than one 'a'")

    return fsa


def create_a_preceded_by_b_automaton():
    """
    Creates an FSA that recognizes strings where every 'a' is preceded by a 'b'.
    Corresponds to formula φ_6.

    States:
    - 0: Initial state (haven't seen any symbols, or last symbol was 'b')
    - 1: Last symbol was not 'b', so 'a' is not allowed next
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(2, alphabet)

    # Special case for empty string (it's accepted)
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)

    # State 0 transitions (initial state or after 'b')
    for symbol in alphabet:
        if symbol == "a":
            # 'a' is allowed after 'b' or at the beginning
            fsa.set_transition(0, "a", 1)
        elif symbol == "b":
            fsa.set_transition(0, "b", 0)  # After 'b', stay in state 0
        else:
            fsa.set_transition(0, symbol, 1)  # After any other symbol, go to state 1

    # State 1 transitions (after non-'b')
    for symbol in alphabet:
        if symbol == "a":
            # 'a' is not allowed after a non-'b'
            # No transition defined means rejection
            pass
        elif symbol == "b":
            fsa.set_transition(1, "b", 0)  # After 'b', can have 'a' again
        else:
            fsa.set_transition(1, symbol, 1)  # After any other symbol, stay in state 1

    # Both states are accepting since what matters is each specific 'a' occurrence
    fsa.set_accepting_state(1)

    # Set state labels
    fsa.set_state_label(0, "After 'b' or start")
    fsa.set_state_label(1, "After non-'b'")

    return fsa


def create_alternating_ab_automaton():
    """
    Creates an FSA that recognizes strings where 'a' and 'b' alternate, starting with 'a'.
    Corresponds to formula φ_7.

    States:
    - 0: Initial state (expecting 'a')
    - 1: After 'a', expecting 'b'
    - 2: After 'b', expecting 'a' again
    - 3: Invalid pattern detected (sink state)
    """
    alphabet = ["a", "b", "c"]
    fsa = FiniteStateAutomaton(4, alphabet)

    # State 0 transitions (initial state, expecting 'a')
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(0, "a", 1)  # First character is 'a', good
        else:
            fsa.set_transition(0, symbol, 3)  # First character not 'a', invalid

    # State 1 transitions (after 'a', expecting 'b')
    for symbol in alphabet:
        if symbol == "b":
            fsa.set_transition(1, "b", 2)  # Saw 'b' after 'a', good
        else:
            fsa.set_transition(1, symbol, 3)  # Didn't see 'b' after 'a', invalid

    # State 2 transitions (after 'b', expecting 'a')
    for symbol in alphabet:
        if symbol == "a":
            fsa.set_transition(2, "a", 1)  # Saw 'a' after 'b', good
        else:
            fsa.set_transition(2, symbol, 3)  # Didn't see 'a' after 'b', invalid

    # State 3 transitions (invalid pattern detected)
    for symbol in alphabet:
        fsa.set_transition(3, symbol, 3)  # Stay in invalid state

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)  # Empty string is valid
    fsa.set_accepting_state(1)  # Ending with 'a' is valid
    fsa.set_accepting_state(2)  # Ending with 'b' is valid

    # Set state labels
    fsa.set_state_label(0, "Start")
    fsa.set_state_label(1, "After 'a'")
    fsa.set_state_label(2, "After 'b'")
    fsa.set_state_label(3, "Invalid")

    return fsa
