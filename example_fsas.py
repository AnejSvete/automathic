from fsa import FiniteStateAutomaton


def create_substring_101_automaton():
    """
    Creates an FSA that recognizes strings containing '101' as a substring.

    States:
    - 0: Initial state (haven't seen any pattern yet)
    - 1: Seen '1'
    - 2: Seen '10'
    - 3: Seen '101' (accepting state)
    """
    # Create FSA with 4 states and alphabet {0,1}
    fsa = FiniteStateAutomaton(4, 2)

    # Set up transitions
    # State 0 transitions
    fsa.set_transition(0, 0, 0)  # On input 0, stay in state 0
    fsa.set_transition(0, 1, 1)  # On input 1, move to state 1

    # State 1 transitions
    fsa.set_transition(1, 0, 2)  # On input 0, move to state 2
    fsa.set_transition(1, 1, 1)  # On input 1, stay in state 1

    # State 2 transitions
    fsa.set_transition(2, 0, 0)  # On input 0, go back to state 0
    fsa.set_transition(2, 1, 3)  # On input 1, move to accepting state 3

    # State 3 transitions (once in accepting state, stay there)
    fsa.set_transition(3, 0, 3)  # On input 0, stay in state 3
    fsa.set_transition(3, 1, 3)  # On input 1, stay in state 3

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(3)

    return fsa


def create_end_with_01_automaton():
    """
    Creates an FSA that recognizes strings ending with '01'.

    States:
    - 0: Initial state
    - 1: Seen '0' as the potential start of '01'
    - 2: Seen '01' (accepting state)

    Returns:
        FiniteStateAutomaton: The automaton accepting strings ending with '01'
    """
    fsa = FiniteStateAutomaton(3, 2)

    # Set transitions
    fsa.set_transition(0, 0, 1)  # On input 0, move to state 1
    fsa.set_transition(0, 1, 0)  # On input 1, stay in state 0

    fsa.set_transition(1, 0, 1)  # On input 0, stay in state 1
    fsa.set_transition(1, 1, 2)  # On input 1, move to accepting state 2

    fsa.set_transition(2, 0, 1)  # On input 0, go to state 1
    fsa.set_transition(2, 1, 0)  # On input 1, go to state 0

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
    fsa = FiniteStateAutomaton(3, 2)

    # Set transitions
    # State 0 (remainder 0)
    fsa.set_transition(0, 0, 0)  # 0 * 2 = 0 (mod 3)
    fsa.set_transition(0, 1, 1)  # 0 * 2 + 1 = 1 (mod 3)

    # State 1 (remainder 1)
    fsa.set_transition(1, 0, 2)  # 1 * 2 = 2 (mod 3)
    fsa.set_transition(1, 1, 0)  # 1 * 2 + 1 = 3 = 0 (mod 3)

    # State 2 (remainder 2)
    fsa.set_transition(2, 0, 1)  # 2 * 2 = 4 = 1 (mod 3)
    fsa.set_transition(2, 1, 2)  # 2 * 2 + 1 = 5 = 2 (mod 3)

    # Set initial and accepting states
    fsa.set_initial_state(0)
    fsa.set_accepting_state(0)  # Only accept if remainder is 0

    return fsa


# Example usage in a notebook:
if __name__ == "__main__":
    print("Creating example FSAs...")
    substring_fsa = create_substring_101_automaton()
    print(substring_fsa)

    end_with_fsa = create_end_with_01_automaton()
    print(end_with_fsa)

    divisible_fsa = create_divisible_by_3_automaton()
    print(divisible_fsa)
