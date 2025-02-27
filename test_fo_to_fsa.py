from parser import parse_fo_formula

from example_fsas import create_end_with_01_automaton, create_substring_101_automaton
from fo_to_fsa import convert_fo_to_fsa


def test_substring_101():
    # Parse the formula for strings containing "101"
    formula = "exists x (exists y (exists z (P_1(x) and P_0(y) and P_1(z) and y = x+1 and z = y+1)))"
    parsed_formula = parse_fo_formula(formula)

    # Convert the formula to FSA
    fsa = convert_fo_to_fsa(parsed_formula)

    # Get the reference FSA
    reference_fsa = create_substring_101_automaton()

    # Test with some inputs
    test_strings = [
        "",  # Should reject
        "0",  # Should reject
        "1",  # Should reject
        "01",  # Should reject
        "10",  # Should reject
        "101",  # Should accept
        "0101",  # Should accept
        "1010",  # Should accept
        "11011",  # Should accept
    ]

    print("Testing 101 substring formula:")
    print(f"Original formula: {formula}")
    print("\nTest results:")

    for s in test_strings:
        input_sequence = [int(c) for c in s]
        expected = reference_fsa.accepts(input_sequence)
        actual = fsa.accepts(input_sequence)

        match = "✓" if expected == actual else "✗"
        result = "Accept" if actual else "Reject"
        print(f"{match} {s}: {result} (Expected: {expected})")


def test_ends_with_01():
    # Parse the formula for strings ending with "01"
    formula = (
        "exists x (exists y (P_0(x) and P_1(y) and y = x+1 and !(exists z (z > y))))"
    )
    parsed_formula = parse_fo_formula(formula)

    # Convert the formula to FSA
    fsa = convert_fo_to_fsa(parsed_formula)

    # Get the reference FSA
    reference_fsa = create_end_with_01_automaton()

    # Test with some inputs
    test_strings = [
        "",  # Should reject
        "0",  # Should reject
        "1",  # Should reject
        "01",  # Should accept
        "101",  # Should accept
        "0101",  # Should accept
        "1100",  # Should reject
        "001010",  # Should reject
    ]

    print("\nTesting ends-with-01 formula:")
    print(f"Original formula: {formula}")
    print("\nTest results:")

    for s in test_strings:
        input_sequence = [int(c) for c in s]
        expected = reference_fsa.accepts(input_sequence)
        actual = fsa.accepts(input_sequence)

        match = "✓" if expected == actual else "✗"
        result = "Accept" if actual else "Reject"
        print(f"{match} {s}: {result} (Expected: {expected})")


if __name__ == "__main__":
    test_substring_101()
    test_ends_with_01()
