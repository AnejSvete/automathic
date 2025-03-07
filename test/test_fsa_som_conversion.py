import unittest

from automathic.fo.formulas_collection import φ_0, φ_1, φ_2, φ_3, φ_4, φ_5, φ_6, φ_7
from automathic.fo.parser import parse_fo_formula
from automathic.fsa.example_fsas import (
    create_a_followed_by_b_automaton,
    create_a_preceded_by_b_automaton,
    create_alternating_ab_automaton,
    create_contains_a_followed_by_b_automaton,
    create_divisible_by_3_automaton,
    create_end_with_ab_automaton,
    create_ends_with_b_automaton,
    create_even_length_automaton,
    create_even_number_of_as_automaton,
    create_even_parity_automaton,
    create_exactly_one_a_automaton,
    create_no_aa_automaton,
    create_odd_parity_automaton,
    create_sigma_star_automaton,
    create_starts_with_a_and_ends_with_b_automaton,
    create_starts_with_a_automaton,
    create_substring_abc_automaton,
)


class TestFSAtoSOM(unittest.TestCase):

    def test_a_followed_by_b_automaton(self):
        A = create_a_followed_by_b_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    def test_a_preceded_by_b_automaton(self):
        A = create_a_preceded_by_b_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    # def test_alternating_ab_automaton(self):
    #     A = create_alternating_ab_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    # def test_contains_a_followed_by_b_automaton(self):
    #     A = create_contains_a_followed_by_b_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    # def test_divisible_by_3_automaton(self):
    #     A = create_divisible_by_3_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    # def test_end_with_ab_automaton(self):
    #     A = create_end_with_ab_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    def test_ends_with_b_automaton(self):
        A = create_ends_with_b_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    def test_even_length_automaton(self):
        A = create_even_length_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    def test_even_number_of_as_automaton(self):
        A = create_even_number_of_as_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    def test_even_parity_automaton(self):
        A = create_even_parity_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    # def test_exactly_one_a_automaton(self):
    #     A = create_exactly_one_a_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    def test_no_aa_automaton(self):
        A = create_no_aa_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    def test_odd_parity_automaton(self):
        A = create_odd_parity_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    def test_sigma_star_automaton(self):
        A = create_sigma_star_automaton()
        A_ = A.to_som().to_basic_form().simplify().to_fsa()

        self.assertTrue(A.is_equivalent(A_))

    # def test_starts_with_a_and_ends_with_b_automaton(self):
    #     A = create_starts_with_a_and_ends_with_b_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    # def test_starts_with_a_automaton(self):
    #     A = create_starts_with_a_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))

    # def test_substring_abc_automaton(self):
    #     A = create_substring_abc_automaton()
    #     A_ = A.to_som().to_basic_form().simplify().to_fsa()

    #     self.assertTrue(A.is_equivalent(A_))


if __name__ == "__main__":
    unittest.main()
