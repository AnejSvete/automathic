import unittest

from automathic.som.formulas_collection import φ_0, φ_1, φ_2, φ_3, φ_4, φ_5, φ_6, φ_7
from automathic.som.parser import parse_fo_formula
from automathic.som.translator import convert_fo_to_fsa


class TestSOMtoFSA(unittest.TestCase):
    def test_strings_that_start_with_a_and_end_with_b(self):
        # φ_0: Strings that start with a and end with b
        automaton = convert_fo_to_fsa(parse_fo_formula(φ_0).to_basic_form().simplify())

        # Strings that should be accepted
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("aab"))
        self.assertTrue(automaton.accepts("abb"))
        self.assertTrue(automaton.accepts("abab"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts(""))
        self.assertFalse(automaton.accepts("a"))
        self.assertFalse(automaton.accepts("b"))
        self.assertFalse(automaton.accepts("ba"))
        self.assertFalse(automaton.accepts("bb"))
        self.assertFalse(automaton.accepts("bab"))

    def test_strings_that_start_with_a(self):
        # φ_1: Strings that start with a
        automaton = convert_fo_to_fsa(
            parse_fo_formula(φ_1).to_basic_form().simplify(), alphabet=["a", "b"]
        )

        # Strings that should be accepted
        self.assertTrue(automaton.accepts("a"))
        self.assertTrue(automaton.accepts("aa"))
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("aba"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts(""))
        self.assertFalse(automaton.accepts("b"))
        self.assertFalse(automaton.accepts("ba"))
        self.assertFalse(automaton.accepts("bab"))

    def test_strings_that_end_with_b(self):
        # φ_2: Strings that end with b
        automaton = convert_fo_to_fsa(
            parse_fo_formula(φ_2).to_basic_form().simplify(), alphabet=["a", "b"]
        )

        # Strings that should be accepted
        self.assertTrue(automaton.accepts("b"))
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("bb"))
        self.assertTrue(automaton.accepts("abb"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts(""))
        self.assertFalse(automaton.accepts("a"))
        self.assertFalse(automaton.accepts("ba"))
        self.assertFalse(automaton.accepts("baa"))

    def test_every_a_immediately_followed_by_b(self):
        # φ_3: Every a is immediately followed by b
        automaton = convert_fo_to_fsa(parse_fo_formula(φ_3).to_basic_form().simplify())

        # Strings that should be accepted
        self.assertTrue(automaton.accepts(""))
        self.assertTrue(automaton.accepts("b"))
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("abab"))
        self.assertTrue(automaton.accepts("bb"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts("a"))
        self.assertFalse(automaton.accepts("aa"))
        self.assertFalse(automaton.accepts("aba"))
        self.assertFalse(automaton.accepts("abaa"))

    def test_no_a_immediately_followed_by_a(self):
        # φ_4: No a is immediately followed by another a
        automaton = convert_fo_to_fsa(
            parse_fo_formula(φ_4).to_basic_form().simplify(), alphabet=["a", "b"]
        )

        # Strings that should be accepted
        self.assertTrue(automaton.accepts(""))
        self.assertTrue(automaton.accepts("a"))
        self.assertTrue(automaton.accepts("b"))
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("aba"))
        self.assertTrue(automaton.accepts("bab"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts("aa"))
        self.assertFalse(automaton.accepts("baa"))
        self.assertFalse(automaton.accepts("aab"))
        self.assertFalse(automaton.accepts("baab"))
        self.assertFalse(automaton.accepts("ababaa"))

    def test_exactly_one_a(self):
        # φ_5: Strings with exactly one a
        automaton = convert_fo_to_fsa(
            parse_fo_formula(φ_5).to_basic_form().simplify(), alphabet=["a", "b"]
        )

        # Strings that should be accepted
        self.assertTrue(automaton.accepts("a"))
        self.assertTrue(automaton.accepts("ba"))
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("bab"))
        self.assertTrue(automaton.accepts("bbabb"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts(""))
        self.assertFalse(automaton.accepts("b"))
        self.assertFalse(automaton.accepts("aa"))
        self.assertFalse(automaton.accepts("aba"))
        self.assertFalse(automaton.accepts("aab"))

    def test_every_a_preceded_by_b(self):
        # φ_6: Every a is preceded by a b
        automaton = convert_fo_to_fsa(parse_fo_formula(φ_6).to_basic_form().simplify())

        # Strings that should be accepted
        self.assertTrue(automaton.accepts(""))
        self.assertTrue(automaton.accepts("b"))
        self.assertTrue(automaton.accepts("ba"))
        self.assertTrue(automaton.accepts("bba"))
        self.assertTrue(automaton.accepts("bbba"))
        self.assertTrue(automaton.accepts("bab"))
        self.assertTrue(automaton.accepts("bbabbba"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts("a"))
        self.assertFalse(automaton.accepts("ab"))
        self.assertFalse(automaton.accepts("aba"))
        self.assertFalse(automaton.accepts("abba"))

    def test_alternating_a_b_starting_with_a(self):
        # φ_7: Strings where a and b alternate, starting with a
        automaton = convert_fo_to_fsa(parse_fo_formula(φ_7).to_basic_form().simplify())

        # Strings that should be accepted
        self.assertTrue(automaton.accepts("a"))
        self.assertTrue(automaton.accepts("ab"))
        self.assertTrue(automaton.accepts("aba"))
        self.assertTrue(automaton.accepts("abab"))
        self.assertTrue(automaton.accepts("ababa"))

        # Strings that should be rejected
        self.assertFalse(automaton.accepts(""))
        self.assertFalse(automaton.accepts("b"))
        self.assertFalse(automaton.accepts("aa"))
        self.assertFalse(automaton.accepts("abb"))
        self.assertFalse(automaton.accepts("abaa"))
        self.assertFalse(automaton.accepts("ababb"))
        self.assertFalse(automaton.accepts("ba"))


if __name__ == "__main__":
    unittest.main()
