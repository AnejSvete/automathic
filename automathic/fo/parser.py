import re

from automathic.fo.formula import (
    Conjunction,
    Disjunction,
    Equivalence,  # two new formula classes
    ExistentialQuantifier,
    ExistentialSetQuantifier,
    Implication,  # Need to import these
    Negation,
    Predicate,
    Relation,
    SetMembership,
    SymbolPredicate,
    UniversalQuantifier,
    UniversalSetQuantifier,
)


# === SOM FORMULA PARSER ===
def tokenize(formula: str):
    # Define patterns for different token types
    quantifier_pattern = r"exists|forall"
    identifier_pattern = r"[A-Za-z_][A-Za-z0-9_]*"
    number_pattern = r"\d+"
    operator_pattern = r"<=>|=>|<=|>=|=|<|>|\+|in"  # Added "<=>" and "=>" for logical equivalence and implication
    special_chars_pattern = r"[()\&|!.]"
    keyword_pattern = (
        r"and|or|not|iff|implies"  # Added "iff" and "implies" as alternatives
    )

    # Combine patterns
    combined_pattern = f"({quantifier_pattern})|({identifier_pattern})|({number_pattern})|({operator_pattern})|({special_chars_pattern})|({keyword_pattern})"

    # Find all tokens
    tokens = re.findall(combined_pattern, formula)

    # Flatten the groups and remove empty matches
    result = []
    for match_groups in tokens:
        for group in match_groups:
            if group:  # Only add non-empty groups
                result.append(group)
                break  # Only take the first non-empty group from each match

    return result


class SOMParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def consume(self):
        token = self.peek()
        self.index += 1
        return token

    def parse_formula(self):
        return self.parse_implies_expression()

    def parse_implies_expression(self):
        left = self.parse_iff_expression()
        while self.peek() in {"=>", "implies"}:
            self.consume()  # Consume "=>" or "implies"
            right = self.parse_iff_expression()
            left = Implication(left, right)
        return left

    def parse_iff_expression(self):
        left = self.parse_bool_expression()
        while self.peek() in {"<=>", "iff"}:
            self.consume()  # Consume "<=>" or "iff"
            right = self.parse_bool_expression()
            left = Equivalence(left, right)
        return left

    def parse_bool_expression(self):
        left = self.parse_term()
        while self.peek() == "or":
            self.consume()  # Consume "or"
            right = self.parse_term()
            left = Disjunction(left, right)
        return left

    def parse_term(self):
        left = self.parse_primary()
        while self.peek() == "and":
            self.consume()  # Consume "and"
            right = self.parse_primary()
            left = Conjunction(left, right)
        return left

    def parse_primary(self):
        token = self.peek()

        if token == "!" or token == "not":  # Support both "!" and "not"
            self.consume()
            return Negation(self.parse_primary())
        elif token in {"exists", "forall"}:
            return self.parse_quantifier()
        elif token == "(":
            self.consume()  # Consume "("
            subformula = self.parse_formula()
            if self.peek() == ")":
                self.consume()  # Consume ")"
            return subformula
        else:
            return self.parse_atomic_formula()

    def parse_quantifier(self):
        quantifier = self.consume()
        variable = self.consume()

        # Check if this is a set variable (uppercase first letter)
        is_set_variable = variable[0].isupper()

        # If there's a "." after the variable, consume it
        if self.peek() == ".":
            self.consume()

        # Create the appropriate quantifier based on type and variable kind
        if quantifier == "exists":
            if is_set_variable:
                return ExistentialSetQuantifier(variable, self.parse_primary())
            else:
                return ExistentialQuantifier(variable, self.parse_primary())
        else:  # forall
            if is_set_variable:
                return UniversalSetQuantifier(variable, self.parse_primary())
            else:
                return UniversalQuantifier(variable, self.parse_primary())

    def parse_atomic_formula(self):
        token = self.consume()

        # Check for set membership: "x in X"
        if self.peek() == "in":
            self.consume()  # Consume "in"
            set_variable = self.consume()
            return SetMembership(token, set_variable)

        if self.peek() == "(":  # This is a predicate
            self.consume()  # Consume "("
            var = self.consume()
            if self.peek() == ")":
                self.consume()  # Consume ")"
            if token[:1] == "Q":
                return SymbolPredicate(token[:1], var, token[1:])
            else:
                return Predicate(token, var)
        else:
            # This might be a variable in a relation
            if self.peek() in {"<", "<=", ">", ">=", "="}:
                return self.parse_relation(token)
            return token  # Just return the variable name

    def parse_relation(self, left_operand):
        operator = self.consume()
        right_operand = self.consume()

        # Handle expressions like "x + 1"
        if self.peek() == "+":
            self.consume()  # Consume "+"
            additional_term = self.consume()
            right_operand = f"{right_operand}+{additional_term}"

        return Relation(left_operand, operator, right_operand)


def parse_som_formula(formula_str: str):
    tokens = tokenize(formula_str)
    parser = SOMParser(tokens)
    return parser.parse_formula()


# Keep the original parse_fo_formula for backward compatibility
def parse_fo_formula(formula_str: str):
    tokens = tokenize(formula_str)
    parser = SOMParser(tokens)  # Use SOMParser since it's a superset of FOParser
    return parser.parse_formula()
