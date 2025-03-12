import re

from automathic.ltl.formula import (
    BooleanConstant,
    Conjunction,
    Disjunction,
    Eventually,  # F operator (future)
    LTLFormula,
    Negation,
    Past,  # Add Past operator (P)
    Proposition,
    Since,  # Add Since operator (S)
    SymbolPredicate,
    Until,
)


# === LTL FORMULA PARSER ===
def tokenize(formula: str):
    # Define patterns for different token types
    symbol_predicate_pattern = r"Q[A-Za-z0-9]"  # Match Qa, Qb, etc. as single tokens
    identifier_pattern = r"[A-Za-z_][A-Za-z0-9_]*"
    number_pattern = r"\d+"
    operator_pattern = r"<->|->|&&|\|\|"  # Logical operators
    temporal_pattern = r"[FPSU]"  # Temporal operators (F, P, S, U)
    special_chars_pattern = r"[()\&|!]"  # Removed Q since we handle Qa separately
    keyword_pattern = r"true|false|TRUE|FALSE|True|False|and|or|not|implies|iff|since|until|past|future"  # Added natural language connectors

    # Combine patterns - note that symbol_predicate_pattern comes first to prioritize it
    combined_pattern = f"({symbol_predicate_pattern})|({identifier_pattern})|({number_pattern})|({operator_pattern})|({temporal_pattern})|({special_chars_pattern})|({keyword_pattern})"

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


class LTLParser:
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
        return self.parse_boolean_expression()

    def parse_boolean_expression(self):
        return self.parse_disjunction_expression()

    def parse_disjunction_expression(self):
        left = self.parse_conjunction_expression()
        while self.peek() in ["||", "|", "or"]:
            self.consume()  # Consume "||", "|", or "or"
            right = self.parse_conjunction_expression()
            left = Disjunction(left, right)
        return left

    def parse_conjunction_expression(self):
        left = self.parse_temporal_expression()
        while self.peek() in ["&&", "&", "and"]:
            self.consume()  # Consume "&&", "&", or "and"
            right = self.parse_temporal_expression()
            left = Conjunction(left, right)
        return left

    def parse_temporal_expression(self):
        # Handle binary temporal operators (S, U)
        left = self.parse_unary_expression()

        while self.peek() in {"S", "U", "since", "until"}:
            op = self.consume()
            right = self.parse_unary_expression()
            if op in ["U", "until"]:
                left = Until(left, right)
            elif op in ["S", "since"]:
                left = Since(left, right)

        return left

    def parse_unary_expression(self):
        token = self.peek()

        # Unary operators: negation and temporal (P, F)
        if token in ["!", "not"]:
            self.consume()  # Consume "!" or "not"
            return Negation(self.parse_unary_expression())
        elif token in ["F", "future"]:
            self.consume()  # Consume "F" or "future"
            return Eventually(self.parse_unary_expression())
        elif token in ["P", "past"]:
            self.consume()  # Consume "P" or "past"
            return Past(self.parse_unary_expression())
        else:
            return self.parse_primary()

    def parse_primary(self):
        token = self.peek()

        # Handle symbol predicates like Qa directly
        if token and len(token) == 2 and token[0] == "Q":
            self.consume()  # Consume the token
            return SymbolPredicate(token[1])  # Extract the symbol
        elif token == "(":
            self.consume()  # Consume "("
            subformula = self.parse_formula()
            if self.peek() == ")":
                self.consume()  # Consume ")"
            return subformula
        else:
            return self.parse_atomic_formula()

    def parse_atomic_formula(self):
        token = self.consume()

        # Boolean constants (with natural language support)
        if token.lower() == "true":
            return BooleanConstant(True)
        elif token.lower() == "false":
            return BooleanConstant(False)
        else:
            # Atomic proposition
            return Proposition(token)


def parse_ltl(formula_str: str):
    """
    Parse an LTL formula from a string.

    Args:
        formula_str: A string containing an LTL formula

    Returns:
        LTLFormula: The parsed formula
    """
    tokens = tokenize(formula_str)
    parser = LTLParser(tokens)
    return parser.parse_formula()
