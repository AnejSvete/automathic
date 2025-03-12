"""
Linear Temporal Logic Formula Representations

This module provides classes for representing Linear Temporal Logic (LTL) formulas,
supporting:

- Basic formula types (propositions, negation, conjunction, etc.)
- Temporal operators (Next, Eventually, Globally, Until, Release)

LTL formulas can express properties of infinite execution paths and are widely used
in formal verification of reactive systems.
"""

from dataclasses import dataclass
from typing import Set


class LTLFormula:
    """
    Base class for all Linear Temporal Logic formulas.

    Linear Temporal Logic extends propositional logic with operators that
    describe properties of execution paths over time, such as:
    - "X p" (next): p holds in the next state
    - "F p" (eventually): p holds sometime in the future
    - "G p" (globally): p holds in every future state
    - "p U q" (until): p holds until q holds
    - "p R q" (release): q holds until and including when p holds, or q holds forever

    LTL is widely used in formal verification to specify requirements for
    reactive systems.
    """

    def get_propositions(self) -> Set[str]:
        """
        Extract all atomic propositions used in the formula.

        Returns:
            set: A set of proposition names used in the formula
        """
        if isinstance(self, Proposition):
            return {self.name}
        elif isinstance(self, BooleanConstant):
            return set()
        elif isinstance(self, Negation):
            return self.subformula.get_propositions()
        elif isinstance(self, BinaryOp):  # Covers Conjunction, Disjunction, etc.
            return self.left.get_propositions().union(self.right.get_propositions())
        elif isinstance(self, TemporalOp):  # Covers Next, Eventually, Globally
            return self.subformula.get_propositions()
        elif isinstance(self, BinaryTemporalOp):  # Covers Until, Release
            return self.left.get_propositions().union(self.right.get_propositions())
        else:
            return set()

    def simplify(self):
        """
        Simplifies a formula in LTL by applying basic logical transformations.

        This includes:
        - Eliminating double negations
        - Applying temporal operator equivalences
        - Simplifying Boolean tautologies and contradictions

        Returns:
            LTLFormula: A simplified equivalent formula
        """
        if isinstance(self, Proposition) or isinstance(self, BooleanConstant):
            return self
        elif isinstance(self, Negation):
            # Simplify the subformula first
            simplified_subformula = self.subformula.simplify()

            # Eliminate double negation: ¬¬φ ≡ φ
            if isinstance(simplified_subformula, Negation):
                return simplified_subformula.subformula

            # Apply De Morgan's laws for temporal operators
            if isinstance(simplified_subformula, Globally):
                # ¬(G φ) ≡ F(¬φ)
                return Eventually(Negation(simplified_subformula.subformula).simplify())
            elif isinstance(simplified_subformula, Eventually):
                # ¬(F φ) ≡ G(¬φ)
                return Globally(Negation(simplified_subformula.subformula).simplify())

            # No simplification applicable
            return Negation(simplified_subformula)

        elif isinstance(self, Conjunction):
            # Simplify both sides
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()

            # Boolean simplifications
            if isinstance(left_simplified, BooleanConstant):
                if left_simplified.value:  # True ∧ φ ≡ φ
                    return right_simplified
                else:  # False ∧ φ ≡ False
                    return BooleanConstant(False)
            if isinstance(right_simplified, BooleanConstant):
                if right_simplified.value:  # φ ∧ True ≡ φ
                    return left_simplified
                else:  # φ ∧ False ≡ False
                    return BooleanConstant(False)

            # Eliminate redundant conjunctions: A ∧ A ≡ A
            if left_simplified.to_string() == right_simplified.to_string():
                return left_simplified

            # Default case - keep the simplified conjunction
            return Conjunction(left_simplified, right_simplified)

        elif isinstance(self, Disjunction):
            # Simplify both sides
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()

            # Boolean simplifications
            if isinstance(left_simplified, BooleanConstant):
                if left_simplified.value:  # True ∨ φ ≡ True
                    return BooleanConstant(True)
                else:  # False ∨ φ ≡ φ
                    return right_simplified
            if isinstance(right_simplified, BooleanConstant):
                if right_simplified.value:  # φ ∨ True ≡ True
                    return BooleanConstant(True)
                else:  # φ ∨ False ≡ φ
                    return left_simplified

            # Eliminate redundant disjunctions: A ∨ A ≡ A
            if left_simplified.to_string() == right_simplified.to_string():
                return left_simplified

            # Default case - keep the simplified disjunction
            return Disjunction(left_simplified, right_simplified)

        elif isinstance(self, Implication):
            # A → B ≡ ¬A ∨ B
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()
            return Disjunction(Negation(left_simplified), right_simplified).simplify()

        elif isinstance(self, Equivalence):
            # A ⟺ B ≡ (A → B) ∧ (B → A)
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()
            return Conjunction(
                Implication(left_simplified, right_simplified),
                Implication(right_simplified, left_simplified),
            ).simplify()

        elif isinstance(self, Next):
            # Simplify the subformula
            simplified_subformula = self.subformula.simplify()
            return Next(simplified_subformula)

        elif isinstance(self, Eventually):
            # Simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # F F φ ≡ F φ
            if isinstance(simplified_subformula, Eventually):
                return simplified_subformula

            # F true ≡ true
            if (
                isinstance(simplified_subformula, BooleanConstant)
                and simplified_subformula.value
            ):
                return BooleanConstant(True)

            # Default case
            return Eventually(simplified_subformula)

        elif isinstance(self, Globally):
            # Simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # G G φ ≡ G φ
            if isinstance(simplified_subformula, Globally):
                return simplified_subformula

            # G false ≡ false
            if (
                isinstance(simplified_subformula, BooleanConstant)
                and not simplified_subformula.value
            ):
                return BooleanConstant(False)

            # Default case
            return Globally(simplified_subformula)

        elif isinstance(self, Until):
            # Simplify both sides
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()

            # φ U true ≡ true
            if isinstance(right_simplified, BooleanConstant) and right_simplified.value:
                return BooleanConstant(True)

            # false U φ ≡ φ
            if (
                isinstance(left_simplified, BooleanConstant)
                and not left_simplified.value
            ):
                return right_simplified

            # φ U false ≡ false (assuming we're on infinite traces)
            if (
                isinstance(right_simplified, BooleanConstant)
                and not right_simplified.value
            ):
                return BooleanConstant(False)

            # Default case
            return Until(left_simplified, right_simplified)

        elif isinstance(self, Release):
            # Simplify both sides
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()

            # true R φ ≡ φ
            if isinstance(left_simplified, BooleanConstant) and left_simplified.value:
                return right_simplified

            # false R φ ≡ G φ
            if (
                isinstance(left_simplified, BooleanConstant)
                and not left_simplified.value
            ):
                return Globally(right_simplified).simplify()

            # φ R false ≡ false
            if (
                isinstance(right_simplified, BooleanConstant)
                and not right_simplified.value
            ):
                return BooleanConstant(False)

            # Default case
            return Release(left_simplified, right_simplified)

        else:
            # Unknown formula type, return as is
            return self

    def to_nnf(self):
        """
        Convert the formula to Negation Normal Form (NNF).
        In NNF, negations only appear directly before atomic propositions.

        Returns:
            LTLFormula: The formula in NNF
        """
        if isinstance(self, Proposition) or isinstance(self, BooleanConstant):
            return self
        elif isinstance(self, Negation):
            subformula = self.subformula
            if isinstance(subformula, Proposition) or isinstance(
                subformula, BooleanConstant
            ):
                # Negation of an atomic proposition stays as is
                return self
            elif isinstance(subformula, Negation):
                # Double negation elimination: ¬¬φ ≡ φ
                return subformula.subformula.to_nnf()
            elif isinstance(subformula, Conjunction):
                # De Morgan's law: ¬(φ ∧ ψ) ≡ ¬φ ∨ ¬ψ
                return Disjunction(
                    Negation(subformula.left).to_nnf(),
                    Negation(subformula.right).to_nnf(),
                )
            elif isinstance(subformula, Disjunction):
                # De Morgan's law: ¬(φ ∨ ψ) ≡ ¬φ ∧ ¬ψ
                return Conjunction(
                    Negation(subformula.left).to_nnf(),
                    Negation(subformula.right).to_nnf(),
                )
            elif isinstance(subformula, Implication):
                # ¬(φ → ψ) ≡ φ ∧ ¬ψ
                return Conjunction(
                    subformula.left.to_nnf(), Negation(subformula.right).to_nnf()
                )
            elif isinstance(subformula, Equivalence):
                # ¬(φ ↔ ψ) ≡ (φ ∧ ¬ψ) ∨ (¬φ ∧ ψ)
                return Disjunction(
                    Conjunction(
                        subformula.left.to_nnf(), Negation(subformula.right).to_nnf()
                    ),
                    Conjunction(
                        Negation(subformula.left).to_nnf(), subformula.right.to_nnf()
                    ),
                )
            elif isinstance(subformula, Next):
                # ¬(X φ) ≡ X(¬φ)
                return Next(Negation(subformula.subformula).to_nnf())
            elif isinstance(subformula, Eventually):
                # ¬(F φ) ≡ G(¬φ)
                return Globally(Negation(subformula.subformula).to_nnf())
            elif isinstance(subformula, Globally):
                # ¬(G φ) ≡ F(¬φ)
                return Eventually(Negation(subformula.subformula).to_nnf())
            elif isinstance(subformula, Until):
                # ¬(φ U ψ) ≡ (¬φ R ¬ψ)
                return Release(
                    Negation(subformula.left).to_nnf(),
                    Negation(subformula.right).to_nnf(),
                )
            elif isinstance(subformula, Release):
                # ¬(φ R ψ) ≡ (¬φ U ¬ψ)
                return Until(
                    Negation(subformula.left).to_nnf(),
                    Negation(subformula.right).to_nnf(),
                )
            else:
                # Unknown formula type
                return Negation(subformula.to_nnf())
        elif isinstance(self, Conjunction):
            return Conjunction(self.left.to_nnf(), self.right.to_nnf())
        elif isinstance(self, Disjunction):
            return Disjunction(self.left.to_nnf(), self.right.to_nnf())
        elif isinstance(self, Implication):
            # φ → ψ ≡ ¬φ ∨ ψ
            return Disjunction(Negation(self.left).to_nnf(), self.right.to_nnf())
        elif isinstance(self, Equivalence):
            # φ ↔ ψ ≡ (φ ∧ ψ) ∨ (¬φ ∧ ¬ψ)
            return Disjunction(
                Conjunction(self.left.to_nnf(), self.right.to_nnf()),
                Conjunction(
                    Negation(self.left).to_nnf(), Negation(self.right).to_nnf()
                ),
            )
        elif isinstance(self, Next):
            return Next(self.subformula.to_nnf())
        elif isinstance(self, Eventually):
            return Eventually(self.subformula.to_nnf())
        elif isinstance(self, Globally):
            return Globally(self.subformula.to_nnf())
        elif isinstance(self, Until):
            return Until(self.left.to_nnf(), self.right.to_nnf())
        elif isinstance(self, Release):
            return Release(self.left.to_nnf(), self.right.to_nnf())
        else:
            # Unknown formula type
            return self

    def to_gnf(self):
        """
        Convert formula to Generalized Normal Form (GNF).
        This is useful for automata construction, as it normalizes
        complex formulas into a form amenable to tableaux algorithms.

        Returns:
            LTLFormula: Formula in GNF
        """
        # First convert to NNF to push negations inward
        nnf_formula = self.to_nnf()

        # Then apply GNF transformations
        return nnf_formula._to_gnf_helper()

    def _to_gnf_helper(self):
        """
        Helper method for GNF conversion, to be overridden by subclasses
        """
        return self

    def to_fsa(self):
        """
        Convert the LTL formula to an automaton.

        Returns:
            Automaton: An automaton accepting the same language as the formula
        """
        from automathic.ltl.translator import convert_ltl_to_fsa

        return convert_ltl_to_fsa(self)

    def __str__(self):
        """
        String representation of the formula.
        This allows formulas to be printed directly with print(formula).
        """
        return self.formatted_string(formatted=True)

    def formatted_string(self, formatted=True, prefix=""):
        """
        Print a nicely formatted string representation of the formula.

        Args:
            formatted: If True, add formatting to highlight the formula structure
            prefix: Text to print before the formula
        """
        formula_str = self.to_string()

        if formatted:
            # Replace logical symbols with colored versions if in terminal
            try:
                # ANSI color codes for terminals
                BLUE = "\033[94m"
                GREEN = "\033[92m"
                RED = "\033[91m"
                YELLOW = "\033[93m"
                BOLD = "\033[1m"
                ENDC = "\033[0m"

                # Highlight logical operators
                formula_str = formula_str.replace("∧", f"{BLUE}∧{ENDC}")
                formula_str = formula_str.replace("∨", f"{YELLOW}∨{ENDC}")
                formula_str = formula_str.replace("¬", f"{RED}¬{ENDC}")
                formula_str = formula_str.replace("→", f"{BLUE}→{ENDC}")
                formula_str = formula_str.replace("⟺", f"{BLUE}⟺{ENDC}")

                # Highlight temporal operators
                formula_str = formula_str.replace("X", f"{GREEN}X{ENDC}")
                formula_str = formula_str.replace("F", f"{GREEN}F{ENDC}")
                formula_str = formula_str.replace("G", f"{GREEN}G{ENDC}")
                formula_str = formula_str.replace("U", f"{GREEN}U{ENDC}")
                formula_str = formula_str.replace("R", f"{GREEN}R{ENDC}")

                # Highlight parentheses
                formula_str = formula_str.replace("(", f"{BOLD}({ENDC}")
                formula_str = formula_str.replace(")", f"{BOLD}){ENDC}")
            except:
                # If there's any issue with color codes (e.g., non-terminal environment),
                # just use the regular string
                pass

        return f"{prefix}{formula_str}"

    def to_canonical_string(self):
        """
        Convert formula to a canonical string representation that
        can be parsed back into an equivalent formula.
        """
        if isinstance(self, Proposition):
            return self.name
        elif isinstance(self, BooleanConstant):
            return str(self.value).lower()
        elif isinstance(self, Negation):
            return f"!({self.subformula.to_canonical_string()})"
        elif isinstance(self, Conjunction):
            return f"({self.left.to_canonical_string()} && {self.right.to_canonical_string()})"
        elif isinstance(self, Disjunction):
            return f"({self.left.to_canonical_string()} || {self.right.to_canonical_string()})"
        elif isinstance(self, Implication):
            return f"({self.left.to_canonical_string()} -> {self.right.to_canonical_string()})"
        elif isinstance(self, Equivalence):
            return f"({self.left.to_canonical_string()} <-> {self.right.to_canonical_string()})"
        elif isinstance(self, Next):
            return f"X({self.subformula.to_canonical_string()})"
        elif isinstance(self, Eventually):
            return f"F({self.subformula.to_canonical_string()})"
        elif isinstance(self, Globally):
            return f"G({self.subformula.to_canonical_string()})"
        elif isinstance(self, Until):
            return f"({self.left.to_canonical_string()} U {self.right.to_canonical_string()})"
        elif isinstance(self, Release):
            return f"({self.left.to_canonical_string()} R {self.right.to_canonical_string()})"
        else:
            return str(self)

    def to_string(self):
        """Convert formula to string representation"""
        return str(self)

    def to_ascii(self):
        """Generate an ASCII representation of the formula"""
        result = []
        self._to_ascii_helper(result, 0)
        return "\n".join(result)

    def _to_ascii_helper(self, lines, indent):
        """Helper method for ASCII visualization - to be implemented by subclasses"""
        lines.append(" " * indent + str(self))

    def _repr_html_(self):
        """Special method for Jupyter notebook HTML display"""
        css = """
            <style>
            .formula-tree {
                font-family: 'Consolas', 'Courier New', monospace;
                line-height: 1.4;
            }
            .formula-node {
                position: relative;
                margin: 2px 0;
            }
            .node-content {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 3px;
            }
            .node-operator {
                font-weight: bold;
            }
            .node-proposition {
                background-color: #e6ffe6;
                color: #006600;
            }
            .node-boolean {
                background-color: #e6f3ff;
                color: #0066cc;
                font-weight: bold;
            }
            .node-conjunction {
                background-color: #fff0e6;
                color: #cc6600;
                font-weight: bold;
            }
            .node-disjunction {
                background-color: #fff0e6;
                color: #cc6600;
                font-weight: bold;
            }
            .node-implication {
                background-color: #f0e6ff;
                color: #6600cc;
                font-weight: bold;
            }
            .node-equivalence {
                background-color: #e6f0ff;
                color: #0066cc;
                font-weight: bold;
            }
            .node-negation {
                background-color: #ffe6e6;
                color: #cc0000;
                font-weight: bold;
            }
            .node-temporal {
                background-color: #e6ffff;
                color: #006666;
                font-weight: bold;
            }
            .node-children {
                margin-left: 20px;
                padding-left: 10px;
                border-left: 1px solid #ccc;
            }
            .tree-line {
                color: #999;
                margin-right: 5px;
                font-size: 0.9em;
            }
            </style>
            """
        return css + f'<div class="formula-tree">{self.to_html()}</div>'

    def to_html(self):
        """Convert AST to HTML for Jupyter Notebooks"""
        # Each subclass will implement this
        return ""


# Abstract classes for organizing the hierarchy
class BinaryOp(LTLFormula):
    """Abstract base class for binary logical operators"""

    left: LTLFormula
    right: LTLFormula


class TemporalOp(LTLFormula):
    """Abstract base class for unary temporal operators"""

    subformula: LTLFormula


class BinaryTemporalOp(LTLFormula):
    """Abstract base class for binary temporal operators"""

    left: LTLFormula
    right: LTLFormula


@dataclass
class SymbolPredicate(LTLFormula):
    """
    Represents a predicate checking if the current position has a specific symbol.
    Qa means "the current position has symbol 'a'"
    """

    symbol: str

    def to_string(self):
        return f"Q{self.symbol}"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + f"Q{self.symbol}")

    def __hash__(self):
        return hash(("SymbolPredicate", self.symbol))

    def __eq__(self, other):
        if not isinstance(other, SymbolPredicate):
            return False
        return self.symbol == other.symbol

    def to_html(self):
        return f'<div class="formula-node"><span class="node-content node-proposition">Q{self.symbol}</span></div>'

    def get_propositions(self) -> Set[str]:
        """
        Symbol predicates are treated as special propositions when collecting formula parts
        """
        return {f"Q{self.symbol}"}

    # Override simplify and to_nnf to handle SymbolPredicate appropriately
    def simplify(self):
        return self

    def to_nnf(self):
        return self


@dataclass
class Proposition(LTLFormula):
    """
    Represents an atomic proposition.
    """

    name: str

    def to_string(self):
        return self.name

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Proposition):
            return False
        return self.name == other.name

    def to_html(self):
        return f'<div class="formula-node"><span class="node-content node-proposition">{self.name}</span></div>'


@dataclass
class BooleanConstant(LTLFormula):
    """
    Represents a boolean constant (True or False).
    """

    value: bool

    def to_string(self):
        return "⊤" if self.value else "⊥"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + self.to_string())

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, BooleanConstant):
            return False
        return self.value == other.value

    def to_html(self):
        text = "⊤ (TRUE)" if self.value else "⊥ (FALSE)"
        return f'<div class="formula-node"><span class="node-content node-boolean">{text}</span></div>'


@dataclass
class Negation(LTLFormula):
    """
    Represents the logical negation (¬).
    """

    subformula: LTLFormula

    def to_string(self):
        return f"¬({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "¬(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-negation">¬ (NOT)</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class Conjunction(BinaryOp):
    """
    Represents logical conjunction (∧).
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} ∧ {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  ∧")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-conjunction">∧ (AND)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """


@dataclass
class Disjunction(BinaryOp):
    """
    Represents logical disjunction (∨).
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} ∨ {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  ∨")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-disjunction">∨ (OR)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """


@dataclass
class Implication(BinaryOp):
    """
    Represents logical implication (→).
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} → {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  →")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-implication">→ (IMPLIES)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """


@dataclass
class Equivalence(BinaryOp):
    """
    Represents logical equivalence (⟺).
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} ⟺ {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  ⟺")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-equivalence">⟺ (IFF)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """


@dataclass
class Next(TemporalOp):
    """
    Represents the "next" temporal operator (X).
    X φ means that φ holds in the next state.
    """

    subformula: LTLFormula

    def to_string(self):
        return f"X({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "X(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">X (NEXT)</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class Eventually(TemporalOp):
    """
    Represents the "eventually" temporal operator (F).
    F φ means that φ will hold at some point in the future.
    """

    subformula: LTLFormula

    def to_string(self):
        return f"F({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "F(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">F (EVENTUALLY)</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class Globally(TemporalOp):
    """
    Represents the "globally" temporal operator (G).
    G φ means that φ holds at every point in the future.
    """

    subformula: LTLFormula

    def to_string(self):
        return f"G({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "G(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">G (GLOBALLY)</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class Until(BinaryTemporalOp):
    """
    Represents the "until" temporal operator (U).
    φ U ψ means that φ holds continuously until ψ becomes true,
    and ψ must eventually become true.
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} U {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  U")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">U (UNTIL)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """


@dataclass
class Release(BinaryTemporalOp):
    """
    Represents the "release" temporal operator (R).
    φ R ψ means that ψ holds continuously until and including when φ becomes true,
    or ψ holds forever if φ never becomes true.
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} R {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  R")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">R (RELEASE)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """


@dataclass
class Past(TemporalOp):
    """
    Represents the "past" temporal operator (P).
    P φ means that φ held at some point in the past.
    """

    subformula: LTLFormula

    def to_string(self):
        return f"P({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "P(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">P (PAST)</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """

    def simplify(self):
        simplified_subformula = self.subformula.simplify()

        # P false ≡ false
        if (
            isinstance(simplified_subformula, BooleanConstant)
            and not simplified_subformula.value
        ):
            return BooleanConstant(False)

        # P P φ ≡ P φ (idempotence of past operator)
        if isinstance(simplified_subformula, Past):
            return simplified_subformula

        return Past(simplified_subformula)

    def to_nnf(self):
        return Past(self.subformula.to_nnf())


@dataclass
class Since(BinaryTemporalOp):
    """
    Represents the "since" temporal operator (S).
    φ1 S φ2 means that there exists a past position where φ2 holds,
    and φ1 holds at all positions in between.
    """

    left: LTLFormula
    right: LTLFormula

    def to_string(self):
        return f"({self.left.to_string()} S {self.right.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + "(")
        self.left._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + "  S")
        self.right._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-temporal">S (SINCE)</span>
            <div class="node-children">
                <span class="tree-line">├─ </span>{self.left.to_html().replace('<div class="formula-node">', '<div class="formula-node" style="margin-bottom: 6px;">')}
                <span class="tree-line">└─ </span>{self.right.to_html()}
            </div>
        </div>
        """

    def simplify(self):
        left_simplified = self.left.simplify()
        right_simplified = self.right.simplify()

        # φ S true ≡ true (since true was true at some point in the past)
        if isinstance(right_simplified, BooleanConstant) and right_simplified.value:
            return BooleanConstant(True)

        # false S φ ≡ false (since false can never hold at all positions in between)
        if isinstance(left_simplified, BooleanConstant) and not left_simplified.value:
            return BooleanConstant(False)

        # true S φ ≡ P φ (true holds at all positions, so we just need φ to hold at some point)
        if isinstance(left_simplified, BooleanConstant) and left_simplified.value:
            return Past(right_simplified)

        return Since(left_simplified, right_simplified)

    def to_nnf(self):
        return Since(self.left.to_nnf(), self.right.to_nnf())
