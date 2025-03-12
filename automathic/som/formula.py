"""
Second-Order Monadic Logic Formula Representations

This module provides classes for representing Second-Order Monadic Logic (SOM) formulas
over words, supporting:

- Basic formula types (predicates, negation, conjunction, etc.)
- First-order quantifiers over positions
- Second-order quantifiers over sets of positions
- Set membership predicates
- Symbol predicates and position relations

SOM formulas can express all regular languages, while FO formulas are limited to
star-free regular languages.
"""

from dataclasses import dataclass


class SOMFormula:
    """
    Base class for all Second-Order Monadic Logic formulas.

    Second-Order Monadic Logic extends First-Order Logic by allowing quantification
    over sets of positions (monadic predicates) in addition to individual positions.

    This enables expressing properties like:
    - "There exists a set X of positions such that every position in X has symbol 'a'"
    - "For all sets X, if all positions in X have symbol 'a', then X is finite"

    SOM corresponds to all regular languages, while FO corresponds only to star-free
    regular languages.
    """

    def get_variables(self):
        """
        Extract all first-order variables used in the formula.

        Returns:
            set: A set of variable names used in the formula
        """
        if isinstance(self, Predicate):
            # For predicates, return the variable being tested
            return {self.variable}

        elif isinstance(self, SymbolPredicate):
            # For SymbolPredicates, also just the variable being tested
            return {self.variable}

        elif isinstance(self, Relation):
            # For relations, include both sides if they are variables (not constants)
            variables = set()
            # Check if left side is a variable (not a number or expression)
            if (
                isinstance(self.left, str)
                and not self.left.isdigit()
                and not "+" in self.left
            ):
                variables.add(self.left)

            # Check if right side is a variable (not a number or expression)
            if (
                isinstance(self.right, str)
                and not self.right.isdigit()
                and not "+" in self.right
            ):
                variables.add(self.right)

            return variables

        elif isinstance(self, SetMembership):
            # For set membership, just the position variable
            return {self.position_variable}

        elif isinstance(self, Negation):
            # For negation, recursively get variables from the subformula
            return self.subformula.get_variables()

        elif isinstance(self, Conjunction) or isinstance(self, Disjunction):
            # For binary operations, combine variables from both sides
            return self.left.get_variables().union(self.right.get_variables())

        elif isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            # For first-order quantifiers, include the quantified variable and variables from subformula
            return {self.variable}.union(self.subformula.get_variables())

        elif isinstance(self, ExistentialSetQuantifier) or isinstance(
            self, UniversalSetQuantifier
        ):
            # For second-order quantifiers, get variables from subformula (set variables handled separately)
            return self.subformula.get_variables()

        else:
            return set()

    def get_set_variables(self):
        """
        Extract all set variables used in the formula.

        Returns:
            set: A set of set variable names used in the formula
        """
        if isinstance(self, SetMembership):
            # For set membership predicates, return the set variable
            return {self.set_variable}

        elif isinstance(self, Negation):
            # For negation, recursively get set variables from the subformula
            return self.subformula.get_set_variables()

        elif isinstance(self, Conjunction) or isinstance(self, Disjunction):
            # For binary operations, combine set variables from both sides
            return self.left.get_set_variables().union(self.right.get_set_variables())

        elif isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            # For first-order quantifiers, get set variables from subformula
            return self.subformula.get_set_variables()

        elif isinstance(self, ExistentialSetQuantifier) or isinstance(
            self, UniversalSetQuantifier
        ):
            # For second-order quantifiers, include the quantified variable and variables from subformula
            return {self.set_variable}.union(self.subformula.get_set_variables())
            # return self.subformula.get_set_variables()

        else:
            # Base formulas have no set variables
            return set()

    def get_free_variables(self):
        """
        Extract all free first-order variables used in the formula.

        Returns:
            set: A set of free variable names used in the formula
        """
        all_variables = self.get_variables()
        bound_variables = self._get_bound_variables()
        return all_variables - bound_variables

    def get_free_set_variables(self):
        """
        Extract all free set variables used in the formula.

        Returns:
            set: A set of free set variable names used in the formula
        """
        all_set_variables = self.get_set_variables()
        bound_set_variables = self._get_bound_set_variables()
        return all_set_variables - bound_set_variables

    def _get_bound_variables(self):
        """
        Helper method to get all first-order variables that are bound by quantifiers.

        Returns:
            set: A set of variables bound by quantifiers in the formula
        """
        if (
            isinstance(self, Predicate)
            or isinstance(self, Relation)
            or isinstance(self, SetMembership)
        ):
            # Atomic formulas don't bind any variables
            return set()

        elif isinstance(self, Negation):
            # Negation doesn't bind any new variables, pass through
            return self.subformula._get_bound_variables()

        elif isinstance(self, Conjunction) or isinstance(self, Disjunction):
            # Combine bound variables from both sides
            left_bound = self.left._get_bound_variables()
            right_bound = self.right._get_bound_variables()
            return left_bound.union(right_bound)

        elif isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            # Add this quantified variable to the bound variables from subformula
            subformula_bound = self.subformula._get_bound_variables()
            return {self.variable}.union(subformula_bound)

        elif isinstance(self, ExistentialSetQuantifier) or isinstance(
            self, UniversalSetQuantifier
        ):
            # Second-order quantifiers don't bind first-order variables
            return self.subformula._get_bound_variables()

        else:
            return set()

    def _get_bound_set_variables(self):
        """
        Helper method to get all set variables that are bound by quantifiers.

        Returns:
            set: A set of set variables bound by quantifiers in the formula
        """
        if (
            isinstance(self, Predicate)
            or isinstance(self, Relation)
            or isinstance(self, SetMembership)
        ):
            # Atomic formulas don't bind any set variables
            return set()

        elif isinstance(self, Negation):
            # Negation doesn't bind any new variables, pass through
            return self.subformula._get_bound_set_variables()

        elif isinstance(self, Conjunction) or isinstance(self, Disjunction):
            # Combine bound variables from both sides
            left_bound = self.left._get_bound_set_variables()
            right_bound = self.right._get_bound_set_variables()
            return left_bound.union(right_bound)

        elif isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            # First-order quantifiers don't bind set variables
            return self.subformula._get_bound_set_variables()

        elif isinstance(self, ExistentialSetQuantifier) or isinstance(
            self, UniversalSetQuantifier
        ):
            # Add this quantified set variable to the bound set variables from subformula
            subformula_bound = self.subformula._get_bound_set_variables()
            return {self.set_variable}.union(subformula_bound)

        else:
            return set()

    def is_sentence(self):
        """
        Check if the formula is a sentence (has no free variables of any kind).

        Returns:
            bool: True if the formula is a sentence, False otherwise
        """
        # Check both first-order and second-order free variables
        return (not self.get_free_variables()) and (not self.get_free_set_variables())

    def get_alphabet(self):
        """
        Extract all symbols used in the formula's predicates.
        Specifically looks for predicates with names starting with 'Q'.

        Returns:
            list: A list of symbols in the alphabet used in the formula
        """
        symbols = set()

        if isinstance(self, SymbolPredicate):
            symbols.add(self.symbol)

        elif isinstance(self, Negation):
            # Recursively get symbols from the subformula
            symbols.update(self.subformula.get_alphabet())

        elif isinstance(self, Conjunction) or isinstance(self, Disjunction):
            # Combine symbols from both sides
            symbols.update(self.left.get_alphabet())
            symbols.update(self.right.get_alphabet())

        elif (
            isinstance(self, ExistentialQuantifier)
            or isinstance(self, UniversalQuantifier)
            or isinstance(self, ExistentialSetQuantifier)
            or isinstance(self, UniversalSetQuantifier)
        ):
            # Get symbols from subformula
            symbols.update(self.subformula.get_alphabet())

        # Return as a sorted list for consistent ordering
        return sorted(list(symbols))

    def to_basic_form(self):
        """
        Convert formula to a simplified form using only:
        - negation (¬)
        - conjunction (∧)
        - existential quantifiers (∃)
        - less-than relations (<)

        This applies the following transformations:
        - A ∨ B  ≡  ¬(¬A ∧ ¬B)      (De Morgan's law)
        - A → B  ≡  ¬(A ∧ ¬B)       (Implication definition)
        - A ⟺ B  ≡  ¬(¬(¬A ∧ ¬B) ∧ ¬(A ∧ B))  (Equivalence definition)
        - ∀x.A   ≡  ¬∃x.¬A          (Quantifier duality)
        - x ≤ y  ≡  x < y ∨ x = y   (Relation transformations)
        - x > y  ≡  y < x           (Relation transformations)
        - x ≥ y  ≡  y < x ∨ x = y   (Relation transformations)
        - x = y  ≡  ¬(x < y) ∧ ¬(y < x)  (Relation transformations)
        - x = y+1 ≡  y < x ∧ ¬∃z.(y < z ∧ z < x)  (Successor relation)
        - y = x+1 ≡  x < y ∧ ¬∃z.(x < z ∧ z < y)  (Predecessor relation)

        Returns:
            FOFormula: A simplified formula using only ¬, ∧, ∃, and <
        """
        if isinstance(self, Predicate) or isinstance(self, SetMembership):
            # Atomic formulas without relations remain unchanged
            return self

        elif isinstance(self, Relation):
            left, op, right = self.left, self.operator, self.right

            # Check if this is a successor relation (x = y+1)
            if op == "=" and right.endswith("+1"):
                base_var = right[:-2]  # Remove "+1"
                # x = y+1 means: y < x AND NOT EXISTS z: (y < z < x)
                return Conjunction(
                    Relation(base_var, "<", left),
                    Negation(
                        ExistentialQuantifier(
                            "z",
                            Conjunction(
                                Relation(base_var, "<", "z"), Relation("z", "<", left)
                            ),
                        )
                    ),
                )
            # Check if this is a predecessor relation (y = x+1)
            elif op == "=" and left.endswith("+1"):
                base_var = left[:-2]  # Remove "+1"
                # y = x+1 means: x < y AND NOT EXISTS z: (x < z < y)
                return Conjunction(
                    Relation(right, "<", base_var),
                    Negation(
                        ExistentialQuantifier(
                            "z",
                            Conjunction(
                                Relation(right, "<", "z"), Relation("z", "<", base_var)
                            ),
                        )
                    ),
                )
            # Other relations can be directly converted
            elif op == "<=":
                # x ≤ y means: x < y OR x = y
                # where x = y means NOT(x < y) AND NOT(y < x)
                equal_part = Conjunction(
                    Negation(Relation(left, "<", right)),
                    Negation(Relation(right, "<", left)),
                )
                return Disjunction(
                    Relation(left, "<", right), equal_part
                ).to_basic_form()
            elif op == ">=":
                # x ≥ y means: y < x OR x = y
                equal_part = Conjunction(
                    Negation(Relation(left, "<", right)),
                    Negation(Relation(right, "<", left)),
                )
                return Disjunction(
                    Relation(right, "<", left), equal_part
                ).to_basic_form()
            elif op == ">":
                # x > y means: y < x
                return Relation(right, "<", left)
            elif op == "=":
                # x = y means: NOT(x < y) AND NOT(y < x)
                return Conjunction(
                    Negation(Relation(left, "<", right)),
                    Negation(Relation(right, "<", left)),
                )
            elif op == "<":
                # Already in canonical form
                return self
            else:
                raise ValueError(f"Unsupported operator: {op}")

        elif isinstance(self, Negation):
            # Apply negation to the simplified subformula
            return Negation(self.subformula.to_basic_form())

        elif isinstance(self, Conjunction):
            # Apply recursively to both sides
            return Conjunction(self.left.to_basic_form(), self.right.to_basic_form())

        elif isinstance(self, Disjunction):
            # A ∨ B ≡ ¬(¬A ∧ ¬B)
            left_simplified = self.left.to_basic_form()
            right_simplified = self.right.to_basic_form()
            return Negation(
                Conjunction(Negation(left_simplified), Negation(right_simplified))
            )

        elif isinstance(self, Implication):
            # A → B ≡ ¬(A ∧ ¬B)
            left_simplified = self.left.to_basic_form()
            right_simplified = self.right.to_basic_form()
            return Negation(Conjunction(left_simplified, Negation(right_simplified)))

        elif isinstance(self, Equivalence):
            # A ⟺ B ≡ (A → B) ∧ (B → A) ≡ ¬(A ∧ ¬B) ∧ ¬(B ∧ ¬A)
            left_simplified = self.left.to_basic_form()
            right_simplified = self.right.to_basic_form()

            # (A → B) part: ¬(A ∧ ¬B)
            left_implication = Negation(
                Conjunction(left_simplified, Negation(right_simplified))
            )

            # (B → A) part: ¬(B ∧ ¬A)
            right_implication = Negation(
                Conjunction(right_simplified, Negation(left_simplified))
            )

            return Conjunction(left_implication, right_implication)

        elif isinstance(self, ExistentialQuantifier):
            # Existential quantifier stays, but simplify subformula
            return ExistentialQuantifier(self.variable, self.subformula.to_basic_form())

        elif isinstance(self, UniversalQuantifier):
            # ∀x.A ≡ ¬∃x.¬A
            return Negation(
                ExistentialQuantifier(
                    self.variable, Negation(self.subformula.to_basic_form())
                )
            )

        elif isinstance(self, ExistentialSetQuantifier):
            # Same for second-order existential
            return ExistentialSetQuantifier(
                self.set_variable, self.subformula.to_basic_form()
            )

        elif isinstance(self, UniversalSetQuantifier):
            # Same for second-order universal
            return Negation(
                ExistentialSetQuantifier(
                    self.set_variable, Negation(self.subformula.to_basic_form())
                )
            )

        else:
            raise ValueError(f"Unsupported formula type: {type(self).__name__}")

    def simplify(self):
        """
        Simplifies a formula in SOM by applying basic logical transformations
        while maintaining the SOM syntax restrictions (existential quantifiers,
        conjunctions, negations, and less-than relations).

        This includes:
        - Eliminating double negations
        - Simplifying Boolean tautologies and contradictions
        - Simplifying redundant conjunctions
        - Handling set membership predicates

        Returns:
            SOMFormula: A simplified equivalent formula
        """
        if (
            isinstance(self, Predicate)
            or isinstance(self, Relation)
            or isinstance(self, SetMembership)
        ):
            # Atomic formulas remain unchanged
            return self

        elif isinstance(self, Negation):
            # First simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # Eliminate double negation: ¬¬φ ≡ φ
            if isinstance(simplified_subformula, Negation):
                return simplified_subformula.subformula

            # Apply negation to the simplified subformula
            return Negation(simplified_subformula)

        elif isinstance(self, Conjunction):
            # Simplify both sides
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()

            # Check if either side is a contradiction, which makes the whole conjunction false
            # Note: We would need a representation for False, which isn't defined yet

            # Eliminate redundant conjunctions: A ∧ A ≡ A
            if left_simplified.to_string() == right_simplified.to_string():
                return left_simplified

            # Otherwise, keep the simplified conjunction
            return Conjunction(left_simplified, right_simplified)

        elif isinstance(self, Disjunction):
            # In SOM, disjunction is represented as ¬(¬A ∧ ¬B)
            # Simplify both sides first
            left_simplified = self.left.simplify()
            right_simplified = self.right.simplify()

            # Convert disjunction to the SOM form and simplify again
            return Negation(
                Conjunction(Negation(left_simplified), Negation(right_simplified))
            ).simplify()

        elif isinstance(self, ExistentialQuantifier):
            # Simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # Check if the variable is actually used in the subformula
            if self.variable not in simplified_subformula.get_variables():
                # If the variable isn't used, we can remove the quantifier
                return simplified_subformula

            return ExistentialQuantifier(self.variable, simplified_subformula)

        elif isinstance(self, UniversalQuantifier):
            # In SOM, universal quantifiers are represented as ¬∃x.¬φ
            # First simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # Check if the variable is actually used in the subformula
            if self.variable not in simplified_subformula.get_variables():
                # If the variable isn't used, we can remove the quantifier
                return simplified_subformula

            # Convert to SOM form and simplify again
            negated_subformula = Negation(simplified_subformula)
            existential = ExistentialQuantifier(self.variable, negated_subformula)
            return Negation(existential).simplify()

        elif isinstance(self, ExistentialSetQuantifier):
            # Simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # Check if the set variable is actually used in the subformula
            if self.set_variable not in simplified_subformula.get_set_variables():
                # If the set variable isn't used, we can remove the quantifier
                return simplified_subformula

            return ExistentialSetQuantifier(self.set_variable, simplified_subformula)

        elif isinstance(self, UniversalSetQuantifier):
            # In SOM, universal set quantifiers are represented as ¬∃X.¬φ
            # First simplify the subformula
            simplified_subformula = self.subformula.simplify()

            # Check if the set variable is actually used in the subformula
            if self.set_variable not in simplified_subformula.get_set_variables():
                # If the set variable isn't used, we can remove the quantifier
                return simplified_subformula

            # Convert to SOM form and simplify again
            negated_subformula = Negation(simplified_subformula)
            existential = ExistentialSetQuantifier(
                self.set_variable, negated_subformula
            )
            return Negation(existential).simplify()

        else:
            # For any unsupported formula type, return as is
            return self

    def to_fsa(self, alphabet=None):
        """
        Convert formula to equivalent Finite State Automaton

        Args:
            alphabet (list): List of symbols in the alphabet (default: None)
        """
        from automathic.som.translator import convert_som_to_fsa

        return convert_som_to_fsa(self, alphabet)

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

                # Highlight logical operators and quantifiers
                formula_str = formula_str.replace("∧", f"{BLUE}∧{ENDC}")
                formula_str = formula_str.replace("∨", f"{YELLOW}∨{ENDC}")
                formula_str = formula_str.replace("¬", f"{RED}¬{ENDC}")
                formula_str = formula_str.replace("∃", f"{GREEN}∃{ENDC}")
                formula_str = formula_str.replace("∀", f"{GREEN}∀{ENDC}")
                formula_str = formula_str.replace("∈", f"{BLUE}∈{ENDC}")

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
        if isinstance(self, Predicate):
            return f"{self.name}({self.variable})"
        elif isinstance(self, SetMembership):
            return f"{self.position_variable} in {self.set_variable}"
        elif isinstance(self, Negation):
            return f"!({self.subformula.to_canonical_string()})"
        elif isinstance(self, Conjunction):
            return f"({self.left.to_canonical_string()} and {self.right.to_canonical_string()})"
        elif isinstance(self, Disjunction):
            return f"({self.left.to_canonical_string()} or {self.right.to_canonical_string()})"
        elif isinstance(self, ExistentialQuantifier):
            return f"exists {self.variable} ({self.subformula.to_canonical_string()})"
        elif isinstance(self, UniversalQuantifier):
            return f"forall {self.variable} ({self.subformula.to_canonical_string()})"
        elif isinstance(self, ExistentialSetQuantifier):
            return (
                f"exists {self.set_variable} ({self.subformula.to_canonical_string()})"
            )
        elif isinstance(self, UniversalSetQuantifier):
            return (
                f"forall {self.set_variable} ({self.subformula.to_canonical_string()})"
            )
        elif isinstance(self, Relation):
            return f"{self.left} {self.operator} {self.right}"
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
            .node-quantifier {
                background-color: #e6f3ff;
                color: #0066cc;
                font-weight: bold;
            }
            .node-set-quantifier {
                background-color: #e6ffff;
                color: #006666;
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
            .node-predicate {
                background-color: #e6ffe6;
                color: #006600;
            }
            .node-relation {
                background-color: #f9f9f9;
                color: #333333;
            }
            .node-set-membership {
                background-color: #e6e6ff;
                color: #000066;
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


@dataclass
class SetMembership(SOMFormula):
    """
    Represents a set membership predicate (position x is in set X).
    """

    position_variable: str
    set_variable: str

    def to_string(self):
        return f"{self.position_variable} ∈ {self.set_variable}"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + self.to_string())

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return (
            self.position_variable == other.position_variable
            and self.set_variable == other.set_variable
        )

    def __lt__(self, other):
        return (self.set_variable, self.position_variable) < (
            other.set_variable,
            other.position_variable,
        )

    def __str__(self):
        return f"{self.set_variable}({self.position_variable})"

    def __repr__(self):
        return self.__str__()

    def to_html(self):
        return f'<div class="formula-node"><span class="node-content node-set-membership">{self.position_variable} ∈ {self.set_variable}</span></div>'


@dataclass
class ExistentialSetQuantifier(SOMFormula):
    """
    Represents an existential quantifier over sets (∃X.φ).
    """

    set_variable: str
    subformula: SOMFormula

    def to_string(self):
        return f"∃{self.set_variable}.({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + f"∃{self.set_variable}.(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-set-quantifier">∃{self.set_variable}</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class UniversalSetQuantifier(SOMFormula):
    """
    Represents a universal quantifier over sets (∀X.φ).
    """

    set_variable: str
    subformula: SOMFormula

    def to_string(self):
        return f"∀{self.set_variable}.({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + f"∀{self.set_variable}.(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-set-quantifier">∀{self.set_variable}</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


# First-Order Logic formula classes are subclasses of SOMFormula
class FOFormula(SOMFormula):
    """
    Base class for First-Order Logic formulas.

    First-Order Logic over words allows us to express properties of words
    by quantifying over positions and stating relationships between them.

    Examples:
        - "There exists a position with symbol 'a'": ∃x.Qa(x)
        - "Every 'a' is followed by a 'b'": ∀x.(Qa(x) → ∃y.(x < y ∧ Qb(y)))
    """

    def to_enf(self):
        """
        Convert formula to Existential Normal Form (ENF).

        In ENF:
        1. All quantifiers are at the beginning of the formula
        2. All quantifiers are existential (∃)
        3. No negations apply to subformulas with quantifiers

        ENF is useful for:
        - Matching the canonical form for automata conversion
        - Simplifying formula analysis and manipulation
        """
        # First push negations inward
        formula = self._push_negation_inward()

        # Then convert to prenex form (keeping universal quantifiers)
        prenex_form = formula.to_prenex_form()

        # Finally convert universal quantifiers to existential ones
        return prenex_form._convert_universals_to_existentials()

    def _push_negation_inward(self):
        """
        Push negations inward until they only apply to atomic formulas.

        This is a crucial step in formula normalization, applying rules like:
        - ¬¬φ ≡ φ
        - ¬(φ ∧ ψ) ≡ ¬φ ∨ ¬ψ
        - ¬(φ ∨ ψ) ≡ ¬φ ∧ ¬ψ
        - ¬∃x.φ(x) ≡ ∀x.¬φ(x)
        - ¬∀x.φ(x) ≡ ∃x.¬φ(x)

        Returns:
            FOFormula: Formula with negations pushed inward
        """
        if isinstance(self, Predicate) or isinstance(self, Relation):
            # Base case: atomic formulas remain unchanged
            return self
        elif isinstance(self, Negation):
            subformula = self.subformula
            if isinstance(subformula, Negation):
                # Double negation: ¬¬φ ≡ φ
                return subformula.subformula._push_negation_inward()
            elif isinstance(subformula, Conjunction):
                # De Morgan's law: ¬(φ ∧ ψ) ≡ ¬φ ∨ ¬ψ
                return Disjunction(
                    Negation(subformula.left)._push_negation_inward(),
                    Negation(subformula.right)._push_negation_inward(),
                )
            elif isinstance(subformula, Disjunction):
                # De Morgan's law: ¬(φ ∨ ψ) ≡ ¬φ ∧ ¬ψ
                return Conjunction(
                    Negation(subformula.left)._push_negation_inward(),
                    Negation(subformula.right)._push_negation_inward(),
                )
            elif isinstance(subformula, ExistentialQuantifier):
                # ¬∃x.φ(x) ≡ ∀x.¬φ(x)
                return UniversalQuantifier(
                    subformula.variable,
                    Negation(subformula.subformula)._push_negation_inward(),
                )
            elif isinstance(subformula, UniversalQuantifier):
                # ¬∀x.φ(x) ≡ ∃x.¬φ(x)
                return ExistentialQuantifier(
                    subformula.variable,
                    Negation(subformula.subformula)._push_negation_inward(),
                )
            else:
                # For any other formula type, keep the negation
                return Negation(subformula._push_negation_inward())
        elif isinstance(self, Conjunction):
            return Conjunction(
                self.left._push_negation_inward(), self.right._push_negation_inward()
            )
        elif isinstance(self, Disjunction):
            return Disjunction(
                self.left._push_negation_inward(), self.right._push_negation_inward()
            )
        elif isinstance(self, ExistentialQuantifier):
            return ExistentialQuantifier(
                self.variable, self.subformula._push_negation_inward()
            )
        elif isinstance(self, UniversalQuantifier):
            return UniversalQuantifier(
                self.variable, self.subformula._push_negation_inward()
            )
        else:
            raise ValueError(f"Unsupported formula type: {type(self).__name__}")

    def to_prenex_form(self):
        """
        Convert formula to prenex form (quantifiers at front).

        A formula is in prenex form when all quantifiers appear at the beginning,
        followed by a quantifier-free matrix. For example:
        - ∃x.∀y.(P(x) ∧ Q(y)) is in prenex form
        - (∃x.P(x)) ∧ Q(y) is not in prenex form

        This version keeps universal quantifiers as universal.

        Returns:
            FOFormula: Formula in prenex form
        """
        prenex_form, _ = self._to_prenex_form_helper()
        return prenex_form

    def _to_prenex_form_helper(self):
        """
        Helper for prenex form conversion.

        Returns:
            tuple: (prenex_formula, matrix)
            - prenex_formula: Formula with all quantifiers at front
            - matrix: The quantifier-free part of the formula
        """
        if isinstance(self, Predicate) or isinstance(self, Relation):
            # Base case: atomic formulas have no quantifiers
            return self, self

        elif isinstance(self, Negation):
            prenex, matrix = self.subformula._to_prenex_form_helper()
            if isinstance(prenex, ExistentialQuantifier) or isinstance(
                prenex, UniversalQuantifier
            ):
                # If subformula had quantifiers, negate only the matrix
                return prenex, Negation(matrix)
            else:
                # Otherwise negate the whole formula
                return Negation(prenex), Negation(matrix)

        elif isinstance(self, Conjunction):
            left_prenex, left_matrix = self.left._to_prenex_form_helper()
            right_prenex, right_matrix = self.right._to_prenex_form_helper()

            # Combine matrices
            combined_matrix = Conjunction(left_matrix, right_matrix)

            # Pull quantifiers from left and right
            result = combined_matrix
            if isinstance(left_prenex, ExistentialQuantifier) or isinstance(
                left_prenex, UniversalQuantifier
            ):
                result = self._pull_quantifiers_prenex(left_prenex, result)
            if isinstance(right_prenex, ExistentialQuantifier) or isinstance(
                right_prenex, UniversalQuantifier
            ):
                result = self._pull_quantifiers_prenex(right_prenex, result)

            # Extract matrix from final result
            _, final_matrix = result._get_innermost_formula()

            return result, final_matrix

        elif isinstance(self, Disjunction):
            # Same as conjunction but with disjunction
            left_prenex, left_matrix = self.left._to_prenex_form_helper()
            right_prenex, right_matrix = self.right._to_prenex_form_helper()

            combined_matrix = Disjunction(left_matrix, right_matrix)

            result = combined_matrix
            if isinstance(left_prenex, ExistentialQuantifier) or isinstance(
                left_prenex, UniversalQuantifier
            ):
                result = self._pull_quantifiers_prenex(left_prenex, result)
            if isinstance(right_prenex, ExistentialQuantifier) or isinstance(
                right_prenex, UniversalQuantifier
            ):
                result = self._pull_quantifiers_prenex(right_prenex, result)

            _, final_matrix = result._get_innermost_formula()

            return result, final_matrix

        elif isinstance(self, ExistentialQuantifier):
            inner_prenex, matrix = self.subformula._to_prenex_form_helper()

            # Create a new existential quantifier with the inner formula
            result = ExistentialQuantifier(self.variable, inner_prenex)

            return result, matrix

        elif isinstance(self, UniversalQuantifier):
            inner_prenex, matrix = self.subformula._to_prenex_form_helper()

            # Create a new universal quantifier with the inner formula
            # KEEP universal quantifier in prenex form
            result = UniversalQuantifier(self.variable, inner_prenex)

            return result, matrix

        else:
            raise ValueError(f"Unsupported formula type: {type(self).__name__}")

    def _pull_quantifiers_prenex(self, quantifier_formula, target_formula):
        """
        Pull quantifiers from quantifier_formula to the front of target_formula.
        This version maintains the type of quantifier (existential or universal).
        """
        if isinstance(quantifier_formula, ExistentialQuantifier):
            # Create a new existential quantifier with the target formula
            return ExistentialQuantifier(
                quantifier_formula.variable,
                self._pull_quantifiers_prenex(
                    quantifier_formula.subformula, target_formula
                ),
            )
        elif isinstance(quantifier_formula, UniversalQuantifier):
            # Create a new universal quantifier with the target formula
            # KEEP universal quantifier during prenex conversion
            return UniversalQuantifier(
                quantifier_formula.variable,
                self._pull_quantifiers_prenex(
                    quantifier_formula.subformula, target_formula
                ),
            )
        else:
            # No more quantifiers, return target formula
            return target_formula

    def _convert_universals_to_existentials(self):
        """
        Convert universal quantifiers to existential quantifiers.
        Used as the final step in ENF conversion.
        """
        if isinstance(self, UniversalQuantifier):
            # ∀x.φ(x) ≡ ¬∃x.¬φ(x)
            negated_subformula = Negation(
                self.subformula._convert_universals_to_existentials()
            )
            return Negation(ExistentialQuantifier(self.variable, negated_subformula))
        elif isinstance(self, ExistentialQuantifier):
            return ExistentialQuantifier(
                self.variable, self.subformula._convert_universals_to_existentials()
            )
        elif isinstance(self, Negation):
            return Negation(self.subformula._convert_universals_to_existentials())
        elif isinstance(self, Conjunction):
            return Conjunction(
                self.left._convert_universals_to_existentials(),
                self.right._convert_universals_to_existentials(),
            )
        elif isinstance(self, Disjunction):
            return Disjunction(
                self.left._convert_universals_to_existentials(),
                self.right._convert_universals_to_existentials(),
            )
        else:
            # Predicates and Relations don't contain quantifiers
            return self

    def _get_innermost_formula(self):
        """
        Extract the innermost formula (matrix) from a quantified formula.

        Returns:
            tuple: (quantified_formula, matrix)
        """
        if isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            inner, matrix = self.subformula._get_innermost_formula()
            return self, matrix
        else:
            return self, self


@dataclass
class Predicate(FOFormula):
    name: str
    variable: str

    def to_string(self):
        return f"{self.name}({self.variable})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + self.to_string())

    def to_html(self):
        return f'<div class="formula-node"><span class="node-content node-predicate">{self.name}({self.variable})</span></div>'


@dataclass
class SymbolPredicate(Predicate):
    name: str
    variable: str
    symbol: str

    def to_string(self):
        return f"{self.name}{self.symbol}({self.variable})"

    def to_html(self):
        return f'<div class="formula-node"><span class="node-content node-predicate">{self.name}{self.symbol}({self.variable})</span></div>'


@dataclass
class Negation(FOFormula):
    subformula: SOMFormula  # Note that subformula can be any SOMFormula

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
class Conjunction(FOFormula):
    left: SOMFormula  # Can be any SOMFormula
    right: SOMFormula  # Can be any SOMFormula

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
class Disjunction(FOFormula):
    left: SOMFormula  # Can be any SOMFormula
    right: SOMFormula  # Can be any SOMFormula

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
class Equivalence(FOFormula):
    """
    Represents logical equivalence (if and only if, ⟺).
    A ⟺ B is equivalent to (A → B) ∧ (B → A).
    """

    left: SOMFormula
    right: SOMFormula

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
class Implication(FOFormula):
    """
    Represents logical implication (→).
    A → B is equivalent to ¬A ∨ B.
    """

    left: SOMFormula
    right: SOMFormula

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
class ExistentialQuantifier(FOFormula):
    variable: str
    subformula: SOMFormula  # Can be any SOMFormula

    def to_string(self):
        return f"∃{self.variable}.({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + f"∃{self.variable}.(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-quantifier">∃{self.variable}</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class UniversalQuantifier(FOFormula):
    variable: str
    subformula: SOMFormula  # Can be any SOMFormula

    def to_string(self):
        return f"∀{self.variable}.({self.subformula.to_string()})"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + f"∀{self.variable}.(")
        self.subformula._to_ascii_helper(lines, indent + 2)
        lines.append(" " * indent + ")")

    def to_html(self):
        return f"""
        <div class="formula-node">
            <span class="node-content node-quantifier">∀{self.variable}</span>
            <div class="node-children">
                <span class="tree-line">└─ </span>{self.subformula.to_html()}
            </div>
        </div>
        """


@dataclass
class Relation(FOFormula):
    left: str
    operator: str
    right: str

    def to_string(self):
        return f"{self.left} {self.operator} {self.right}"

    def _to_ascii_helper(self, lines, indent):
        lines.append(" " * indent + self.to_string())

    def to_html(self):
        # Replace common operator symbols for HTML display
        op_map = {
            "<": "&lt;",
            "<=": "&le;",
            ">": "&gt;",
            ">=": "&ge;",
            "=": "=",
        }
        html_op = op_map.get(self.operator, self.operator)

        return f'<div class="formula-node"><span class="node-content node-relation">{self.left} {html_op} {self.right}</span></div>'
