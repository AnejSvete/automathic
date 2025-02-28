from dataclasses import dataclass


class FOFormula:
    def get_variables(self):
        """
        Extract all variables used in the formula. This includes quantified variables
        and free variables used in predicates and relations.

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

        elif isinstance(self, Negation):
            # For negation, recursively get variables from the subformula
            return self.subformula.get_variables()

        elif isinstance(self, Conjunction) or isinstance(self, Disjunction):
            # For binary operations, combine variables from both sides
            return self.left.get_variables().union(self.right.get_variables())

        elif isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            # For quantifiers, include the quantified variable and variables from subformula
            return {self.variable}.union(self.subformula.get_variables())

        else:
            return set()

    def is_sentence(self):
        """
        Check if the formula is a sentence (has no free variables).
        A sentence is a formula where all variables are bound by quantifiers.

        Returns:
            bool: True if the formula is a sentence (no free variables), False otherwise
        """
        # Get all variables used in the formula
        all_variables = self.get_variables()

        # Get all bound variables (variables introduced by quantifiers)
        bound_variables = self._get_bound_variables()

        # If all variables are bound, then there are no free variables
        return all_variables.issubset(bound_variables)

    def _get_bound_variables(self):
        """
        Helper method to get all variables that are bound by quantifiers.

        Returns:
            set: A set of variables bound by quantifiers in the formula
        """
        if isinstance(self, Predicate) or isinstance(self, Relation):
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

        else:
            return set()

    def to_fo_less(self):
        """
        Convert a formula into one that only uses the < relation (canonical form for FO[<]).
        This transforms formulas with <=, >=, >, =, and successor (x = y+1) relations.
        """
        if isinstance(self, Relation):
            left, op, right = self.left, self.operator, self.right

            # Check if this is a successor relation (x = y+1)
            if op == "=" and right.endswith("+1"):
                base_var = right[:-2]  # Remove "+1"
                # x = y+1 means: y < x AND there's no z between them
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
                # y = x+1 means: x < y AND there's no z between them
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
                # x ≤ y means: x < y OR NOT(y < x)
                return Disjunction(
                    Relation(left, "<", right), Negation(Relation(right, "<", left))
                )
            elif op == ">=":
                # x ≥ y means: y < x OR NOT(x < y)
                return Disjunction(
                    Relation(right, "<", left), Negation(Relation(left, "<", right))
                )
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

        # Recursively transform subformulas
        elif isinstance(self, Negation):
            return Negation(self.subformula.to_fo_less())
        elif isinstance(self, Conjunction):
            return Conjunction(self.left.to_fo_less(), self.right.to_fo_less())
        elif isinstance(self, Disjunction):
            return Disjunction(self.left.to_fo_less(), self.right.to_fo_less())
        elif isinstance(self, ExistentialQuantifier):
            return ExistentialQuantifier(self.variable, self.subformula.to_fo_less())
        elif isinstance(self, UniversalQuantifier):
            return UniversalQuantifier(self.variable, self.subformula.to_fo_less())
        elif isinstance(self, Predicate):
            # Predicates don't contain relations, so they remain unchanged
            return self
        else:
            raise ValueError(f"Unsupported formula type: {type(self).__name__}")

    def to_enf(self):
        """
        Convert formula to Existential Normal Form (ENF).
        In ENF:
        1. All quantifiers are at the beginning of the formula
        2. All quantifiers are existential (∃)
        3. No negations apply to subformulas with quantifiers
        """
        # First push negations inward
        formula = self._push_negation_inward()

        # Then convert to prenex form (keeping universal quantifiers)
        prenex_form = formula.to_prenex_form()

        # Finally convert universal quantifiers to existential ones
        return prenex_form._convert_universals_to_existentials()

    def _push_negation_inward(self):
        """Push negations inward until they only apply to atomic formulas"""
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
        This version keeps universal quantifiers as universal.
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
            return Negation(
                ExistentialQuantifier(
                    self.variable, Negation(self.subformula)._push_negation_inward()
                )
            )  # Remove the _push_negation_inward() call here
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

    def to_fsa(self, alphabet=None):
        """Convert formula to equivalent Finite State Automaton

        Args:
            alphabet (list): List of symbols in the alphabet (default: None)
        """
        from fo_to_fsa import convert_fo_to_fsa

        return convert_fo_to_fsa(self, alphabet)

    def get_alphabet(self):
        """
        Extract all symbols used in the formula's predicates.
        Specifically looks for predicates with names starting with 'Q_'.

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

        elif isinstance(self, ExistentialQuantifier) or isinstance(
            self, UniversalQuantifier
        ):
            # Get symbols from subformula
            symbols.update(self.subformula.get_alphabet())

        # Return as a sorted list for consistent ordering
        return sorted(list(symbols))

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
    subformula: FOFormula

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
    left: FOFormula
    right: FOFormula

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
    left: FOFormula
    right: FOFormula

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
class ExistentialQuantifier(FOFormula):
    variable: str
    subformula: FOFormula

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
    subformula: FOFormula

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
        return f'<div class="formula-node"><span class="node-content node-relation">{self.left} {self.operator} {self.right}</span></div>'
