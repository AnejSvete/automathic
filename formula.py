from dataclasses import dataclass

from IPython.display import HTML


# === FO FORMULA AST REPRESENTATION ===
class FOFormula:
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
