# Automathic: A Library for Formal Language Theory

Automathic is a Python library for exploring formal language theory, including first-order logic formulas, finite state automata (FSA), and their relationships. It provides tools for creating, manipulating, and analyzing automata, as well as converting first-order logic formulas into their equivalent finite state automaton representations.

## Features

- **Finite State Automata**: Create, manipulate, and visualize FSAs
- **First-Order Logic**: Define and work with first-order logic formulas
- **FO to FSA Conversion**: Convert first-order logic formulas to equivalent finite state automata
- **Operations on FSAs**: Union, intersection, complement, minimization, and more
- **Visualization**: Text and graphical representations of automata

## Getting Started

### Installation

Clone the repository:

```bash
$ git clone https://github.com/anejsvete/FO-FSA.git
$ cd FO-FSA
```

It's recommended to create a new [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) before proceeding.

Install the package in editable mode:

```bash
$ pip install -e .
```

### Basic Usage

Here are some examples to get you started with Automathic:

#### Creating a Finite State Automaton

```python
from automathic.fsa.fsa import FiniteStateAutomaton

# Create a FSA with 3 states over alphabet {0, 1}
fsa = FiniteStateAutomaton(3, ['0', '1'])

# Set state 0 as the initial state
fsa.set_initial_state(0)

# Set state 2 as an accepting state
fsa.set_accepting_state(2)

# Add transitions
fsa.set_transition(0, '0', 0)  # Stay in state 0 on symbol '0'
fsa.set_transition(0, '1', 1)  # Go to state 1 on symbol '1'
fsa.set_transition(1, '0', 2)  # Go to state 2 on symbol '0'
fsa.set_transition(1, '1', 0)  # Go to state 0 on symbol '1'
fsa.set_transition(2, '0', 1)  # Go to state 1 on symbol '0'
fsa.set_transition(2, '1', 2)  # Stay in state 2 on symbol '1'

# Check if the FSA accepts a string
print(fsa.accepts('010'))  # True
print(fsa.accepts('011'))  # False
```

#### Working with First-Order Logic Formulas

```python
from automathic.fo.parser import parse_fo_formula

# Parse a formula from a string
formula_str = "exists x. (Qa(x) and forall y. (!Qa(y) or x = y))"
formula = parse_fo_formula(formula_str)

# Display the parsed formula (with color formatting in terminals)
print(formula)  # ∃x.(Qa(x) ∧ ∀y.(¬Qa(y) ∨ x = y))

# Convert to form with only "<" relation and simplify
simplified = formula.to_fo_less().simplify()
print(simplified)
```

## First-Order Logic and Finite State Automata

### The Connection Between Logic and Automata

First-order logic over words (FO) and finite state automata (FSA) are deeply connected. Büchi's theorem establishes that a language is definable in the Monadic Second-Order Logic (MSO) if and only if it is recognizable by a finite automaton. For first-order logic specifically:

- **FO[<]**: First-order logic with the "less than" relation corresponds to a proper subset of regular languages called the **star-free languages**.
- **Straubing's Construction**: This library implements Straubing's construction, which provides an algorithm to convert FO[<] formulas to equivalent FSAs.

### How the Conversion Works

The conversion from FO to FSA is compositional, meaning we:

1. Build automata for atomic formulas (like position predicates and ordering relations)
2. Combine these automata using operations that mirror logical operations:
   - Conjunction → Automata intersection
   - Disjunction → Automata union
   - Negation → Automata complement
   - Existential quantification → Projection and determinization

The resulting automaton accepts exactly the words that satisfy the original formula.

#### Converting FO Formula to FSA

```python
from automathic.fo.parser import parse_fo_formula

# Define a formula for "strings that start with 'a'"
formula_str = "exists x. (forall y. (!(y < x)) and Qa(x))"

# Parse the string into a formula object
formula = parse_fo_formula(formula_str)

# Convert to FSA over alphabet {a, b}
fsa = formula.to_fsa(alphabet=["a", "b"])

# Test the resulting automaton
print(fsa.accepts("ab"))     # True: starts with 'a'
print(fsa.accepts("abb"))    # True: starts with 'a'
print(fsa.accepts("ba"))     # False: doesn't start with 'a'
```

#### Working with Formula Collections

```python
from automathic.fo.parser import parse_fo_formula

# Language of strings with exactly one 'a'
formula_str = "exists x. (Qa(x) and forall y. (!Qa(y) or x = y))"
formula = parse_fo_formula(formula_str)

# Convert to FSA and reindex states for cleaner visualization
fsa = formula.to_fsa(alphabet=["a", "b"]).reindex()

# Test the automaton
print(fsa.accepts("a"))      # True: exactly one 'a'
print(fsa.accepts("aba"))    # False: more than one 'a'
print(fsa.accepts("bab"))    # True: exactly one 'a'
print(fsa.accepts("bbb"))    # False: no 'a's
```

### Advanced Operations

```python
# Minimizing an FSA
minimized = fsa.minimize()

# Creating the complement of an FSA
complement = fsa.complement()

# Intersecting two FSAs
intersection = fsa1.intersect(fsa2)

# Checking language equivalence via minimization
def are_equivalent(fsa1, fsa2):
    min1 = fsa1.minimize()
    min2 = fsa2.minimize()
    return min1.is_equivalent(min2)
```

## Documentation

For full documentation, including API reference and tutorials, visit our [documentation site](https://anejsvete.github.io/FO-FSA/) (coming soon).

## Development

We welcome contributions to Automathic! Here are some guidelines for development:

### Code Style

We use:
- [Black](https://github.com/psf/black) for code formatting
- [ruff](https://docs.astral.sh/ruff/) for linting
- [Pytest](https://docs.pytest.org) for unit testing

### Running Tests

```bash
$ pytest
```

### Future Plans

- Support for weighted finite state automata
- Integration with syntactic monoids
- More efficient algorithms for automata operations
- Additional visualization options

## Citing

If you use Automathic in your research, please cite us:

```
@software{automathic2025,
  author = {Anej Svete},
  title = {Automathic: A Library for Formal Language Theory},
  year = {2025},
  url = {https://github.com/anejsvete/FO-FSA}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.