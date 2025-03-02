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
$ git clone https://github.com/yourusername/FO-FSA.git
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
from automathic.fo.formula import Atom, Exists, ForAll, Conjunction

# Create the formula ∃x. P(x) ∧ Q(x)
p_x = Atom("P", ["x"])
q_x = Atom("Q", ["x"])
conjunction = Conjunction(p_x, q_x)
formula = Exists("x", conjunction)

print(formula)  # ∃x.(P(x) ∧ Q(x))
```

#### Converting FO Formula to FSA

```python
from automathic.fo.convert import formula_to_fsa
from automathic.fo.formula import *

# Create a formula: ∃x.(∀y. x < y)
formula = Exists("x", ForAll("y", LessThan("x", "y")))

# Convert to FSA
fsa = formula_to_fsa(formula)

# Test the resulting automaton
print(fsa.accepts("01"))     # True: 0 < 1
print(fsa.accepts("0123"))   # True: 0 < all others
print(fsa.accepts("10"))     # False: not 1 < 0
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
    # Compare minimized automata structure
    # ... (implementation details)
```

## Documentation

For full documentation, including API reference and tutorials, visit our [documentation site](https://yourusername.github.io/FO-FSA/) (coming soon).

## Development

We welcome contributions to Automathic! Here are some guidelines for development:

### Code Style

We use:
- [Black](https://github.com/psf/black) for code formatting
- [Flake8](https://flake8.pycqa.org/en/latest/) for linting
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
