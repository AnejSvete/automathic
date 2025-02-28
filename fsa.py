import json
from collections import defaultdict
from uuid import uuid4


class State:
    """
    Represents a state in a finite state automaton
    """

    def __init__(self, id, label=None):
        """
        Initialize a state

        Args:
            id: Numeric ID for the state
            label: Optional label for the state (defaults to str(id))
        """
        self.id = id
        self.label = label if label is not None else str(id)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        return False

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"State({self.id}, '{self.label}')"

    def __str__(self):
        return self.label


class FiniteStateAutomaton:
    def __init__(self, num_states, alphabet):
        """
        Initialize a finite state automaton

        Args:
            num_states: Number of states in the automaton
            alphabet: List of symbols in the alphabet or the size of the alphabet
        """
        # Handle both alphabet list and alphabet size
        if isinstance(alphabet, list):
            self.alphabet = alphabet
            self.alphabet_size = len(alphabet)
        else:
            # For backward compatibility, if an integer is passed
            self.alphabet_size = alphabet
            self.alphabet = list(range(alphabet))

        self.num_symbols = len(self.alphabet)

        # Create state objects
        self.states = [State(i) for i in range(num_states)]
        self.num_states = num_states

        # Initialize transitions dictionary: {(state, symbol): next_state}
        self.transitions = {}
        self.initial_state = None  # Single initial state
        self.accepting_states = set()

        # For visualization
        self.theme = "dark"  # Default theme

    def _get_state_obj(self, state_id):
        """Convert state ID to state object if needed"""
        if isinstance(state_id, State):
            return state_id
        elif isinstance(state_id, int) and 0 <= state_id < self.num_states:
            return self.states[state_id]
        else:
            raise ValueError(f"Invalid state: {state_id}")

    def set_state_label(self, state_id, label):
        """Set a human-readable label for a state"""
        state = self._get_state_obj(state_id)
        state.label = label

    def set_transition(self, from_state, symbol, to_state):
        """
        Set a transition from one state to another on a given symbol

        Args:
            from_state: Source state (int or State)
            symbol: Transition symbol (string or integer)
            to_state: Destination state (int or State)
        """
        from_obj = self._get_state_obj(from_state)
        to_obj = self._get_state_obj(to_state)
        self.transitions[(from_obj, symbol)] = to_obj

    def set_initial_state(self, state):
        """Set the initial state"""
        state_obj = self._get_state_obj(state)
        self.initial_state = state_obj

    def set_accepting_state(self, state):
        """Set a state as an accepting state"""
        state_obj = self._get_state_obj(state)
        self.accepting_states.add(state_obj)

    def transition(self, state, symbol):
        """Get the next state for a given state and symbol"""
        state_obj = self._get_state_obj(state)
        return self.transitions.get((state_obj, symbol), None)

    def accepts(self, input_string):
        """Check if the FSA accepts the given input string"""
        if self.initial_state is None:
            return False

        state = self.initial_state
        for symbol in input_string:
            state = self.transition(state, symbol)
            if state is None:
                return False
        return state in self.accepting_states

    def minimize(self):
        """
        Create a new minimized FSA using Hopcroft's algorithm.

        Returns:
            FiniteStateAutomaton: A minimized version of this FSA
        """
        # Start with the partition [accepting states, non-accepting states]
        partition = [
            self.accepting_states,
            set(self.states) - self.accepting_states,
        ]

        # Remove the empty set if one of the partitions is empty
        partition = [p for p in partition if p]

        # Repeatedly refine the partition until no more refinement is possible
        new_partition = []
        while new_partition != partition:
            new_partition = []
            for group in partition:
                # Split each group based on transitions
                split_groups = defaultdict(set)
                for state in group:
                    # Create a signature for this state based on its transitions to other groups
                    signature = tuple(
                        self.transition(state, symbol) for symbol in self.alphabet
                    )
                    split_groups[signature].add(state)

                # Add each split group to the new partition
                new_partition.extend(split_groups.values())

            partition = new_partition

        # If no states, return empty FSA
        if not partition:
            return FiniteStateAutomaton(0, self.alphabet)

        # Create a mapping from old states to new states
        state_map = {state: i for i, group in enumerate(partition) for state in group}

        # Create new minimal FSA
        result = FiniteStateAutomaton(len(partition), self.alphabet)

        # Map initial state
        if self.initial_state:
            result.set_initial_state(state_map[self.initial_state])

        # Map accepting states
        for old_state in self.accepting_states:
            result.set_accepting_state(state_map[old_state])

        # Map transitions
        for (src, symbol), dst in self.transitions.items():
            result.set_transition(state_map[src], symbol, state_map[dst])

        return result

    def get_transitions(self, q):
        """Yield transitions from state q"""
        state_obj = self._get_state_obj(q)
        for (src, symbol), dst in self.transitions.items():
            if src == state_obj:
                yield symbol, dst

    def intersect(self, other):
        """
        Create an FSA that accepts the intersection of languages accepted by self and other.
        Uses the standard product construction.
        """
        # Check if either automaton has no initial state
        if self.initial_state is None or other.initial_state is None:
            # Return an empty automaton
            return FiniteStateAutomaton(0, self.alphabet)

        # Create a new FSA with states being the product of the two input FSAs
        num_states = self.num_states * other.num_states
        result = FiniteStateAutomaton(num_states, self.alphabet)

        # Create a mapping from pairs of states to the new state IDs
        state_map = {}
        counter = 0
        for q1 in range(self.num_states):
            for q2 in range(other.num_states):
                state_map[(q1, q2)] = counter
                counter += 1

        # Set transitions
        for (q1, q2), q in state_map.items():
            for symbol in self.alphabet:
                next_q1 = self.transition(q1, symbol)
                next_q2 = other.transition(q2, symbol)

                if next_q1 is not None and next_q2 is not None:
                    next_q = state_map[(next_q1.id, next_q2.id)]
                    result.set_transition(q, symbol, next_q)

        # Set initial state
        result.set_initial_state(
            state_map[(self.initial_state.id, other.initial_state.id)]
        )

        # Set accepting states - for intersection, a state is accepting if both components are accepting
        for (q1, q2), q in state_map.items():
            state1 = self.states[q1]
            state2 = other.states[q2]
            if state1 in self.accepting_states and state2 in other.accepting_states:
                result.set_accepting_state(q)

        return result

    def union(self, other):
        """
        Create an FSA that accepts the union of languages accepted by self and other.
        Uses the standard product construction.
        """
        # Check if either automaton has no initial state
        if self.initial_state is None and other.initial_state is None:
            # Return an empty automaton
            return FiniteStateAutomaton(0, self.alphabet)
        elif self.initial_state is None:
            return other.copy()
        elif other.initial_state is None:
            return self.copy()

        # Create a new FSA with states being the product of the two input FSAs
        num_states = self.num_states * other.num_states
        result = FiniteStateAutomaton(num_states, self.alphabet)

        # Create a mapping from pairs of states to the new state IDs
        state_map = {}
        counter = 0
        for q1 in range(self.num_states):
            for q2 in range(other.num_states):
                state_map[(q1, q2)] = counter
                counter += 1

        # Set transitions
        for (q1, q2), q in state_map.items():
            for symbol in self.alphabet:
                next_q1 = self.transition(q1, symbol)
                next_q2 = other.transition(q2, symbol)

                if next_q1 is not None and next_q2 is not None:
                    next_q = state_map[(next_q1.id, next_q2.id)]
                    result.set_transition(q, symbol, next_q)

        # Set initial state
        result.set_initial_state(
            state_map[(self.initial_state.id, other.initial_state.id)]
        )

        # Set accepting states - for union, a state is accepting if either component is accepting
        for (q1, q2), q in state_map.items():
            state1 = self.states[q1]
            state2 = other.states[q2]
            if state1 in self.accepting_states or state2 in other.accepting_states:
                result.set_accepting_state(q)

        return result

    def is_complete(self):
        """
        Check if the FSA is complete (has transitions for all symbols from all states)
        """
        for state in self.states:
            for symbol in self.alphabet:
                if self.transition(state, symbol) is None:
                    return False
        return True

    def complete(self):
        """
        Make the FSA complete by adding a sink state if needed.
        Returns a new complete FSA.
        """
        if self.is_complete():
            return self.copy()

        # Create a new FSA with an additional sink state
        result = FiniteStateAutomaton(self.num_states + 1, self.alphabet)
        sink_state = self.num_states  # The new sink state

        # Copy all existing transitions
        for state in range(self.num_states):
            for symbol in self.alphabet:
                next_state = self.transition(state, symbol)
                if next_state is not None:
                    result.set_transition(state, symbol, next_state)
                else:
                    result.set_transition(state, symbol, sink_state)

        # Add transitions from sink state to itself
        for symbol in self.alphabet:
            result.set_transition(sink_state, symbol, sink_state)

        # Copy initial state
        if self.initial_state is not None:
            result.set_initial_state(self.initial_state.id)

        # Copy accepting states
        for state in self.accepting_states:
            result.set_accepting_state(state.id)

        return result

    def complement(self):
        """
        Create an FSA that accepts exactly the strings rejected by self.
        Ensures the FSA is complete first.
        """
        # Make sure the FSA is complete
        complete_fsa = self.complete()

        # Create a new FSA with the same structure but complemented accepting states
        result = FiniteStateAutomaton(complete_fsa.num_states, complete_fsa.alphabet)

        # Copy all transitions
        for (src, symbol), dst in complete_fsa.transitions.items():
            result.set_transition(src.id, symbol, dst.id)

        # Copy initial state
        if complete_fsa.initial_state is not None:
            result.set_initial_state(complete_fsa.initial_state.id)

        # Complement accepting states
        for state in complete_fsa.states:
            if state not in complete_fsa.accepting_states:
                result.set_accepting_state(state.id)

        return result

    def copy(self):
        """
        Create a deep copy of this FSA
        """
        result = FiniteStateAutomaton(self.num_states, self.alphabet)

        # Copy transitions
        for (src, symbol), dst in self.transitions.items():
            result.set_transition(src.id, symbol, dst.id)

        # Copy initial state
        if self.initial_state is not None:
            result.set_initial_state(self.initial_state.id)

        # Copy accepting states
        for state in self.accepting_states:
            result.set_accepting_state(state.id)

        return result

    def project(self, position):
        """
        Project out a position from the FSA.
        """
        return self.copy()  # Simplified implementation

    def trim(self):
        """
        Create a new FSA with all unreachable and dead-end states removed.
        """
        # Find all reachable states from initial state
        reachable = set()
        if self.initial_state is not None:
            queue = [self.initial_state]
            while queue:
                state = queue.pop(0)
                if state not in reachable:
                    reachable.add(state)
                    for symbol in self.alphabet:
                        next_state = self.transition(state, symbol)
                        if next_state is not None and next_state not in reachable:
                            queue.append(next_state)

        # Find all states that can reach an accepting state
        # First, build a reverse transition map
        reverse_transitions = defaultdict(list)
        for (src, symbol), dst in self.transitions.items():
            reverse_transitions[dst].append((src, symbol))

        # Do a reverse search from accepting states
        can_reach_accepting = set()
        queue = list(self.accepting_states)
        while queue:
            state = queue.pop(0)
            if state not in can_reach_accepting:
                can_reach_accepting.add(state)
                for prev_state, _ in reverse_transitions[state]:
                    if prev_state not in can_reach_accepting:
                        queue.append(prev_state)

        # Keep only states that are both reachable and can reach an accepting state
        useful_states = reachable.intersection(can_reach_accepting)

        # If no useful states remain, return an empty automaton
        if not useful_states:
            return FiniteStateAutomaton(0, self.alphabet)

        # Create a new automaton with only the useful states
        # First, create a mapping from old state IDs to new state IDs
        old_to_new = {}
        for new_id, old_state in enumerate(sorted(useful_states, key=lambda s: s.id)):
            old_to_new[old_state] = State(new_id, old_state.label)

        # Create the new automaton
        result = FiniteStateAutomaton(len(useful_states), self.alphabet)

        # Copy transitions
        for (src, symbol), dst in self.transitions.items():
            if src in useful_states and dst in useful_states:
                result.set_transition(old_to_new[src], symbol, old_to_new[dst])

        # Copy initial state if it's useful
        if self.initial_state in useful_states:
            result.set_initial_state(old_to_new[self.initial_state])

        # Copy accepting states
        for state in self.accepting_states:
            if state in useful_states:
                result.set_accepting_state(old_to_new[state])

        return result

    def __str__(self):
        initial_id = self.initial_state.id if self.initial_state else None
        return f"FSA(states={self.num_states}, alphabet={self.alphabet}, initial={initial_id}, accepting={[s.id for s in self.accepting_states]})"

    def ascii(self):
        """
        Generate a clear ASCII representation of the FSA
        """
        lines = []
        lines.append("=" * 50)
        lines.append("FINITE STATE AUTOMATON")
        lines.append("=" * 50)

        # Basic information
        lines.append(f"States: {self.num_states} (0 to {self.num_states - 1})")
        lines.append(f"Alphabet: {self.alphabet}")

        initial_str = "None"
        if self.initial_state:
            initial_str = f"{self.initial_state.id} ({self.initial_state.label})"
        lines.append(f"Initial state: {initial_str}")

        accepting_str = "[]"
        if self.accepting_states:
            accepting_str = ", ".join(
                [
                    f"{s.id} ({s.label})"
                    for s in sorted(self.accepting_states, key=lambda s: s.id)
                ]
            )
        lines.append(f"Accepting states: [{accepting_str}]")
        lines.append("-" * 50)

        # Transition table
        lines.append("TRANSITION TABLE:")
        header = "State |" + "".join(f" {sym:^5} |" for sym in self.alphabet)
        lines.append(header)
        lines.append("-" * len(header))

        for state in self.states:
            # Mark initial and accepting states
            if state == self.initial_state and state in self.accepting_states:
                state_marker = f"→({state.id}:{state.label})←"
            elif state == self.initial_state:
                state_marker = f"→{state.id}:{state.label}  "
            elif state in self.accepting_states:
                state_marker = f" ({state.id}:{state.label})←"
            else:
                state_marker = f"  {state.id}:{state.label}  "

            # Trim if too long
            if len(state_marker) > 12:
                state_marker = state_marker[:9] + "..."

            row = f"{state_marker:12} |"

            for symbol in self.alphabet:
                next_state = self.transition(state, symbol)
                cell = "-"
                if next_state is not None:
                    cell = str(next_state.id)
                row += f" {cell:^4} |"
            lines.append(row)

        lines.append("-" * len(header))
        print("\n".join(lines))

    def _repr_html_(self):
        """
        When returned from a Jupyter cell, this will generate the FSA visualization
        """
        ret = []
        if self.num_states == 0:
            return "<code>Empty FSA</code>"

        if self.num_states > 64:
            return (
                "FSA too large to draw graphic, use fsa.ascii()<br />"
                + f"<code>FSA(states={self.num_states})</code>"
            )

        # Add node for initial state
        if self.initial_state is not None:
            q = self.initial_state
            if q in self.accepting_states:
                label = str(q)
                color = "af8dc3"  # Purple for initial+accepting
            else:
                label = str(q)
                color = "66c2a5"  # Green for initial

            ret.append(
                f'g.setNode("{q.id}", '
                + f'{{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )
            ret.append(f'g.node("{q.id}").style = "fill: #{color}"; \n')

        # Add nodes for normal states
        for q in self.states:
            if q in self.accepting_states:
                continue
            if q == self.initial_state:
                continue  # Skip initial state, already added
            ret.append(
                f'g.setNode("{q.id}",{{label:{json.dumps(str(q))},shape:"circle"}});\n'
            )
            ret.append(
                f'g.node("{q.id}").style = "fill: #8da0cb"; \n'
            )  # Blue for normal

        # Add nodes for accepting states
        for q in self.accepting_states:
            if q == self.initial_state:
                continue  # Skip initial+accepting, already added
            ret.append(
                f'g.setNode("{q.id}",{{label:{json.dumps(str(q))},shape:"circle"}});\n'
            )
            ret.append(
                f'g.node("{q.id}").style = "fill: #fc8d62"; \n'
            )  # Orange for accepting

        # Add edges for transitions
        for q in self.states:
            to = defaultdict(list)
            for symbol, next_state in self.get_transitions(q):
                to[next_state].append(str(symbol))

            for d, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label = json.dumps(", ".join(values))
                color = "rgb(192, 192, 192)" if self.theme == "dark" else "#333"
                edge_string = (
                    f'g.setEdge("{q.id}","{d.id}",{{arrowhead:"vee",'
                    + f'label:{label},"style": "stroke: {color}; fill: none;", '
                    + f'"labelStyle": "fill: {color}; stroke: {color}; ", '
                    + f'"arrowheadStyle": "fill: {color}; stroke: {color};"}});\n'
                )
                ret.append(edge_string)

        # # Add a special invisible node and edge for initial state marker
        # if self.initial_state is not None:
        #     color = "rgb(192, 192, 192)" if self.theme == "dark" else "#333"
        #     ini_id = self.initial_state.id
        #     ret.append(f'g.setNode("start", {{label:"", width:0, height:0}});\n')
        #     edge_string = (
        #         f'g.setEdge("start","{ini_id}",{{arrowhead:"vee",'
        #         + f'label:"","style": "stroke: {color}; fill: none;", '
        #         + f'"labelStyle": "fill: {color}; stroke: {color}; ", '
        #         + f'"arrowheadStyle": "fill: {color}; stroke: {color};"}});\n'
        #     )
        #     ret.append(edge_string)

        # If the machine is too big, don't attempt to display it
        if len(ret) > 256:
            return (
                "FSA too large to draw graphic, use fsa.ascii()<br />"
                + f"<code>FSA(states={self.num_states})</code>"
            )

        # Build the HTML with embedded JavaScript
        ret2 = [
            """
       <script>
       try {
       require.config({
       paths: {
       "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
       "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
       }
       });
       } catch {
       ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
       "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(
            function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
            }
        )
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        """
        ]

        obj = "fsa_" + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
          d3 = require('d3');
          dagreD3 = require('dagreD3');
        } catch (e) {
          // for google colab
          if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined"){
            d3 = window.d3;
            dagreD3 = window.dagreD3;
          } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
          }
        }
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        """
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            """
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate(
            (svg.attr("width")-g.graph().width*initialScale)/2,20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        })();

        </script>
        """
        )

        return "".join(ret2)
