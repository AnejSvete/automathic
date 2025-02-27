import json
from collections import defaultdict
from uuid import uuid4


class FiniteStateAutomaton:
    def __init__(self, num_states, num_symbols):
        self.num_states = num_states
        self.num_symbols = num_symbols
        self.alphabet = list(range(num_symbols))
        self.transitions = {state: {} for state in range(num_states)}
        self.initial_states = set()
        self.accepting_states = set()
        self.theme = "dark"
        self.q2str = {i: str(i) for i in range(num_states)}

    def set_transition(self, state, symbol, next_state):
        if state in self.transitions and symbol in self.alphabet:
            self.transitions[state][symbol] = next_state
        else:
            raise ValueError("Invalid state or symbol")

    def set_initial_state(self, state):
        self.initial_states.add(state)

    def set_accepting_state(self, state):
        self.accepting_states.add(state)

    def transition(self, state, symbol):
        return self.transitions.get(state, {}).get(symbol, None)

    def accepts(self, input_string):
        for start_state in self.initial_states:
            state = start_state
            for symbol in input_string:
                state = self.transition(state, symbol)
                if state is None:
                    break
            if state in self.accepting_states:
                return True
        return False

    def minimize(self):
        """
        Create a new minimized FSA using Hopcroft's algorithm.

        Returns:
            FiniteStateAutomaton: A minimized version of this FSA
        """
        # Start with the partition [accepting states, non-accepting states]
        partition = [
            self.accepting_states,
            set(range(self.num_states)) - self.accepting_states,
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
            return FiniteStateAutomaton(0, self.num_symbols)

        # Create a mapping from old states to new states
        state_map = {state: i for i, group in enumerate(partition) for state in group}

        # Create new minimal FSA
        result = FiniteStateAutomaton(len(partition), self.num_symbols)

        # Map initial and accepting states
        for old_state in self.initial_states:
            result.set_initial_state(state_map[old_state])

        for old_state in self.accepting_states:
            result.set_accepting_state(state_map[old_state])

        # Map transitions - here we need to iterate over the ORIGINAL transitions
        for old_state in range(self.num_states):
            for symbol, next_state in self.transitions.get(old_state, {}).items():
                if next_state is not None:  # Ensure transition exists
                    result.set_transition(
                        state_map[old_state], symbol, state_map[next_state]
                    )

        return result

    def __str__(self):
        return f"FSA(states={self.num_states}, alphabet={self.alphabet}, initial={self.initial_states}, accepting={self.accepting_states})"

    @property
    def n_states(self):
        return self.num_states

    @property
    def n_symbols(self):
        return self.num_symbols

    @property
    def Q(self):
        return set(range(self.num_states))

    @property
    def I(self):  # noqa: E741, E743
        return self.initial_states

    @property
    def F(self):
        return self.accepting_states

    def arcs(self, q):
        """Yield transitions from state q"""
        for symbol, next_state in self.transitions.get(q, {}).items():
            # Yield (symbol, next_state, weight=1.0)
            yield symbol, next_state, 1.0

    def intersect(self, other):
        """
        Create an FSA that accepts the intersection of languages accepted by self and other.
        Uses the standard product construction.
        """
        # Create a new FSA with states being the product of the two input FSAs
        num_states = self.num_states * other.num_states
        result = FiniteStateAutomaton(num_states, self.num_symbols)

        # Create a mapping from pairs of states to the new state IDs
        state_map = {}
        counter = 0
        for q1 in range(self.num_states):
            for q2 in range(other.num_states):
                state_map[(q1, q2)] = counter
                counter += 1

        # Set transitions
        for (q1, q2), q in state_map.items():
            for symbol in range(self.num_symbols):
                next_q1 = self.transition(q1, symbol)
                next_q2 = other.transition(q2, symbol)

                if next_q1 is not None and next_q2 is not None:
                    next_q = state_map[(next_q1, next_q2)]
                    result.set_transition(q, symbol, next_q)

        # Set initial states
        for q1 in self.initial_states:
            for q2 in other.initial_states:
                result.set_initial_state(state_map[(q1, q2)])

        # Set accepting states - for intersection, a state is accepting if both components are accepting
        for (q1, q2), q in state_map.items():
            if q1 in self.accepting_states and q2 in other.accepting_states:
                result.set_accepting_state(q)

        return result

    def union(self, other):
        """
        Create an FSA that accepts the union of languages accepted by self and other.
        Uses the standard product construction.
        """
        # Create a new FSA with states being the product of the two input FSAs
        num_states = self.num_states * other.num_states
        result = FiniteStateAutomaton(num_states, self.num_symbols)

        # Create a mapping from pairs of states to the new state IDs
        state_map = {}
        counter = 0
        for q1 in range(self.num_states):
            for q2 in range(other.num_states):
                state_map[(q1, q2)] = counter
                counter += 1

        # Set transitions
        for (q1, q2), q in state_map.items():
            for symbol in range(self.num_symbols):
                next_q1 = self.transition(q1, symbol)
                next_q2 = other.transition(q2, symbol)

                if next_q1 is not None and next_q2 is not None:
                    next_q = state_map[(next_q1, next_q2)]
                    result.set_transition(q, symbol, next_q)

        # Set initial states
        for q1 in self.initial_states:
            for q2 in other.initial_states:
                result.set_initial_state(state_map[(q1, q2)])

        # Set accepting states - for union, a state is accepting if either component is accepting
        for (q1, q2), q in state_map.items():
            if q1 in self.accepting_states or q2 in other.accepting_states:
                result.set_accepting_state(q)

        return result

    def is_complete(self):
        """
        Check if the FSA is complete (has transitions for all symbols from all states)
        """
        for state in range(self.num_states):
            for symbol in range(self.num_symbols):
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
        result = FiniteStateAutomaton(self.num_states + 1, self.num_symbols)
        sink_state = self.num_states  # The new sink state

        # Copy all existing transitions
        for state in range(self.num_states):
            for symbol in range(self.num_symbols):
                next_state = self.transition(state, symbol)
                if next_state is not None:
                    result.set_transition(state, symbol, next_state)
                else:
                    result.set_transition(state, symbol, sink_state)

        # Add transitions from sink state to itself
        for symbol in range(self.num_symbols):
            result.set_transition(sink_state, symbol, sink_state)

        # Copy initial and accepting states
        for state in self.initial_states:
            result.set_initial_state(state)
        for state in self.accepting_states:
            result.set_accepting_state(state)

        return result

    def complement(self):
        """
        Create an FSA that accepts exactly the strings rejected by self.
        Ensures the FSA is complete first.
        """
        # Make sure the FSA is complete
        complete_fsa = self.complete()

        # Create a new FSA with the same structure but complemented accepting states
        result = FiniteStateAutomaton(complete_fsa.num_states, complete_fsa.num_symbols)

        # Copy all transitions
        for state in range(complete_fsa.num_states):
            for symbol in range(complete_fsa.num_symbols):
                next_state = complete_fsa.transition(state, symbol)
                if next_state is not None:
                    result.set_transition(state, symbol, next_state)

        # Copy initial states
        for state in complete_fsa.initial_states:
            result.set_initial_state(state)

        # Complement accepting states
        for state in range(complete_fsa.num_states):
            if state not in complete_fsa.accepting_states:
                result.set_accepting_state(state)

        return result

    def copy(self):
        """
        Create a deep copy of this FSA
        """
        result = FiniteStateAutomaton(self.num_states, self.num_symbols)

        # Copy transitions
        for state in range(self.num_states):
            for symbol, next_state in self.transitions.get(state, {}).items():
                result.set_transition(state, symbol, next_state)

        # Copy initial and accepting states
        for state in self.initial_states:
            result.set_initial_state(state)
        for state in self.accepting_states:
            result.set_accepting_state(state)

        return result

    def project(self, position):
        """
        Project out a position from the FSA.
        This is a simplified implementation that assumes positions are independent.

        In a full implementation, this would involve more complex operations to
        handle dependencies between positions.
        """
        # For this simplified version, we'll just return a copy of the FSA
        # In a real implementation, we would use automata with epsilon transitions
        # or use the powerset construction
        return self.copy()

    def trim(self):
        """
        Create a new FSA with all unreachable and dead-end states removed.

        A state is unreachable if it cannot be reached from an initial state.
        A state is a dead-end if no accepting state can be reached from it.

        Returns:
            FiniteStateAutomaton: A trimmed version of this FSA
        """
        # Find all reachable states from initial states
        reachable = set()
        queue = list(self.initial_states)
        while queue:
            state = queue.pop(0)
            if state not in reachable:
                reachable.add(state)
                for symbol in range(self.num_symbols):
                    next_state = self.transition(state, symbol)
                    if next_state is not None and next_state not in reachable:
                        queue.append(next_state)

        # Find all states that can reach an accepting state
        # First, build a reverse transition map
        reverse_transitions = defaultdict(list)
        for state in range(self.num_states):
            for symbol in range(self.num_symbols):
                next_state = self.transition(state, symbol)
                if next_state is not None:
                    reverse_transitions[next_state].append((state, symbol))

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
            return FiniteStateAutomaton(0, self.num_symbols)

        # Create a new automaton with only the useful states
        # First, create a mapping from old state IDs to new state IDs
        old_to_new = {}
        for new_id, old_id in enumerate(sorted(useful_states)):
            old_to_new[old_id] = new_id

        # Create the new automaton
        result = FiniteStateAutomaton(len(useful_states), self.num_symbols)

        # Copy transitions
        for old_state in useful_states:
            for symbol in range(self.num_symbols):
                next_state = self.transition(old_state, symbol)
                if next_state is not None and next_state in useful_states:
                    result.set_transition(
                        old_to_new[old_state], symbol, old_to_new[next_state]
                    )

        # Copy initial states
        for old_state in self.initial_states:
            if old_state in useful_states:
                result.set_initial_state(old_to_new[old_state])

        # Copy accepting states
        for old_state in self.accepting_states:
            if old_state in useful_states:
                result.set_accepting_state(old_to_new[old_state])

        return result

    def to_ascii(self):
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
        lines.append(f"Initial states: {sorted(list(self.initial_states))}")
        lines.append(f"Accepting states: {sorted(list(self.accepting_states))}")
        lines.append("-" * 50)

        # Transition table
        lines.append("TRANSITION TABLE:")
        header = "State |" + "".join(f" {sym:^5} |" for sym in self.alphabet)
        lines.append(header)
        lines.append("-" * len(header))

        for state in range(self.num_states):
            # Mark initial and accepting states
            if state in self.initial_states and state in self.accepting_states:
                state_label = f"→({state})←"
            elif state in self.initial_states:
                state_label = f"→{state}  "
            elif state in self.accepting_states:
                state_label = f" ({state})←"
            else:
                state_label = f"  {state}  "

            row = f"{state_label:6} |"
            for symbol in self.alphabet:
                next_state = self.transition(state, symbol)
                cell = f" {next_state if next_state is not None else '-'}"
                row += f" {cell:^4} |"
            lines.append(row)

        lines.append("-" * len(header))

        # State diagram in ASCII art (for small automata)
        if self.num_states <= 10 and self.num_symbols <= 5:
            lines.append("\nSTATE DIAGRAM:")

            # Generate state labels
            state_labels = {}
            for state in range(self.num_states):
                if state in self.initial_states and state in self.accepting_states:
                    state_labels[state] = f"→({state})←"
                elif state in self.initial_states:
                    state_labels[state] = f"→{state}  "
                elif state in self.accepting_states:
                    state_labels[state] = f" ({state})←"
                else:
                    state_labels[state] = f"  {state}  "

            # Generate ASCII art representation
            transitions_by_state = {}
            for state in range(self.num_states):
                transitions_by_state[state] = []
                for symbol in self.alphabet:
                    next_state = self.transition(state, symbol)
                    if next_state is not None:
                        transitions_by_state[state].append((symbol, next_state))

            # Display states with their transitions
            for state in range(self.num_states):
                lines.append(f"{state_labels[state]}")
                for symbol, next_state in sorted(transitions_by_state[state]):
                    if next_state == state:  # Self-loop
                        lines.append(f"  {symbol} → ↺")
                    else:
                        lines.append(f"  {symbol} → {next_state}")
                lines.append("")

        # Compact transition list
        lines.append("TRANSITIONS:")
        grouped_transitions = defaultdict(list)
        for state in range(self.num_states):
            for symbol, next_state in self.transitions.get(state, {}).items():
                grouped_transitions[(state, next_state)].append(symbol)

        for (from_state, to_state), symbols in sorted(grouped_transitions.items()):
            if len(symbols) > 3:
                symbol_str = f"{symbols[0]},{symbols[1]},..."
            else:
                symbol_str = ",".join(str(s) for s in symbols)
            lines.append(f"  {from_state} --({symbol_str})--> {to_state}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def visualize_ascii(self):
        """Display ASCII representation of the FSA"""
        print(self.to_ascii())

    def _repr_html_(self):
        """
        When returned from a Jupyter cell, this will generate the FSA visualization
        Based on the PFSA's implementation
        """
        ret = []
        if self.num_states == 0:
            return "<code>Empty FSA</code>"

        if self.num_states > 64:
            return (
                "FSA too large to draw graphic, use fsa.visualize_ascii()<br />"
                + f"<code>FSA(states={self.num_states})</code>"
            )

        # Add nodes for initial states
        for q in self.I:
            if q in self.F:
                label = f"{self.q2str[q]}"
                color = "af8dc3"
            else:
                label = f"{self.q2str[q]}"
                color = "66c2a5"

            ret.append(
                f'g.setNode("{q}", '
                + f'{{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )
            ret.append(f'g.node("{q}").style = "fill: #{color}"; \n')

        # Add nodes for normal states
        for q in (self.Q - self.F) - self.I:
            lbl = self.q2str[q]
            ret.append(
                f'g.setNode("{q}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{q}").style = "fill: #8da0cb"; \n')

        # Add nodes for accepting states
        for q in self.F:
            # Skip if already added as initial+accepting
            if q in self.I:
                continue

            lbl = f"{self.q2str[q]}"
            ret.append(
                f'g.setNode("{q}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{q}").style = "fill: #fc8d62"; \n')

        # Add edges for transitions
        for q in self.Q:
            to = defaultdict(list)
            for symbol, next_state, _ in self.arcs(q):
                to[next_state].append(str(symbol))

            for d, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label = json.dumps(", ".join(values))
                color = "rgb(192, 192, 192)" if self.theme == "dark" else "#333"
                edge_string = (
                    f'g.setEdge("{q}","{d}",{{arrowhead:"vee",'
                    + f'label:{label},"style": "stroke: {color}; fill: none;", '
                    + f'"labelStyle": "fill: {color}; stroke: {color}; ", '
                    + f'"arrowheadStyle": "fill: {color}; stroke: {color};"}});\n'
                )
                ret.append(edge_string)

        # If the machine is too big, don't attempt to display it
        if len(ret) > 256:
            return (
                "FSA too large to draw graphic, use fsa.visualize_ascii()<br />"
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
