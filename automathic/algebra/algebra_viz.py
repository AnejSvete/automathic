"""
Mixin classes for integrating visualizations into algebra classes.
"""

import ipywidgets as widgets  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from IPython.display import display


class VisualizableMixin:
    """
    A mixin class that adds visualization capabilities to algebra classes.
    """

    def visualize(self, **kwargs):
        """
        Visualize the algebra with a dashboard of multiple views.
        This is the main entry point for visualizations.

        Args:
            **kwargs: Additional visualization parameters
        """
        # Check if we're in a Jupyter environment
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                self.show_dashboard(**kwargs)
            else:
                # Fallback to simple visualization when not in Jupyter
                self.visualize_cayley_table(**kwargs)
        except ImportError:
            # Fallback to simple visualization when IPython is not available
            self.visualize_cayley_table(**kwargs)

    def visualize_cayley_table(
        self, cmap="viridis", annot=True, show_names=True, title=None
    ):
        """
        Create a heatmap visualization of the Cayley table.

        Args:
            cmap: Matplotlib colormap name
            annot: Whether to show annotations in cells
            show_names: Whether to use element names (True) or indices (False)
            title: Custom title for the visualization

        Returns:
            The matplotlib figure
        """
        elements = self.elements
        n = len(elements)

        # Get the table data
        if show_names:
            table_data = self.table.to_list_with_names(elements)
        else:
            table_data = self.table.tolist()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(n + 2, n + 1))

        # Create a heatmap
        sns.heatmap(
            (
                np.array([[elements.index(cell) for cell in row] for row in table_data])
                if show_names
                else np.array(table_data)
            ),
            annot=np.array(table_data) if annot else False,
            fmt="" if show_names else "d",
            cmap=cmap,
            linewidths=0.5,
            cbar=False,
            ax=ax,
        )

        # Set the title based on the algebra type
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(
                f"{self.__class__.__name__} Cayley Table: {self.name}", fontsize=14
            )

        # Add element labels
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(elements)
        ax.set_yticklabels(elements)

        plt.xlabel("Column Element")
        plt.ylabel("Row Element")

        # Rotate the tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add grid lines
        plt.grid(False)

        return fig

    def visualize_element_graph(self, element=None, depth=2, annotate=True):
        """
        Visualize the relationships between elements in the algebra.

        Args:
            element: The starting element (if None, uses identity or first element)
            depth: How many operations to follow
            annotate: Whether to show annotations on edges

        Returns:
            The matplotlib figure
        """
        elements = self.elements

        if element is None:
            # Use identity if exists, otherwise first element
            element = self.identity if self.has_identity() else elements[0]

        # Create a directed graph
        G = nx.DiGraph()

        # Add all elements as nodes
        for elem in elements:
            G.add_node(elem)

        # Add edges based on the operation
        processed = set()
        to_process = {element}
        current_depth = 0

        while to_process and current_depth < depth:
            next_to_process = set()

            for e1 in to_process:
                for e2 in elements:
                    result = self.op(e1, e2)
                    G.add_edge(e1, e2, result=result, operation="op")

                    if result not in processed:
                        next_to_process.add(result)

                processed.add(e1)

            to_process = next_to_process
            current_depth += 1

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a nice layout algorithm
        pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(G.nodes())))

        # Draw nodes
        identity_node = self.identity if self.has_identity() else None

        # Draw regular nodes
        regular_nodes = [n for n in G.nodes() if n != identity_node and n != element]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=regular_nodes,
            node_color="skyblue",
            node_size=700,
            alpha=0.8,
            ax=ax,
        )

        # Draw identity node if exists
        if identity_node:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[identity_node],
                node_color="gold",
                node_size=900,
                alpha=0.9,
                ax=ax,
            )

        # Draw start element
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[element],
            node_color="lightgreen",
            node_size=900,
            alpha=0.9,
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", width=1.0, arrowstyle="->", arrowsize=15, ax=ax
        )

        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", ax=ax)

        # Add edge labels if requested
        if annotate:
            edge_labels = {
                (u, v): f"{u}*{v}={d['result']}" for u, v, d in G.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=8, ax=ax
            )

        # Set title
        ax.set_title(
            f"Element Relationships in {self.__class__.__name__}: {self.name}",
            fontsize=14,
        )

        # Remove axis
        ax.set_axis_off()

        return fig

    def show_dashboard(self, **kwargs):
        """
        Create an interactive dashboard for exploring the algebra.
        This requires being in a Jupyter environment.
        """
        # Create tabs for different visualizations
        tab_titles = ["Cayley Table", "Element Graph"]

        # Create the tabs
        tabs = widgets.Tab()
        children = []

        # Cayley Table Tab
        table_tab = widgets.Output()
        with table_tab:
            self._interactive_cayley_table()
        children.append(table_tab)

        # Element Graph Tab
        graph_tab = widgets.Output()
        with graph_tab:
            self._interactive_element_graph()
        children.append(graph_tab)

        # Set tab children and titles
        tabs.children = children
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)

        # Display the dashboard
        display(
            widgets.HTML(f"<h2>{self.__class__.__name__} Dashboard: {self.name}</h2>")
        )
        display(tabs)

    def _interactive_cayley_table(self):
        """Create an interactive Cayley table visualization with controls."""
        # Define the widgets for interaction
        cmap_dropdown = widgets.Dropdown(
            options=[
                "viridis",
                "plasma",
                "magma",
                "cividis",
                "coolwarm",
                "RdBu",
                "YlOrRd",
            ],
            value="viridis",
            description="Colormap:",
        )

        show_annot = widgets.Checkbox(value=True, description="Show values")

        show_names = widgets.Checkbox(value=True, description="Use element names")

        # Function to update the visualization
        def update_visualization(cmap, show_annot, show_names):
            plt.close("all")  # Close previous plots
            self.visualize_cayley_table(
                cmap=cmap, annot=show_annot, show_names=show_names
            )
            plt.show()

        # Create interactive output
        out = widgets.interactive_output(
            update_visualization,
            {"cmap": cmap_dropdown, "show_annot": show_annot, "show_names": show_names},
        )

        # Display widgets and visualization
        display(widgets.HBox([cmap_dropdown, show_annot, show_names]))
        display(out)

    def _interactive_element_graph(self):
        """Create an interactive element graph visualization with controls."""
        # Create element selection dropdown
        element_dropdown = widgets.Dropdown(
            options=self.elements,
            value=self.identity if self.has_identity() else self.elements[0],
            description="Start Element:",
        )

        depth_slider = widgets.IntSlider(
            value=2, min=1, max=5, step=1, description="Depth:"
        )

        show_labels = widgets.Checkbox(value=True, description="Show Edge Labels")

        def update_graph(element, depth, annotate):
            plt.close("all")
            self.visualize_element_graph(element, depth, annotate)
            plt.tight_layout()
            plt.show()

        graph_out = widgets.interactive_output(
            update_graph,
            {
                "element": element_dropdown,
                "depth": depth_slider,
                "annotate": show_labels,
            },
        )

        display(widgets.HBox([element_dropdown, depth_slider, show_labels]))
        display(graph_out)


class RingVisualizableMixin(VisualizableMixin):
    """
    A mixin class for Ring-specific visualizations
    """

    def visualize_zero_divisors(self):
        """Visualize zero divisors in the ring"""
        zero_divisors = self.zero_divisors()
        if not zero_divisors:
            print(f"No zero divisors found in {self.name}")
            return None

        # Create a directed graph showing zero divisor relationships
        G = nx.DiGraph()

        # Add all elements as nodes
        for elem in self.elements:
            G.add_node(elem)

        # Find all pairs where product is zero
        zero_product_pairs = self.element_pairs_where_product_equals(self.zero)
        zero_div_pairs = [pair for pair in zero_product_pairs if self.zero not in pair]

        # Add edges for zero divisor relationships
        for a, b in zero_div_pairs:
            G.add_edge(a, b, result=self.zero)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a nice layout algorithm
        pos = nx.spring_layout(G, k=1.5 / np.sqrt(len(G.nodes())))

        # Draw regular nodes (non-zero divisors)
        non_zero_divs = [
            elem
            for elem in self.elements
            if elem not in zero_divisors and elem != self.zero
        ]
        if non_zero_divs:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=non_zero_divs,
                node_color="lightgray",
                node_size=500,
                alpha=0.6,
                ax=ax,
            )

        # Draw zero divisors
        if zero_divisors:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=zero_divisors,
                node_color="red",
                node_size=700,
                alpha=0.8,
                ax=ax,
            )

        # Draw identity node (zero)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[self.zero],
            node_color="blue",
            node_size=800,
            alpha=0.9,
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", width=1.0, arrowstyle="->", arrowsize=15, ax=ax
        )

        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", ax=ax)

        # Set title
        ax.set_title(f"Zero Divisors in {self.name}", fontsize=14)

        # Remove axis
        ax.set_axis_off()

        return fig
