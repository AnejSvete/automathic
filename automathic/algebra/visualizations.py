import ipywidgets as widgets  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from IPython.display import display


def visualize_cayley_table(algebra, cmap="viridis", annot=True, show_names=True):
    """
    Create a heatmap visualization of a Cayley table with annotations.

    Args:
        algebra: A SingleElementSetAlgebra instance (Magma, Semigroup, Group, etc.)
        cmap: Matplotlib colormap name
        annot: Whether to show annotations in cells
        show_names: Whether to use element names (True) or indices (False)

    Returns:
        The matplotlib figure
    """
    elements = algebra.elements
    n = len(elements)

    # Get the table data
    if show_names:
        table_data = algebra.table.to_list_with_names(elements)
    else:
        table_data = algebra.table.tolist()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(n + 2, n + 1))

    # Create a heatmap
    sns.heatmap(
        (
            np.array(table_data)
            if not show_names
            else np.array(
                [[elements.index(cell) for cell in row] for row in table_data]
            )
        ),
        annot=np.array(table_data) if annot else False,
        fmt="" if show_names else "d",
        cmap=cmap,
        linewidths=0.5,
        cbar=False,
        ax=ax,
    )

    # Set the title based on the algebra type
    ax.set_title(
        f"{algebra.__class__.__name__} Cayley Table: {algebra.name}", fontsize=14
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


def interactive_cayley_table(algebra):
    """
    Create an interactive Cayley table visualization with controls.

    Args:
        algebra: A SingleElementSetAlgebra instance
    """
    # Define the widgets for interaction
    cmap_dropdown = widgets.Dropdown(
        options=["viridis", "plasma", "magma", "cividis", "coolwarm", "RdBu", "YlOrRd"],
        value="viridis",
        description="Colormap:",
    )

    show_annot = widgets.Checkbox(value=True, description="Show values")

    show_names = widgets.Checkbox(value=True, description="Use element names")

    # Function to update the visualization
    def update_visualization(cmap, show_annot, show_names):
        plt.close("all")  # Close previous plots
        visualize_cayley_table(
            algebra, cmap=cmap, annot=show_annot, show_names=show_names
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


def visualize_element_graph(algebra, element=None, depth=2, annotate=True):
    """
    Visualize the relationships between elements in an algebra.

    Args:
        algebra: A SingleElementSetAlgebra instance
        element: The starting element (if None, uses identity or first element)
        depth: How many operations to follow
        annotate: Whether to show annotations on edges

    Returns:
        The matplotlib figure
    """
    elements = algebra.elements

    if element is None:
        # Use identity if exists, otherwise first element
        element = algebra.identity if algebra.has_identity() else elements[0]

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
                result = algebra.op(e1, e2)
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
    identity_node = algebra.identity if algebra.has_identity() else None

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
        f"Element Relationships in {algebra.__class__.__name__}: {algebra.name}",
        fontsize=14,
    )

    # Remove axis
    ax.set_axis_off()

    return fig


def visualize_substructures(algebra):
    """
    Visualize the substructures (subalgebras) of an algebra.

    Args:
        algebra: A SingleElementSetAlgebra instance

    Returns:
        The matplotlib figure
    """
    # Get substructures
    proper_subalgebras = algebra.proper_subalgebras()

    if not proper_subalgebras:
        print(f"{algebra.name} has no proper subalgebras.")
        return None

    # Add the full algebra itself
    all_algebras = proper_subalgebras + [algebra]

    # Create a graph where subalgebras are nodes
    G = nx.DiGraph()

    # Add nodes for each algebra
    for i, sub in enumerate(all_algebras):
        G.add_node(i, algebra=sub, elements=sub.elements, order=sub.order)

    # Add edges for containment relationships
    for i, sub1 in enumerate(all_algebras):
        for j, sub2 in enumerate(all_algebras):
            if i != j and set(sub1.elements).issubset(set(sub2.elements)):
                # There's a containment relation
                G.add_edge(i, j)

    # Remove transitive edges for cleaner visualization
    G_transitive_reduction = nx.transitive_reduction(G)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a hierarchical layout
    pos = nx.kamada_kawai_layout(G_transitive_reduction)

    # Node sizes based on algebra order
    node_sizes = [300 * G.nodes[n]["order"] for n in G_transitive_reduction.nodes()]

    # Draw nodes with color gradient based on size
    nodes = nx.draw_networkx_nodes(
        G_transitive_reduction,
        pos,
        node_size=node_sizes,
        node_color=node_sizes,
        cmap=plt.cm.YlOrRd,
        alpha=0.8,
        ax=ax,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G_transitive_reduction,
        pos,
        edge_color="gray",
        width=1.5,
        arrowstyle="->",
        arrowsize=20,
        ax=ax,
    )

    # Add labels with algebra information
    labels = {
        i: f"{all_algebras[i].name}\nOrder: {all_algebras[i].order}"
        for i in G_transitive_reduction.nodes()
    }
    nx.draw_networkx_labels(
        G_transitive_reduction,
        pos,
        labels=labels,
        font_size=10,
        font_family="sans-serif",
        ax=ax,
    )

    ax.set_title(f"Substructure Hierarchy of {algebra.name}", fontsize=14)
    ax.set_axis_off()

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
    sm.set_array(node_sizes)
    plt.colorbar(sm, ax=ax, label="Algebra Order")

    return fig


def visualize_element_properties(algebra):
    """
    Create a visual representation of element properties in the algebra.

    Args:
        algebra: A SingleElementSetAlgebra instance

    Returns:
        The matplotlib figure
    """
    elements = algebra.elements
    n = len(elements)

    # Prepare data for visualization
    data = []

    # Properties to visualize
    if isinstance(algebra, algebra.Group):
        headers = ["Element", "Order", "Inverse", "Generates"]

        for elem in elements:
            order = algebra.element_order(elem)
            inverse = algebra.inv(elem)
            generates = elem in algebra.is_cyclic() if algebra.is_cyclic() else False
            data.append([elem, order, inverse, generates])
    else:
        headers = ["Element", "Identity?", "Commutative?", "In Center?"]

        for elem in elements:
            is_identity = (elem == algebra.identity) if algebra.identity else False

            # Check if element commutes with all other elements
            commutative = all(
                algebra.op(elem, x) == algebra.op(x, elem) for x in elements
            )

            # Check if element is in center
            in_center = elem in algebra.center()

            data.append([elem, is_identity, commutative, in_center])

    # Create the figure
    fig, ax = plt.subplots(figsize=(len(headers) * 2, n * 0.5 + 1))
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["lightblue"] * len(headers),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Color special cells
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Headers
            continue

        # Color based on properties
        if j == 0:  # Element name
            cell.set_facecolor("lightyellow")
        elif j >= 1:
            if isinstance(data[i - 1][j], bool) and data[i - 1][j]:
                cell.set_facecolor("lightgreen")
            elif str(data[i - 1][j]).lower() in ["yes", "true"]:
                cell.set_facecolor("lightgreen")

    plt.title(
        f"{algebra.__class__.__name__} Element Properties: {algebra.name}", fontsize=14
    )
    plt.tight_layout()

    return fig


def algebra_dashboard(algebra):
    """
    Create an interactive dashboard for exploring a finite algebra.

    Args:
        algebra: A SingleElementSetAlgebra instance
    """
    # Create tabs for different visualizations
    tab_titles = [
        "Cayley Table",
        "Element Properties",
        "Element Graph",
        "Substructures",
    ]

    # Create the tabs
    tabs = widgets.Tab()
    children = []

    # Cayley Table Tab
    table_tab = widgets.Output()
    with table_tab:
        interactive_cayley_table(algebra)
    children.append(table_tab)

    # Element Properties Tab
    props_tab = widgets.Output()
    with props_tab:
        visualize_element_properties(algebra)
        plt.tight_layout()
        plt.show()
    children.append(props_tab)

    # Element Graph Tab
    graph_tab = widgets.Output()
    with graph_tab:
        # Create element selection dropdown
        element_dropdown = widgets.Dropdown(
            options=algebra.elements,
            value=algebra.identity if algebra.has_identity() else algebra.elements[0],
            description="Start Element:",
        )

        depth_slider = widgets.IntSlider(
            value=2, min=1, max=5, step=1, description="Depth:"
        )

        show_labels = widgets.Checkbox(value=True, description="Show Edge Labels")

        def update_graph(element, depth, annotate):
            plt.close("all")
            visualize_element_graph(algebra, element, depth, annotate)
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
    children.append(graph_tab)

    # Substructures Tab
    sub_tab = widgets.Output()
    with sub_tab:
        visualize_substructures(algebra)
        plt.tight_layout()
        plt.show()
    children.append(sub_tab)

    # Set tab children and titles
    tabs.children = children
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)

    # Display the dashboard
    display(
        widgets.HTML(f"<h2>{algebra.__class__.__name__} Dashboard: {algebra.name}</h2>")
    )
    display(tabs)
