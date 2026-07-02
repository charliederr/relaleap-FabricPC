from __future__ import annotations

import jax


def test_package_imports() -> None:
    import relaleap_fabricpc

    assert relaleap_fabricpc.__version__


def test_fabricpc_resolves_to_upstream() -> None:
    import fabricpc

    assert "FabricPC" in fabricpc.__file__
    assert "cFabricPC" not in fabricpc.__file__


def test_residual_columns_node_in_fabricpc_graph() -> None:
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.core.topology import Edge
    from fabricpc.graph_assembly import TaskMap, graph
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.nodes import IdentityNode

    from relaleap_fabricpc.fabricpc_nodes import ResidualColumnsNode

    hidden = IdentityNode(shape=(8, 4), name="hidden")
    residual = ResidualColumnsNode(
        shape=(8, 4),
        name="residual",
        num_columns=4,
        atoms_per_column=2,
        top_k=1,
    )
    structure = graph(
        nodes=[hidden, residual],
        edges=[Edge(source=hidden, target=residual.slot("in"))],
        task_map=TaskMap(x=hidden),
        inference=InferenceSGD(),
    )
    params = initialize_params(structure, jax.random.PRNGKey(0))

    assert "residual" in structure.nodes
    assert "residual" in params.nodes
    assert params.nodes["residual"].weights["atom_values"].shape == (4, 2, 4)
