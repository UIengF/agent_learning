from __future__ import annotations


def test_import_agent_no_crash():
    """Verify agent module can be imported without web dependencies."""
    from graph_rag_app import agent

    assert agent.LANGGRAPH_AVAILABLE is False or agent.LANGGRAPH_AVAILABLE is True
