"""Unit tests for the Entity Resolution and Mule Ring Graph Engine."""

import pytest
from graph.entity_graph import FraudEntityGraph

def test_isolated_clean_customer():
    graph = FraudEntityGraph()
    res = graph.analyze_customer_syndicate("CUST_001")
    assert res["customer_id"] == "CUST_001"
    assert res["mule_ring_detected"] is False
    assert res["cluster_risk_level"] == "LOW"
    assert len(res["shared_devices"]) == 0

def test_mule_ring_syndicate_detected():
    graph = FraudEntityGraph()
    res = graph.analyze_customer_syndicate("CUST_0042")
    assert res["customer_id"] == "CUST_0042"
    assert res["mule_ring_detected"] is True
    assert res["cluster_risk_level"] in ["HIGH", "CRITICAL"]
    assert "DEV_FARM_01" in res["shared_devices"]
    assert "198.51.100.77" in res["shared_ips"]
    assert len(res["connected_customers"]) >= 3
    assert "CUST_0180" in res["connected_customers"]

def test_dynamic_entity_addition():
    graph = FraudEntityGraph()
    # Add two new customers sharing a burner device
    graph.add_transaction_entities("CUST_ALPHA", device_id="BURNER_DEV_77")
    graph.add_transaction_entities("CUST_BETA", device_id="BURNER_DEV_77")

    res_alpha = graph.analyze_customer_syndicate("CUST_ALPHA")
    assert res_alpha["mule_ring_detected"] is True
    assert "BURNER_DEV_77" in res_alpha["shared_devices"]
    assert "CUST_BETA" in res_alpha["connected_customers"]

def test_subgraph_data_export():
    graph = FraudEntityGraph()
    subgraph_data = graph.get_cluster_subgraph_data("CUST_0042")
    assert "nodes" in subgraph_data
    assert "edges" in subgraph_data
    assert len(subgraph_data["nodes"]) > 1
    assert len(subgraph_data["edges"]) > 1

def test_prune_old_entities():
    graph = FraudEntityGraph(max_nodes=100)
    # Add an entity with an old timestamp (10 days ago)
    old_ts = 1000.0
    graph.add_transaction_entities("CUST_OLD", device_id="DEV_OLD_99", timestamp=old_ts)
    assert graph.graph.has_node("cust:CUST_OLD")

    # Prune entities older than 1 day (86400 seconds)
    pruned_count = graph.prune_old_entities(max_age_seconds=86400.0)
    assert pruned_count >= 1
    assert not graph.graph.has_node("cust:CUST_OLD")

