"""Real MEVA query audit regressions: action constraints must not fall back to people."""

import pytest
from app.retrieval import event_matches, parse_query


@pytest.mark.parametrize(
    "query,event",
    [
        ("vehicle started", "started_moving"),
        ("person approached object", "approached_object"),
        ("person moved away", "moved_away"),
        ("person exited zone", "exited_zone"),
    ],
)
def test_requested_action_is_a_hard_constraint(query, event):
    assert parse_query(query)["event"] == event


@pytest.mark.parametrize(
    "query",
    [
        "person talked to another person",
        "person used phone",
        "vehicle turned left",
        "vehicle turned right",
        "vehicle reversed",
    ],
)
def test_unsupported_action_does_not_retrieve_unrelated_observations(client, query):
    result = client.post("/search", json={"query": query}).json()
    assert result["parsed"].get("unsupported_activity")
    assert result["results"] == []


def test_zone_entry_requires_a_zone_event():
    assert not event_matches("appeared", "entered_zone")
    assert event_matches("entered_zone", "entered_zone")
