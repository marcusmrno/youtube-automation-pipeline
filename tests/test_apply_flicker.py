"""apply_flicker against a fake Palmier MCP server: nothing reaches 127.0.0.1:19789."""
import json

import pytest

import apply_flicker as af


@pytest.fixture
def palmier(monkeypatch, tmp_path):
    """A fake timeline + media library; every tool call is recorded."""
    state = {"calls": [], "tracks": [], "media": []}

    def call_tool(name, arguments):
        state["calls"].append(name)
        if name == "get_timeline":
            return {"fps": 30, "tracks": state["tracks"]}
        if name == "get_media":
            return {"entries": state["media"]}
        return {}

    monkeypatch.setattr(af, "call_tool", call_tool)
    monkeypatch.setattr(af, "initialize", lambda: None)
    monkeypatch.setattr(af, "SNAPSHOT_PATH", tmp_path / ".flicker_snapshot.json")
    return state


def _snapshot(label="V2"):
    af.SNAPSHOT_PATH.write_text(json.dumps({"flicker_track_label": label}))


def test_undo_leaves_a_track_that_no_longer_holds_flicker_clips(palmier):
    # the snapshot only records the label "V2"; in another project V2 is the user's b-roll
    palmier["media"] = [{"id": "m1", "name": "001"}, {"id": "br", "name": "broll-take-3"}]
    palmier["tracks"] = [{"label": "V1", "type": "video", "clips": [{"mediaRef": "m1"}]},
                         {"label": "V2", "type": "video", "clips": [{"mediaRef": "br"}]}]
    _snapshot()
    with pytest.raises(SystemExit):
        af.undo()
    assert "remove_tracks" not in palmier["calls"]


def test_undo_removes_the_flicker_layer(palmier):
    palmier["media"] = [{"id": "b", "name": "001b"}, {"id": "c", "name": "001c"}]
    palmier["tracks"] = [{"label": "V1", "type": "video", "clips": []},
                         {"label": "V2", "type": "video", "clips": [{"mediaRef": "b"}, {"mediaRef": "c"}]}]
    _snapshot()
    af.undo()
    assert "remove_tracks" in palmier["calls"] and not af.SNAPSHOT_PATH.exists()
