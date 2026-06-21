import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from create_profile import next_version_name


def test_base_name_gets_v2():
    assert next_version_name("my-channel") == "my-channel-v2"


def test_v2_gets_v3():
    assert next_version_name("my-channel-v2") == "my-channel-v3"


def test_v9_gets_v10():
    assert next_version_name("my-channel-v9") == "my-channel-v10"


def test_name_ending_in_number_not_version():
    # "channel-2" is not a version slug — treat as base
    assert next_version_name("channel-2") == "channel-2-v2"
