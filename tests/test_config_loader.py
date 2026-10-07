import pytest
import yaml

from utils.config_loader import ConfigLoader


def test_load_returns_dict(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"key": "value", "nested": {"a": 1}}))

    config = ConfigLoader(str(config_file)).load()

    assert config == {"key": "value", "nested": {"a": 1}}


def test_load_empty_file_returns_none(tmp_path):
    config_file = tmp_path / "empty.yml"
    config_file.write_text("")

    config = ConfigLoader(str(config_file)).load()

    assert config is None


def test_load_raises_on_missing_file():
    loader = ConfigLoader("/nonexistent/path/config.yml")

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_load_preserves_types(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({
        "threshold": 0.5,
        "count": 10,
        "enabled": True,
        "name": "test",
    }))

    config = ConfigLoader(str(config_file)).load()

    assert isinstance(config["threshold"], float)
    assert isinstance(config["count"], int)
    assert isinstance(config["enabled"], bool)
    assert isinstance(config["name"], str)


def test_load_raises_on_malformed_yaml(tmp_path):
    config_file = tmp_path / "bad.yml"
    config_file.write_text("key: value\n  bad_indent: [unclosed")

    loader = ConfigLoader(str(config_file))

    with pytest.raises(yaml.YAMLError):
        loader.load()
