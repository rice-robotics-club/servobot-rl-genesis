import json
from os import PathLike
from pathlib import Path

import yaml
from jsonschema import validate
from referencing import Registry, Resource
from referencing.jsonschema import Schema

from .config import Config


def load_config(
    config_path: str | PathLike[str],
    schema_path: str | PathLike[str] = "config/schemas/config.json",
) -> Config:
    """Loads YAML config file and validates it against schema."""
    dir = Path(schema_path).parent
    registry = Registry()
    schema: Schema | None = None

    for schema_file in dir.glob("*.json"):
        with open(schema_file, "r") as f:
            content = json.load(f)
            if schema_file.samefile(schema_path):
                schema = content
            # Use the $id defined in the file, or the filename
            schema_id = content.get("$id", schema_file.name)
            resource = Resource.from_contents(content)
            registry = registry.with_resource(schema_id, resource)

    if not schema:
        raise ValueError(f"Schema not found at {schema_path}")

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        validate(instance=config, schema=schema, registry=registry)
        return config
