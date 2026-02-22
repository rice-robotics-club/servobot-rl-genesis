import sys

import jsonschema_gentypes.cli


def main():
    sys.argv = [
        "jsonschema-gentypes",
        "--json-schema=config/schemas/config.json",
        "--python=src/config/config.py",
    ]
    jsonschema_gentypes.cli.main()


if __name__ == "__main__":
    main()
