import importlib


def get_class(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError):
        print(f"Error: Could not find {class_name} in {module_path}")
        return None
