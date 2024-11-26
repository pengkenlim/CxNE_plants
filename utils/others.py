import importlib
import pickle
import joblib

def parse_parameters(file_path):
    """Imports a Python module from a specified file path.

    Args:
        file_path: The path to the Python file.

    Returns:
        The imported module.
    """
    module_name = file_path.split('/')[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    exec(open(file_path).read(), module.__dict__)
    return module

def pickle_load(path):
    with open(path, "rb") as fbin:
        obj = pickle.load(fbin)
    return obj

def pickle_dump(obj,path):
    with open(path, "wb") as fbin:
        pickle.dump(obj, fbin)




