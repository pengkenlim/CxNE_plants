import importlib
import pickle

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


def preprocessing_worker(preprocess_queue, calculation_queue, Preprocess):
    while True:
        batch = preprocess_queue.get()
        if batch is None:  # Sentinel to stop the worker
            break
        output_1 = Preprocess(*batch)
        calculation_queue.put(output_1)
        preprocess_queue.task_done()

def calculation_worker(calculation_queue, postprocessing_queue, Calculation):
    while True:
        output_1 = calculation_queue.get()
        if output_1 is None:  # Sentinel to stop the worker
            break
        output_2 = Calculation(*output_1)
        postprocessing_queue.put(output_2)
        calculation_queue.task_done()

def postprocessing_worker(postprocessing_queue, Postprocess):
    while True:
        output_2 = postprocessing_queue.get()
        if output_2 is None:  # Sentinel to stop the worker
            break
        Postprocess(*output_2)
        postprocessing_queue.task_done()
