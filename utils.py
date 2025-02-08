import numpy as np

def replace_nan_with_none(obj):
    """ Recursively replace NaN values with None in dict or list """
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return 0  # Replace NaN with 0 (or any default value)
    return obj
