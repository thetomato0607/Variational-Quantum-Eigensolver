"""Result persistence helpers."""

import json
import os
import numpy as np
from datetime import datetime

def save_results(data_dict, folder="results", filename="experiment"):
    """Write ``data_dict`` to ``<folder>/<filename>_<timestamp>.json``.

    Only top-level NumPy values are converted to lists; NumPy arrays nested
    deeper will make ``json.dump`` fail.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    clean_data = {}
    for k, v in data_dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            clean_data[k] = v.tolist()
        else:
            clean_data[k] = v
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{filename}_{timestamp}.json")
    
    with open(path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    print(f"Data saved to {path}")