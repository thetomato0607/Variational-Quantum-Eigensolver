import json
import os
import numpy as np
from datetime import datetime

def save_results(data_dict, folder="results", filename="experiment"):
    """Saves dictionary to JSON with timestamp."""
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