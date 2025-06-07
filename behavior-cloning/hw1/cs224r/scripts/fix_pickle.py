# save this as fix_pickle.py
import pickle
import numpy as np

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core':
            module = 'numpy'
        return super().find_class(module, name)

# Load with custom unpickler
with open('cs224r/data/needle_reach_bc_data_10000.pkl', 'rb') as f:
    try:
        data = CustomUnpickler(f).load()
        # Save with current numpy
        with open('cs224r/data/needle_reach_bc_data_10000.pkl', 'wb') as out_f:
            pickle.dump(data, out_f)
        print("Successfully converted the pickle file!")
    except Exception as e:
        print(f"Error converting pickle: {e}")