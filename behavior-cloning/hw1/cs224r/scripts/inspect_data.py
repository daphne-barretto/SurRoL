"""
Inspect the actual structure of the npz data
"""

import numpy as np

def inspect_npz(data_path):
    """Thoroughly inspect the npz file structure"""
    print(f"Inspecting: {data_path}")
    print("="*80)
    
    data = np.load(data_path, allow_pickle=True)
    
    print(f"Top-level keys: {list(data.keys())}")
    print()
    
    for key in data.keys():
        value = data[key]
        print(f"Key: '{key}'")
        print(f"  Type: {type(value)}")
        print(f"  Shape: {getattr(value, 'shape', 'N/A')}")
        print(f"  Dtype: {getattr(value, 'dtype', 'N/A')}")
        
        # If it's an array, try to inspect its contents
        if hasattr(value, 'shape'):
            if value.shape == ():  # scalar array
                print(f"  Value (scalar): {value.item()}")
            elif len(value.shape) == 1 and value.shape[0] <= 10:
                print(f"  Contents: {value}")
            else:
                print(f"  First few elements: {value.flat[:5] if hasattr(value, 'flat') else 'N/A'}")
        
        # If it's an object array, try to inspect the first element
        if hasattr(value, 'dtype') and value.dtype == object:
            print(f"  Object array - inspecting first element:")
            try:
                first_elem = value[0] if len(value) > 0 else None
                print(f"    First element type: {type(first_elem)}")
                print(f"    First element shape: {getattr(first_elem, 'shape', 'N/A')}")
                
                if hasattr(first_elem, 'keys'):
                    print(f"    First element keys: {list(first_elem.keys())}")
                    
                    # Inspect each key in the first element
                    for subkey in first_elem.keys():
                        subvalue = first_elem[subkey]
                        print(f"      '{subkey}': shape {getattr(subvalue, 'shape', 'N/A')}, dtype {getattr(subvalue, 'dtype', 'N/A')}")
                        
                elif hasattr(first_elem, 'shape'):
                    print(f"    First element contents preview: {first_elem.flat[:5] if hasattr(first_elem, 'flat') else first_elem}")
                    
            except Exception as e:
                print(f"    Error inspecting first element: {e}")
        
        print()

def main():
    data_path = "/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz"
    inspect_npz(data_path)
    
    # Also try to extract data the correct way once we understand the structure
    print("="*80)
    print("ATTEMPTING DATA EXTRACTION")
    print("="*80)
    
    data = np.load(data_path, allow_pickle=True)
    
    # Try different ways to access the data
    print("Trying data['obs']...")
    try:
        obs_data = data['obs']
        print(f"data['obs'] type: {type(obs_data)}")
        print(f"data['obs'] shape: {getattr(obs_data, 'shape', 'N/A')}")
        
        if hasattr(obs_data, 'shape') and len(obs_data.shape) > 0:
            print(f"First obs element: {obs_data[0]}")
            if hasattr(obs_data[0], 'keys'):
                print(f"Keys in first obs: {list(obs_data[0].keys())}")
        
    except Exception as e:
        print(f"Error accessing data['obs']: {e}")
    
    print("\nTrying data['acs']...")
    try:
        acs_data = data['acs']
        print(f"data['acs'] type: {type(acs_data)}")
        print(f"data['acs'] shape: {getattr(acs_data, 'shape', 'N/A')}")
        
        if hasattr(acs_data, 'shape') and len(acs_data.shape) > 0:
            print(f"First action element: {acs_data[0]}")
            if hasattr(acs_data[0], 'shape'):
                print(f"First action shape: {acs_data[0].shape}")
        
    except Exception as e:
        print(f"Error accessing data['acs']: {e}")

if __name__ == "__main__":
    main()