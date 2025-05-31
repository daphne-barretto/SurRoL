# script to view the contents of a .npz file

import numpy as np
import argparse

def view_npz(filepath):
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"Contents of '{filepath}':")
        for key in data.files:
            print(f"Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}")

        # show all the obs
        if 'obs' in data:
            obs = data['obs']
            if isinstance(obs, np.ndarray):
                print(f"Obs shape: {obs.shape}, Dtype: {obs.dtype}")
                print(obs[8])
                # if block_encoding in obs, print its shape and dtype
                if 'block_encoding' in obs:
                    block_encoding = obs['block_encoding']
                    print(f"Block encoding shape: {block_encoding.shape}, Dtype: {block_encoding.dtype}")
                else:
                    print("Block encoding not found in obs.")
            elif isinstance(obs, list):
                print(f"Obs is a list with {len(obs)} items.")
                for i, item in enumerate(obs[:5]):
                    print(f"Item {i}: Shape: {item.shape if hasattr(item, 'shape') else 'N/A'}, Dtype: {type(item)}")
            else:
                print("Obs is neither a numpy array nor a list.")

    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View the contents of a .npz file")
    parser.add_argument("--filepath", type=str, help="Path to the .npz file")
    args = parser.parse_args()
    view_npz(args.filepath)