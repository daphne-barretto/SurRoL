import numpy as np
import os
import glob
import argparse
from datetime import datetime
import pprint

def analyze_npz_file(file_path):
    """Analyze an npz file and determine its structure and content type."""
    print(f"\n{'='*80}")
    print(f"Analyzing file: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    # Load the npz file
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Get basic info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Extract creation time
        creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
        print(f"Creation time: {creation_time}")
        
        # List available keys
        print(f"\nKeys in the file:")
        keys = sorted(data.keys())
        for key in keys:
            print(f"  - {key}")
        
        # Key-based classification
        file_type = "unknown"
        
        # Method 1: Classification based on key structure
        key_set = set(keys)
        if key_set == {'acs', 'info', 'obs'}:
            file_type = "state_based"
            print("\nClassification by keys: STATE-BASED (keys: acs, info, obs only)")
        elif 'rewards' in key_set and ('traditional_obs' in key_set or 'trad_obs' in key_set):
            file_type = "image_based"
            print("\nClassification by keys: IMAGE-BASED (contains rewards and traditional_obs/trad_obs)")
        elif 'images' in key_set or any('image' in k.lower() for k in key_set):
            file_type = "image_based"
            print("\nClassification by keys: IMAGE-BASED (contains image keywords)")
        elif file_path.find('data_images_') >= 0 or file_path.find('image_demos') >= 0:
            # Filename-based hint
            file_type = "image_based"
            print("\nClassification by filename: IMAGE-BASED (filename contains 'data_images_' or 'image_demos')")
        
        # Method 2: Content-based detection (as backup)
        if file_type == "unknown":
            print("\nCould not classify by keys alone. Analyzing content...")
            
            # Check for image data characteristics
            has_image_data = False
            has_state_characteristics = False
            
            for key in keys:
                arr = data[key]
                shape_str = str(arr.shape)
                dtype_str = str(arr.dtype)
                
                print(f"  Key: {key}")
                print(f"    Shape: {shape_str}")
                print(f"    Dtype: {dtype_str}")
                
                # Look for image data characteristics
                if arr.size > 0:
                    if arr.dtype != np.dtype('O') and len(arr.shape) >= 3:
                        # Check for typical image dimensions
                        if arr.shape[-1] == 3 and arr.shape[-2] >= 100 and arr.shape[-3] >= 100:
                            has_image_data = True
                            print(f"    ** Contains image data (detected by dimensions)")
                    
                    # For object arrays, check a sample
                    elif arr.dtype == np.dtype('O') and arr.size > 0:
                        sample = arr.flat[0]
                        if isinstance(sample, dict):
                            # Check if dictionary contains state info
                            contains_state_info = any(
                                state_key in k.lower() 
                                for k in sample.keys() 
                                for state_key in ['state', 'position', 'joint']
                            )
                            if contains_state_info:
                                has_state_characteristics = True
                        elif hasattr(sample, 'shape'):
                            # Check if it looks like an image array
                            if len(sample.shape) >= 2 and sample.shape[-1] == 3:
                                has_image_data = True
                                print(f"    ** Contains image data (detected in object array)")
            
            # Make final determination based on content
            if has_image_data:
                file_type = "image_based"
                print("\nClassification by content: IMAGE-BASED (contains image-like arrays)")
            elif has_state_characteristics:
                file_type = "state_based"
                print("\nClassification by content: STATE-BASED (contains state information)")
        
        # If still unknown, look at file size as a last resort
        if file_type == "unknown":
            if file_size_mb > 50:  # Image files tend to be much larger
                file_type = "image_based"
                print(f"\nClassification by size: IMAGE-BASED (large file size: {file_size_mb:.2f} MB)")
            else:
                file_type = "state_based"  # Default to state-based
                print(f"\nClassification by default: STATE-BASED (small file size: {file_size_mb:.2f} MB)")
        
        print(f"\nFinal classification: {file_type.upper()}")
        return file_type
    
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description='Analyze .npz files to determine if they contain state or image data.')
    parser.add_argument('--dir', type=str, default='.', help='Directory to search for .npz files')
    parser.add_argument('--file', type=str, help='Specific .npz file to analyze')
    args = parser.parse_args()
    
    # Track results
    results = {
        "image_based": [],
        "state_based": [],
        "unknown": [],
        "error": []
    }
    
    if args.file:
        file_type = analyze_npz_file(args.file)
        results[file_type].append(os.path.basename(args.file))
    else:
        # Find all .npz files in the directory
        npz_files = glob.glob(os.path.join(args.dir, "*.npz"))
        print(f"Found {len(npz_files)} .npz files in {args.dir}")
        
        for file_path in npz_files:
            file_type = analyze_npz_file(file_path)
            results[file_type].append(os.path.basename(file_path))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nImage-based files ({len(results['image_based'])}):")
    for file in results['image_based']:
        print(f"  - {file}")
    
    print(f"\nState-based files ({len(results['state_based'])}):")
    for file in results['state_based']:
        print(f"  - {file}")
    
    print(f"\nUnknown type files ({len(results['unknown'])}):")
    for file in results['unknown']:
        print(f"  - {file}")
    
    print(f"\nError files ({len(results['error'])}):")
    for file in results['error']:
        print(f"  - {file}")

if __name__ == "__main__":
    main()