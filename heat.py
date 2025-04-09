import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import glob
import torch
import folium
from folium.plugins import HeatMap
import requests
import json
from io import BytesIO
from PIL import Image

def find_model_files(directory):
    """
    Find all .h5 and .pth files in the given directory and its subdirectories.
    
    Args:
        directory: Directory to search in
        
    Returns:
        List of paths to model files
    """
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5') or file.endswith('.hdf5') or file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    return model_files

def visualize_model_weights(file_path, features=None):
    """
    Visualize model weights from an HDF5 or PyTorch file.
    
    Args:
        file_path: Path to the model file
        features: List of feature names (optional)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.h5', '.hdf5']:
        try:
            # Try to open the HDF5 file
            with h5py.File(file_path, 'r') as h5_file:
                print("Successfully opened HDF5 file")
                
                # Print file structure for debugging
                print("\nFile structure:")
                def print_structure(name):
                    print(f"  - {name}")
                h5_file.visit(print_structure)
                
                # Try to find weights
                weights = None
                weights_path = None
                
                # Common paths for weights in different model formats
                possible_paths = [
                    'model_weights/dense/dense/kernel:0',
                    'model_weights/dense_1/dense_1/kernel:0',
                    'dense/dense/kernel:0',
                    'dense_1/dense_1/kernel:0',
                    'kernel:0'
                ]
                
                for path in possible_paths:
                    if path in h5_file:
                        weights_path = path
                        weights = h5_file[path][:]
                        print(f"\nFound weights at path: {path}")
                        break
                
                if weights is None:
                    print("\nCould not find weights in the expected locations.")
                    print("Available keys in the file:")
                    for key in h5_file.keys():
                        print(f"  - {key}")
                    return
                
                # Create heatmap for the weights
                create_heatmap(weights, features, "HDF5", weights_path)
                
        except Exception as e:
            print(f"Error processing HDF5 file: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Verify the file is a valid HDF5 file")
            print("2. Check if the file is corrupted")
            print("3. Try opening the file with HDFView or similar tools")
            print("4. Ensure you have the correct permissions to access the file")
    
    elif file_ext == '.pth':
        try:
            # Try to load the PyTorch model
            print("Loading PyTorch model file...")
            model_data = torch.load(file_path, map_location=torch.device('cpu'))
            
            # Check if it's a state dict or a full model
            if isinstance(model_data, dict):
                print("\nModel structure (state dict keys):")
                for key in model_data.keys():
                    print(f"  - {key}")
                
                # Try to find weight matrices
                weights = None
                weight_key = None
                
                # Common weight key patterns
                weight_patterns = ['weight', 'kernel', 'W', 'w']
                
                for key in model_data.keys():
                    for pattern in weight_patterns:
                        if pattern in key.lower() and len(model_data[key].shape) >= 2:
                            weights = model_data[key].numpy()
                            weight_key = key
                            print(f"\nFound weights at key: {key}")
                            break
                    if weights is not None:
                        break
                
                if weights is None:
                    print("\nCould not find weight matrices in the model.")
                    return
                
                # Create heatmap for the weights
                create_heatmap(weights, features, "PyTorch", weight_key)
                
            else:
                print("Loaded data is not a dictionary. It might be a full model object.")
                print("Try saving the model's state_dict instead.")
                
        except Exception as e:
            print(f"Error processing PyTorch model file: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Verify the file is a valid PyTorch model file")
            print("2. Check if the file is corrupted")
            print("3. Ensure you have the correct permissions to access the file")
    
    else:
        print(f"Unsupported file format: {file_ext}")

def create_heatmap(weights, features, model_type, weight_name):
    """
    Create a heatmap visualization for model weights.
    
    Args:
        weights: Weight matrix as numpy array
        features: List of feature names (optional)
        model_type: Type of model (HDF5 or PyTorch)
        weight_name: Name or path of the weight in the model
    """
    # Get the shape of the weights
    rows, cols = weights.shape
    print(f"Weight matrix shape: {rows} x {cols}")
    
    # If features not provided or don't match the weight matrix shape, create generic names
    if features is None or len(features) != rows:
        print(f"Using generic feature names (provided {len(features) if features else 0}, needed {rows})")
        features = [f"Feature_{i+1}" for i in range(rows)]
    
    # Create DataFrame for heatmap
    df_weights = pd.DataFrame(weights, index=features)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_weights, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Feature Weights Heatmap ({model_type} - {weight_name})')
    plt.xlabel('Output Neurons')
    plt.ylabel('Features')
    plt.tight_layout()
    
    # Save the plot
    output_path = f'weights_heatmap_{model_type.lower()}.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nHeatmap saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # If the weight matrix is large, create a more focused visualization
    if rows > 10 or cols > 10:
        print("\nCreating a more focused visualization of the first 10x10 elements...")
        
        # Select the first 10 rows and columns
        focused_weights = weights[:min(10, rows), :min(10, cols)]
        focused_features = features[:min(10, rows)]
        
        # Create DataFrame for focused heatmap
        df_focused = pd.DataFrame(focused_weights, index=focused_features)
        
        # Plot focused heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_focused, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Focused Feature Weights Heatmap ({model_type} - {weight_name})')
        plt.xlabel('Output Neurons (first 10)')
        plt.ylabel('Features (first 10)')
        plt.tight_layout()
        
        # Save the focused plot
        focused_output_path = f'focused_weights_heatmap_{model_type.lower()}.png'
        plt.savefig(focused_output_path, dpi=300)
        print(f"Focused heatmap saved to: {focused_output_path}")
        
        # Show the focused plot
        plt.show()
    
    # Try to create a map-based visualization if features might be zipcodes
    try:
        create_zipcode_map(weights, features, model_type, weight_name)
    except Exception as e:
        print(f"Could not create zipcode map: {str(e)}")

def create_zipcode_map(weights, features, model_type, weight_name):
    """
    Create a map-based visualization of weights based on zipcodes.
    
    Args:
        weights: Weight matrix as numpy array
        features: List of feature names
        model_type: Type of model (HDF5 or PyTorch)
        weight_name: Name or path of the weight in the model
    """
    # Check if any features look like zipcodes (5-digit numbers)
    zipcode_features = []
    for feature in features:
        # Check if the feature is a 5-digit number (zipcode)
        if isinstance(feature, str) and feature.isdigit() and len(feature) == 5:
            zipcode_features.append(feature)
    
    if not zipcode_features:
        print("\nNo zipcode features found. Skipping map visualization.")
        return
    
    print(f"\nFound {len(zipcode_features)} zipcode features. Creating map visualization...")
    
    # Create a DataFrame with zipcodes and their average weights
    zipcode_data = []
    for i, zipcode in enumerate(zipcode_features):
        # Calculate the average weight for this zipcode across all outputs
        avg_weight = np.mean(weights[i, :])
        zipcode_data.append({
            'zipcode': zipcode,
            'weight': avg_weight
        })
    
    df_zipcodes = pd.DataFrame(zipcode_data)
    
    # Get coordinates for each zipcode
    coordinates = []
    for zipcode in zipcode_features:
        try:
            # Use a free API to get coordinates for zipcodes
            response = requests.get(f"https://public.opendatasoft.com/api/records/1.0/search/?dataset=us-zip-code-latitude-and-longitude&q={zipcode}")
            data = response.json()
            
            if data['records']:
                lat = data['records'][0]['fields']['latitude']
                lng = data['records'][0]['fields']['longitude']
                coordinates.append((zipcode, lat, lng))
                print(f"Found coordinates for zipcode {zipcode}: {lat}, {lng}")
            else:
                print(f"Could not find coordinates for zipcode {zipcode}")
        except Exception as e:
            print(f"Error getting coordinates for zipcode {zipcode}: {str(e)}")
    
    if not coordinates:
        print("Could not find coordinates for any zipcodes. Skipping map visualization.")
        return
    
    # Create a map centered on the US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Add markers for each zipcode
    for zipcode, lat, lng in coordinates:
        # Find the weight for this zipcode
        weight = df_zipcodes[df_zipcodes['zipcode'] == zipcode]['weight'].values[0]
        
        # Create a color based on the weight
        color = 'red' if weight > 0 else 'blue'
        
        # Add a marker
        folium.CircleMarker(
            location=[lat, lng],
            radius=10,
            color=color,
            fill=True,
            popup=f'Zipcode: {zipcode}<br>Weight: {weight:.4f}'
        ).add_to(m)
    
    # Save the map
    map_path = f'zipcode_weights_map_{model_type.lower()}.html'
    m.save(map_path)
    print(f"Zipcode map saved to: {map_path}")
    
    # Create a heatmap layer
    heat_data = [[lat, lng, abs(weight)] for zipcode, lat, lng in coordinates 
                 for weight in [df_zipcodes[df_zipcodes['zipcode'] == zipcode]['weight'].values[0]]]
    
    HeatMap(heat_data).add_to(m)
    
    # Save the heatmap
    heatmap_path = f'zipcode_weights_heatmap_{model_type.lower()}.html'
    m.save(heatmap_path)
    print(f"Zipcode heatmap saved to: {heatmap_path}")
    
    print("\nTo view the maps, open the HTML files in a web browser.")

def create_sample_heatmap(features):
    """
    Create a sample heatmap with random weights for demonstration.
    
    Args:
        features: List of feature names
    """
    # Create random weights
    num_features = len(features)
    num_outputs = 5  # Example number of output neurons
    weights = np.random.randn(num_features, num_outputs)
    
    # Create DataFrame for heatmap
    df_weights = pd.DataFrame(weights, index=features)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_weights, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Sample Feature Weights Heatmap (Random Data)')
    plt.xlabel('Output Neurons')
    plt.ylabel('Features')
    plt.tight_layout()
    
    # Save the plot
    output_path = 'sample_weights_heatmap.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nSample heatmap saved to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Define features
    features = [
        "POS_DIST_ED_ZP",
        "POS_DIST_CLINIC_ZP",
        "POS_DIST_MEDSURG_ICU_ZP",
        "POS_DIST_TRAUMA_ZP"
    ]
    
    # Original file path
    file_path = 'C:/Users/rajan/OneDrive/Desktop/Hack to Future/demo/models/best_healthcare_model.pth'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("\nSearching for model files in the current directory and subdirectories...")
        
        # Get the current directory
        current_dir = os.getcwd()
        model_files = find_model_files(current_dir)
        
        if model_files:
            print("\nFound the following model files:")
            for i, file in enumerate(model_files, 1):
                print(f"{i}. {file}")
            
            # Ask user to select a file
            try:
                selection = int(input("\nEnter the number of the file to use (or 0 to create a sample heatmap): "))
                if 1 <= selection <= len(model_files):
                    file_path = model_files[selection-1]
                    print(f"\nUsing file: {file_path}")
                    visualize_model_weights(file_path, features)
                elif selection == 0:
                    print("\nCreating a sample heatmap with random data...")
                    create_sample_heatmap(features)
                else:
                    print("Invalid selection. Creating a sample heatmap instead.")
                    create_sample_heatmap(features)
            except ValueError:
                print("Invalid input. Creating a sample heatmap instead.")
                create_sample_heatmap(features)
        else:
            print("\nNo model files found in the current directory or subdirectories.")
            print("Creating a sample heatmap with random data instead...")
            create_sample_heatmap(features)
    else:
        # Try to visualize the weights
        visualize_model_weights(file_path, features)
