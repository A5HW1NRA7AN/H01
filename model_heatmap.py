import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
from flask import Flask, render_template, jsonify
import threading
import time
import random
import torch
import os
import sys

class ModelHeatmap:
    def __init__(self, model_path=None):
        self.app = Flask(__name__)
        self.data_points = []
        self.max_points = 1000  # Maximum number of points to keep in memory
        
        # Load PyTorch model if path is provided
        self.model = None
        self.model_weights = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.load(model_path, map_location=torch.device('cpu'))
                print(f"Model loaded successfully from {model_path}")
                
                # Extract weights from the model
                if isinstance(self.model, dict):
                    # If it's a state dict, find the first weight matrix
                    for key, value in self.model.items():
                        if 'weight' in key and len(value.shape) >= 2:
                            self.model_weights = value.numpy()
                            print(f"Using weights from {key} with shape {self.model_weights.shape}")
                            break
                else:
                    # If it's a full model, try to find weights in the first layer
                    for name, param in self.model.named_parameters():
                        if 'weight' in name and len(param.shape) >= 2:
                            self.model_weights = param.detach().numpy()
                            print(f"Using weights from {name} with shape {self.model_weights.shape}")
                            break
                
                if self.model_weights is None:
                    print("Could not extract weights from the model. Using random weights instead.")
                    self.model_weights = np.random.randn(10, 10)  # Default size if no weights found
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random weights instead.")
                self.model_weights = np.random.randn(10, 10)
        else:
            print("No model path provided or file not found. Using random weights.")
            self.model_weights = np.random.randn(10, 10)
        
        # US cities with their coordinates (latitude, longitude)
        self.us_cities = [
            ("New York", 40.7128, -74.0060),
            ("Los Angeles", 34.0522, -118.2437),
            ("Chicago", 41.8781, -87.6298),
            ("Houston", 29.7604, -95.3698),
            ("Phoenix", 33.4484, -112.0740),
            ("Philadelphia", 39.9526, -75.1652),
            ("San Antonio", 29.4241, -98.4936),
            ("San Diego", 32.7157, -117.1611),
            ("Dallas", 32.7767, -96.7970),
            ("San Jose", 37.3382, -121.8863)
        ]
        
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('heatmap.html')
        
        @self.app.route('/get_data')
        def get_data():
            return jsonify(self.data_points)
    
    def create_base_map(self):
        """Create a base map centered on the US"""
        return folium.Map(
            location=[39.8283, -98.5795],  # Center of the US
            zoom_start=4,
            tiles='CartoDB positron'
        )
    
    def add_heatmap_layer(self, m):
        """Add a heatmap layer to the map"""
        if self.data_points:
            heat_data = [[point['lat'], point['lng'], point['intensity']] 
                        for point in self.data_points]
            HeatMap(heat_data).add_to(m)
    
    def get_model_influenced_intensity(self, city_index):
        """Get intensity value influenced by the model weights"""
        # Use the city index to get a value from the model weights
        # Normalize the value to be between 0 and 1
        if self.model_weights is not None:
            # Get a value from the model weights based on the city index
            # We'll use the first row and column of the weights matrix
            row_idx = city_index % self.model_weights.shape[0]
            col_idx = city_index % self.model_weights.shape[1]
            raw_value = self.model_weights[row_idx, col_idx]
            
            # Normalize to 0-1 range using min-max scaling
            min_val = np.min(self.model_weights)
            max_val = np.max(self.model_weights)
            if max_val > min_val:
                normalized_value = (raw_value - min_val) / (max_val - min_val)
            else:
                normalized_value = 0.5  # Default if all values are the same
            
            # Add some randomness to make it more dynamic
            normalized_value = 0.7 * normalized_value + 0.3 * random.random()
            
            return max(0, min(1, normalized_value))  # Ensure it's between 0 and 1
        
        return random.random()  # Fallback to random if no model weights
    
    def update_data_points(self):
        """Update data points with values influenced by the model"""
        city_index = 0
        while True:
            # Select a city in a round-robin fashion
            city, lat, lng = self.us_cities[city_index]
            city_index = (city_index + 1) % len(self.us_cities)
            
            # Get intensity influenced by the model
            intensity = self.get_model_influenced_intensity(city_index)
            
            # Create new data point
            new_point = {
                'city': city,
                'lat': lat + random.uniform(-0.5, 0.5),  # Add some randomness to location
                'lng': lng + random.uniform(-0.5, 0.5),
                'intensity': intensity,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add new point and maintain maximum size
            self.data_points.append(new_point)
            if len(self.data_points) > self.max_points:
                self.data_points.pop(0)
            
            time.sleep(1)  # Update every second
    
    def run(self):
        """Run the Flask application"""
        # Start data simulation in a separate thread
        data_thread = threading.Thread(target=self.update_data_points)
        data_thread.daemon = True
        data_thread.start()
        
        # Run Flask app
        self.app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    # Check if a model path was provided as a command-line argument
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    heatmap = ModelHeatmap(model_path)
    heatmap.run() 