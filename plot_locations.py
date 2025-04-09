import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os

def plot_locations():
    # Load the data
    try:
        df = pd.read_csv('data/loc.csv')
        print(f"Loaded {len(df)} location points")
        
        # Create a map centered on the mean coordinates
        center_lat = df['latitude'].mean()
        center_lng = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lng], zoom_start=4)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude'], row.get('intensity', 1.0)] 
                    for _, row in df.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Add markers for each point
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='red',
                fill=True,
                popup=f"Lat: {row['latitude']:.4f}, Lng: {row['longitude']:.4f}"
            ).add_to(m)
        
        # Save the map
        m.save('location_map.html')
        print("Map saved as 'location_map.html'")
        
        # Create a scatter plot
        plt.figure(figsize=(12, 8))
        
        # Check if intensity column exists and has valid values
        if 'intensity' in df.columns and not df['intensity'].isna().all():
            scatter = plt.scatter(df['longitude'], df['latitude'], 
                               c=df['intensity'], 
                               cmap='viridis', 
                               alpha=0.6,
                               s=10)  # Smaller point size for better visibility
            plt.colorbar(scatter, label='Intensity')
        else:
            # If no intensity data, plot points in a single color
            plt.scatter(df['longitude'], df['latitude'], 
                       c='blue', 
                       alpha=0.6,
                       s=10)
        
        plt.title('Location Points Distribution')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        
        # Adjust plot limits to show all points
        plt.xlim(df['longitude'].min() - 1, df['longitude'].max() + 1)
        plt.ylim(df['latitude'].min() - 1, df['latitude'].max() + 1)
        
        plt.savefig('location_scatter.png', dpi=300, bbox_inches='tight')
        print("Scatter plot saved as 'location_scatter.png'")
        
        # Print statistics
        print("\nData Statistics:")
        print(f"Total points: {len(df)}")
        print(f"Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        if 'intensity' in df.columns and not df['intensity'].isna().all():
            print(f"Intensity range: {df['intensity'].min():.4f} to {df['intensity'].max():.4f}")
            print(f"Mean intensity: {df['intensity'].mean():.4f}")
        
        # Print sample of the data
        print("\nSample of the data:")
        print(df.head())
        
    except FileNotFoundError:
        print("Error: data/loc.csv not found. Please run app.py first to generate sample data.")
    except Exception as e:
        print(f"Error: {str(e)}")
        # Print more detailed error information
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    plot_locations() 