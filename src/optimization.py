import numpy as np
import pandas as pd
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def recommend_land_use_changes(lat, lon, df):
    """Generate recommendations for land use changes based on coordinates."""
    try:
        # Ensure df is valid
        if df is None or df.empty:
            logging.error("Dataframe is empty or not loaded.")
            return "No recommendations available."

        # Calculate distances and find the closest region
        closest_region = None
        closest_distance = float('inf')

        for index, row in df.iterrows():
            # Calculate Euclidean distance
            distance = np.sqrt((lat - row['latitude']) ** 2 + (lon - row['longitude']) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_region = row

        # Generate recommendations based on closest region's characteristics
        if closest_region is not None:
            if closest_region['sequestration_potential'] < 50:
                return f"Increase forest cover in {closest_region['region']}."
            else:
                return f"Maintain current land use in {closest_region['region']}."
        else:
            logging.info("No closest region found.")
            return "No recommendations available."

    except Exception as e:
        logging.error(f"Error in generating recommendations: {e}")
        return "Error generating recommendations."

def load_data(file_path):
    """Load preprocessed data."""
    try:
        logging.info(f"Loading data from {file_path}...")
        data = np.load(file_path, allow_pickle=True).item()
        df = pd.DataFrame(data)  # Convert to DataFrame for processing
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    df = load_data('data/preprocessed_data.npy')
    lat = 20.1  # Example latitude
    lon = 0.1   # Example longitude
    recommendation = recommend_land_use_changes(lat, lon, df)
    print(recommendation)