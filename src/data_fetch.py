import requests
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def get_capabilities():
    """
    Fetch WMS capabilities from NASA GIBS to check available layers and parameters.
    """
    try:
        logging.info("Fetching WMS capabilities from NASA GIBS...")
        wms_url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi'
        params = {
            'service': 'WMS',
            'request': 'GetCapabilities',
            'VERSION': '1.3.0'
        }
        response = requests.get(wms_url, params=params, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        # Save the capabilities response to a file for review
        with open('data/wms_capabilities.xml', 'w') as file:
            file.write(response.text)
        logging.info("WMS capabilities saved to data/wms_capabilities.xml.")
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch WMS capabilities: {e}")
    except Exception as ex:
        logging.error(f"Unexpected error: {ex}")

def fetch_ghg_data(save_path='data/ghg_map.png'):
    """
    Fetch GHG imagery from NASA GIBS and save it to the specified path.
    """
    try:
        logging.info("Fetching GHG data from NASA GIBS...")
        wms_url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi'
        params = {
            'service': 'WMS',
            'request': 'GetMap',
            'VERSION': '1.3.0',
            'LAYERS': 'MERRA2_2m_Air_Temperature_Monthly',  # Updated layer name
            'CRS': 'EPSG:4326',  # Changed SRS to CRS
            'BBOX': '-180,-90,180,90',
            'WIDTH': '1200',
            'HEIGHT': '600',
            'FORMAT': 'image/png',
            'TIME': '2021-09-01',  # Adjusted to a valid date
            'TRANSPARENT': 'True'
        }

        response = requests.get(wms_url, params=params, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        if response.headers['Content-Type'].startswith('image/png'):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as file:
                file.write(response.content)
            logging.info(f"GHG data saved to {save_path}.")
        else: #handle errors and Log info
            logging.error(f"The fetched content is not an image/png. Content-Type: {response.headers['Content-Type']}")
            logging.debug(f"Response content: {response.text[:500]}")  # Log the first 500 characters of the response
            logging.info("Fetching WMS capabilities for diagnostics...")
            get_capabilities()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch GHG data: {e}")
    except Exception as ex:
        logging.error(f"Unexpected error: {ex}")

if __name__ == "__main__":
    fetch_ghg_data()