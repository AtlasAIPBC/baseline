import os
import json
from multiprocessing import cpu_count
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image
import cv2
from code.data_ml_functions.dataFunctions import _process_file, prepare_data


# Define your configuration parameters
params = {
    'image_format': 'jpg',  # Image file format
    'target_img_size': (224,224),  # Target image size (width, height)
    'metadata_length': 45,  # Length of metadata feature vector
    'category_names': [
        "airport", "car_dealership", "flooded_road", "lake_or_pond", "place_of_worship", 
        "shipyard", "swimming_pool","airport_hangar", "construction_site", "fountain", 
        "lighthouse", "police_station", "shopping_mall", "toll_booth", "airport_terminal", 
        "crop_field", "gas_station", "military_facility", "port", "single-unit_residential",
        "tower","amusement_park", "dam", "golf_course", "multi-unit_residential", "prison", 
        "smokestack", "tunnel_opening", "aquaculture", "debris_or_rubble", "ground_transportation_station", 
        "nuclear_powerplant", "race_track", "solar_farm", "waste_disposal", "archaeological_site", 
        "educational_institution", "helipad", "office_building", "railway_bridge", "space_facility", "water_treatment_facility",
        "barn", "electric_substation", "hospital", "oil_or_gas_facility", "recreational_facility", "stadium", "wind_farm",
        "border_checkpoint", "factory_or_powerplant", "impoverished_settlement", "park", "road_bridge", "storage_tank", "zoo",
        "burial_site", "fire_station", "interchange", "parking_lot_or_garage", "runway", "surface_mine"
    ],
    'num_workers': cpu_count(),  # Number of parallel workers for processing
    'directories': {
        'dataset': '/home/ada/satmae/temporal/data/fmow',  # Root folder of your dataset
        'train_data': '/home/ada/satmae/temporal/preprocessed/fmow/train',  # Directory to save preprocessed training data
        'test_data': '/home/ada/satmae/temporal/preprocessed/fmow/test'  # Directory to save preprocessed test data
    },
    'files': {
        'test_struct': '/home/ada/satmae/temporal/preprocessed/fmow/test_struct.json',  # File path for preprocessed test data structure
        'training_struct': '/home/ada/satmae/temporal/preprocessed/fmow/training_struct.json',  # File path for preprocessed training data structure
        'dataset_stats': '/home/ada/satmae/temporal/preprocessed/fmow/dataset_stats.json'  # File path for dataset statistics
    }
}

# Call the prepare_data function
prepare_data(params)
