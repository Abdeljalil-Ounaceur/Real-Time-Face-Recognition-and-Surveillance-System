import ast
import numpy as np
from PIL import Image
from model_utils import *

# Define global variables
data_list = []
data_file = r'data_file.txt'

# Load binary model that outputs hashes directly
model = load_model(model_type='binary')

def load_data_to_list(filename="data_file.txt"):
    """Load group hashes from a file into a list."""
    data_list = []
    with open(filename, 'r') as file:
        group_hashes = file.readlines()

    for line in group_hashes:
        group_tuple = ast.literal_eval(line.strip())
        # Ensure group_hash is a string
        group_hash = str(group_tuple[0])
        data_list.append((group_hash, group_tuple[1]))
    
    print(f"Loaded {len(group_hashes)} people's hashes with their informations from the file.")
    
    return data_list

def convert_to_hash_string(binary_array):
    return ''.join(str(int(bit)) for bit in binary_array)

def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


# Load the group list on script start
data_list = load_data_to_list(data_file)

min_distance = 35
def match_face_info(new_hash):
    
    new_hash = convert_to_hash_string(new_hash)
    print(f"New face hash: {new_hash}")
    min_dist = np.inf
    matched_info = []
    
    for Hash, info in data_list:
        distance = hamming_distance(new_hash, Hash)
        if distance <= min_distance:
            if distance < min_dist:
                min_dist = distance
                matched_info = info
                
    if min_dist > min_distance:
        print("Unknown face.")
        matched_info = ['','Unknown','person']
        
    print(f"Similar person: {matched_info}")
    print(f"Minimum distance: {min_dist}")
    return matched_info
    
def predict_faces_info_from_image(image):
    try:
        coordinates = extract_coordinates(image)
        actual_faces = extract_faces(image, coordinates)
        new_hash = preprocess_and_predict(model,actual_faces)

        faces_info = []
        for (face_coordinates,new_hash) in zip(coordinates,new_hash):
            matched_info = face_coordinates,match_face_info(new_hash)
            faces_info.append(matched_info)   
        return faces_info
    
    except ValueError as e:
        print(e)
        return []