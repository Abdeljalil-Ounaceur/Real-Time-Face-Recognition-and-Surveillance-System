import os
import cv2
import mtcnn
import numpy as np
from keras_vggface.utils import preprocess_input
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras.layers import Input
from PIL import Image

detector = mtcnn.MTCNN()

def load_model(model_type='base', compressor_path='feature_compressor_128d.h5', threshold=0):

    # Load base VGGFace model
    base_vggface = VGGFace(model='senet50', include_top=False, pooling='avg')
    base_vggface.trainable = False
    
    if model_type == 'base':
        return base_vggface
        
    elif model_type == 'compressed':
        if not os.path.exists(compressor_path):
            raise ValueError(f"Compressor model not found at {compressor_path}")
            
        # Load compressor
        compressor = tf.keras.models.load_model(compressor_path)
        compressor.trainable = False
        
        # Build combined model
        inputs = Input(shape=(224, 224, 3))
        x = base_vggface(inputs)
        outputs = compressor(x)
        
        return tf.keras.Model(inputs, outputs, name='compressed_feature_extractor')
        
    elif model_type == 'binary':
        if not os.path.exists(compressor_path):
            raise ValueError(f"Compressor model not found at {compressor_path}")
            
        # Load compressor
        compressor = tf.keras.models.load_model(compressor_path)
        compressor.trainable = False
        
        # Build binary model
        inputs = Input(shape=(224, 224, 3))
        x = base_vggface(inputs)
        x = compressor(x)
        
        # Add normalization
        x = tf.keras.layers.Lambda(
            lambda x: (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
        )(x)
        
        # Add thresholding
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.cast(x > threshold, dtype=tf.float32)
        )(x)
        
        return tf.keras.Model(inputs, outputs, name='binary_feature_extractor')
        
    else:
        raise ValueError("model_type must be one of: 'base', 'compressed', 'binary'")

def extract_coordinates(image):
    faces_data = detector.detect_faces(image)
    if len(faces_data) == 0:
        raise ValueError("No face detected in image")
    coordinates = []
    for face in faces_data:
        x, y, width, height = face['box']
        coordinates.append((int(x), int(y), int(width), int(height)))
    return coordinates

def extract_faces(pixels, coordinates, required_size=(224, 224)):    
    faces = []
    for (x, y, width, height) in coordinates:
        x2, y2 = x + width, y + height
        face = pixels[y:y2, x:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        faces.append(face_array)
    return faces

def preprocess_and_predict(model,face_arrays):
    samples = np.asarray(face_arrays, 'float32')
    samples = preprocess_input(samples, version=2)
    output = model.predict(samples)
    return output

def image_to_hash(model, image):
    coordinates = extract_coordinates(image)
    actual_faces = extract_faces(image, coordinates)
    hash = preprocess_and_predict(model,actual_faces)
    return hash


def load_dataset(dataset_path, crop=False, preprocess=True):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path '{dataset_path}' does not exist!")

    all_faces = []
    valid_extensions = {'.png', '.jpg', '.jpeg'}

    print(f"Scanning directory: {dataset_path}")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(tuple(valid_extensions)):
                img_path = os.path.join(root, file)
                try:
                    # Load and convert to RGB (VGGFace requirement)
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    img_array = cv2.resize(img_array, (224, 224))
                    if crop:
                        coordinates = extract_coordinates(img_array)
                        actual_faces = extract_faces(img_array, coordinates)
                    
                    all_faces.append((actual_faces[0] if crop else img_array).reshape(224, 224, 3))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        print(f"Found {len(all_faces)} valid images")
    if not all_faces:
        raise ValueError(f"No valid images found in '{dataset_path}'")
    
    if preprocess:
        faces = np.asarray(all_faces, 'float32')
        faces = preprocess_input(faces, version=2)
        

    print(f"Successfully loaded {len(faces)} images")
    return faces
