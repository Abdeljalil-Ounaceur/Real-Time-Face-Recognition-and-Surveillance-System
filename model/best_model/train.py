import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras.layers import Dense, Input
from model_utils import load_dataset

def create_feature_compressor():
    """Compresses 2048D -> 128D"""
    inputs = tf.keras.Input(shape=(2048,))
    outputs = Dense(128, activation='linear')(inputs)

    return tf.keras.Model(inputs, outputs, name='compressor')

def create_feature_decompressor():
    """Decompresses 128D -> 2048D"""
    inputs = tf.keras.Input(shape=(128,))
    outputs = Dense(2048, activation='relu')(inputs)
    return tf.keras.Model(inputs, outputs, name='decompressor')

def train_feature_compression(dataset_path, batch_size=32, epochs=50):
    # Create models
    base_vggface = VGGFace(model='senet50', include_top=False, pooling='avg')
    compressor = create_feature_compressor()
    decompressor = create_feature_decompressor()
    base_vggface.trainable = False
    
    # Build training model
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    vggface_features = base_vggface(input_layer)
    compressed = compressor(vggface_features)
    reconstructed = decompressor(compressed)
    
    training_model = tf.keras.Model(input_layer, reconstructed)
    #training_model.load_weights('best_compressor.h5', by_name=True)
    
    # Configure training
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae', "accuracy"]
    )
    
    # Load and split dataset
    images = load_dataset(dataset_path, crop=True, preprocess=True)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    # Get features for training and validation
    train_features = base_vggface.predict(train_images, batch_size=batch_size)
    val_features = base_vggface.predict(val_images, batch_size=batch_size)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_compressor.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train with proper validation data
    history = training_model.fit(
        train_images,
        train_features,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_images, val_features),
        callbacks=callbacks,
        shuffle=True
    )
    
    # Save best compressor
    compressor.save('feature_compressor_128d.h5')
    
    return compressor, history

def load_model(model_type='base', compressor_path='feature_compressor_128d.h5', threshold=0):
    # Base model
    base_vggface = VGGFace(model='senet50', include_top=False, pooling='avg')
    base_vggface.trainable = False
    
    if model_type == 'base':
        return base_vggface

    # Compressed model components    
    compressor = tf.keras.models.load_model(compressor_path)
    compressor.trainable = False
    
    inputs = Input(shape=(224, 224, 3))
    x = base_vggface(inputs)
    x = compressor(x)
    
    if model_type == 'compressed':
        return tf.keras.Model(inputs, x, name='compressed_feature_extractor')

    # Binary model additional components
    x = tf.keras.layers.Lambda(
        lambda x: (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
    )(x)
    outputs = tf.keras.layers.Lambda(
        lambda x: tf.cast(x > threshold, dtype=tf.float32)
    )(x)
    
    return tf.keras.Model(inputs, outputs, name='binary_feature_extractor')

def calculate_mean_features(images, model_type="compressed"):
    model = load_model(model_type=model_type)
    features = model.predict(images)
    mean_features = np.mean(features, axis=0)
    return mean_features


def binarize_features(features, threshold=0):
    normalized = (features - np.mean(features)) / np.std(features)
    binary = (normalized > threshold).astype(np.float32)
    return binary




if __name__ == "__main__":

    # Train feature compressor
    train_feature_compression("gt_db", batch_size=32, epochs=50)


    # Usage
    # results = calculate_feature_means('gt_db_cropped', limit=1000)
    # print(f"\nResults for {results['processed_count']} images:")
    # print(f"Overall mean: {results['overall_mean']:.4f}")
    # print(f"Standard deviation: {results['std_dev']:.4f}")
    # print(f"Mean min: {results['mean_min']:.4f}")
    # print(f"Mean max: {results['mean_max']:.4f}")