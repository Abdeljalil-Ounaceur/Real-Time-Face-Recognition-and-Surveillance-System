import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import load_model, extract_coordinates, extract_faces, preprocess_and_predict
from PIL import Image

def load_and_process_image(image_path, model):
    """Load and process a single image to get its binary hash"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        coordinates = extract_coordinates(img_array)
        faces = extract_faces(img_array, coordinates)
        hash_vector = preprocess_and_predict(model, faces)
        # Ensure we get a 128-bit hash
        return hash_vector.flatten()[:128]  # Take first 128 bits if longer
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def analyze_mask_differences(root_folder):
    """Analyze differences between masked and unmasked images"""
    model = load_model(model_type='binary')
    
    all_differences = []
    bit_flip_counts = np.zeros(128)
    pair_labels = []  # Track which pairs we're comparing
    
    for person_folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, person_folder)
        if not os.path.isdir(folder_path):
            continue
            
        pairs = [
            ('01.jpg', '16.png'),
            ('02.jpg', '17.png')
        ]
        
        for normal_img, masked_img in pairs:
            normal_path = os.path.join(folder_path, normal_img)
            masked_path = os.path.join(folder_path, masked_img)
            
            if not (os.path.exists(normal_path) and os.path.exists(masked_path)):
                continue
                
            normal_hash = load_and_process_image(normal_path, model)
            masked_hash = load_and_process_image(masked_path, model)
            
            if normal_hash is not None and masked_hash is not None:
                difference = np.abs(normal_hash - masked_hash)
                all_differences.append(difference)
                bit_flip_counts += difference
                pair_labels.append(f"{person_folder}\n({normal_img}-{masked_img})")
    
    all_differences = np.array(all_differences)
    if len(all_differences) == 0:
        print("No valid image pairs found!")
        return
        
    create_visualizations(all_differences, bit_flip_counts, pair_labels)

def create_visualizations(all_differences, bit_flip_counts, pair_labels):
    # Set style for better visualization
    plt.style.use('seaborn')
    
    # 1. Enhanced Bit Flip Heatmap
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    sns.heatmap(all_differences, 
                cmap='YlOrRd',
                cbar_kws={'label': 'Bit Flipped'},
                yticklabels=pair_labels)
    plt.title('Bit Flip Patterns Across Image Pairs')
    plt.xlabel('Bit Position')
    
    # 2. Bit Flip Frequency
    plt.subplot(1, 2, 2)
    # Highlight most frequently flipped bits
    colors = ['red' if count > np.mean(bit_flip_counts) else 'blue' 
             for count in bit_flip_counts]
    plt.bar(range(128), bit_flip_counts, color=colors)
    plt.title('Frequency of Bit Flips by Position')
    plt.xlabel('Bit Position')
    plt.ylabel('Number of Flips')
    
    plt.tight_layout()
    plt.show()

    # 3. Distribution Plot
    plt.figure(figsize=(12, 6))
    changes_per_pair = np.sum(all_differences, axis=1)
    sns.histplot(changes_per_pair, bins=20, kde=True)
    plt.axvline(np.mean(changes_per_pair), color='r', linestyle='--', 
                label=f'Mean: {np.mean(changes_per_pair):.1f}')
    plt.axvline(np.median(changes_per_pair), color='g', linestyle='--', 
                label=f'Median: {np.median(changes_per_pair):.1f}')
    plt.title('Distribution of Changed Bits per Image Pair')
    plt.xlabel('Number of Changed Bits')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Print statistics
    print("\nAnalysis Statistics:")
    print(f"Total image pairs analyzed: {len(all_differences)}")
    print(f"Average bits changed: {np.mean(changes_per_pair):.2f}")
    print(f"Median bits changed: {np.median(changes_per_pair):.2f}")
    print(f"Min bits changed: {np.min(changes_per_pair)}")
    print(f"Max bits changed: {np.max(changes_per_pair)}")
    
    # Show top changing bits
    top_n = 10
    top_indices = np.argsort(bit_flip_counts)[-top_n:][::-1]
    print(f"\nTop {top_n} most frequently changed bits:")
    for idx, pos in enumerate(top_indices, 1):
        print(f"{idx}. Bit {pos}: {bit_flip_counts[pos]:.0f} flips " 
              f"({(bit_flip_counts[pos]/len(all_differences))*100:.1f}% of pairs)")

if __name__ == "__main__":
    root_folder = "mask_nomask"
    analyze_mask_differences(root_folder)