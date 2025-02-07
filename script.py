import os
import json
import sqlite3
import numpy as np
import cv2
from skimage import feature
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from image_similarity import *
# from image_processing import *

# -------------------------------
# Local Binary Pattern (LBP) Texture Detection
# -------------------------------
def is_texture_image_lbp(image_path):
    """Determines if an image is a texture-based image using LBP (Local Binary Pattern)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    image_resized = cv2.resize(image, (128, 128))

    # Define LBP parameters
    radius = 3  
    n_points = 8 * radius  

    # Compute LBP features
    lbp = feature.local_binary_pattern(image_resized, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize

    # Measure uniformity
    uniform_ratio = np.sum(lbp_hist[:-1])  
    return uniform_ratio < 0.421

# -------------------------------
# GLCM Feature Extraction
# -------------------------------
def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extracts GLCM (Gray Level Co-occurrence Matrix) features from an image."""
    if len(image.shape) == 3:
        image = rgb2gray(image)  # Convert to grayscale if needed
    image = (image * 255).astype(np.uint8)  # Convert to uint8

    # Compute GLCM and extract properties
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean()
    }
    return features

# -------------------------------
# Database Functions
# -------------------------------
def insert_image_data(image_type, image_path, features):
    """Inserts extracted image features into the SQLite database."""
    conn = sqlite3.connect('image_database.db')
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO Images (image_type, image_path, contrast, dissimilarity, homogeneity, energy, correlation)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (image_type, image_path, features['contrast'], features['dissimilarity'], 
          features['homogeneity'], features['energy'], features['correlation']))

    conn.commit()
    conn.close()

def extract_features_from_folder(folder_path, image_type):
    """Extracts GLCM features from all images in a specified folder and stores them in the database."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = imread(file_path.replace("\\", "/"))
            features = extract_glcm_features(image)
            insert_image_data(image_type, file_path.replace("\\", "/").replace("static/", ""), features)

# -------------------------------
# Image Similarity Calculation
# -------------------------------
def calculate_similarity(query_features, dataset_features):
    """Computes similarity between the query image and images in the dataset."""
    similarities = []
    for image_path, features in dataset_features.items():
        distance = np.linalg.norm(np.array(list(query_features.values())) - np.array(list(features.values())))
        similarities.append((image_path, distance))

    # Sort by similarity (lower distance is better)
    similarities.sort(key=lambda x: x[1])
    return similarities

# -------------------------------
# Query Image Data Function
# -------------------------------
def query_image_data(image_path, database_file="test.json", top_n=80):
    """Finds similar images based on extracted features."""
    
    if is_texture_image_lbp(image_path):  
        # Texture-based image processing
        query_image = imread(image_path)
        conn = sqlite3.connect('image_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Images")

        query_features = extract_glcm_features(query_image)
        rows = cursor.fetchall()

        # Extract features from database
        columns = ["id", "image_type", "image_path", "contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        dataset_features = {
            row[2]: {columns[i]: row[i] for i in range(3, len(columns))}
            for row in rows if row[1] == 'texture'
        }

        # Compute similarity
        results = calculate_similarity(query_features, dataset_features)
        conn.close()
        return [x[0] for x in results[:top_n]]

    else:
        # Form-based image processing
        # classifier = ImageClassifier()
        # list_results = classifier.find_similar_images(image_path)

        main_folder = 'static/images/form'
        if os.path.exists(database_file):
            database = load_image_database(database_file)
        else:
            print("Building image database...")
            database = build_image_database(main_folder, save_path=database_file)

        # Extract features from the query image
        query_hog_features = extract_hog_features(image_path)
        query_vgg_features = extract_vgg_features(image_path)

        similarities = []
        for entry in database:
            image_path = entry['image_path']
            hog_features = entry['hog_features']
            vgg_features = entry['vgg_features']

            # Compute combined similarity
            hog_similarity = calculate_similarity_2(query_hog_features, hog_features)
            vgg_similarity = calculate_similarity_2(query_vgg_features, vgg_features)
            combined_similarity = 0.5 * hog_similarity + 0.5 * vgg_similarity

            similarities.append((image_path, combined_similarity))

        # Sort and return the top results
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return [x[0] for x in similarities[:top_n]]

# -------------------------------
# Optional: Database Cleanup & Management
# -------------------------------
# def delete_from_db(image_id):
#     """Deletes an image entry from the database based on its ID."""
#     conn = sqlite3.connect('image_database.db')
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM Images WHERE id = ?", (image_id,))
#     conn.commit()
#     conn.close()
#     print(f"Image with id {image_id} has been deleted.")

# def add_all_images_to_db():
#     """Extracts and adds all images from a directory to the database."""
#     extract_features_from_folder("images/texture", "texture")

# -------------------------------
# Example Usage (Uncomment for Testing)
# -------------------------------
# images = query_image_data("static/images/form/00.jpg")
# print(images)
# if images:
#     show_images_in_figure(images, rows=5, cols=6)
# else:
#     print("No valid images found in the provided paths.")
