import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
from keras import applications, models
import matplotlib.pyplot as plt
import os
import json

# Function to load the pre-trained VGG16 model with weights
def get_vgg16_model():
    base_model = applications.vgg16.VGG16(weights=None, include_top=True)  
    model = models.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    return model

# Load the VGG16 model
vgg_model = get_vgg16_model()

# Function to extract HOG features from an image
def extract_hog_features(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    image = cv2.resize(image, (128, 128))  # Resize the image
    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          block_norm='L2-Hys', visualize=True)
    return hog_features

# Function to extract VGG16 features from an image
def extract_vgg_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Required size for VGG16
    image = applications.vgg16.preprocess_input(np.expand_dims(image, axis=0))
    features = vgg_model.predict(image)
    return features.flatten()

# Function to build and save an image database
def build_image_database(folder_path, save_path='test.json'):
    database = []
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(subdir, file)
                hog_features = extract_hog_features(image_path).tolist()
                vgg_features = extract_vgg_features(image_path).tolist()
                database.append({'image_path': image_path, 'hog_features': hog_features, 'vgg_features': vgg_features})
    
    with open(save_path, 'w') as json_file:
        json.dump(database, json_file)
    print(f"Database built and saved to {save_path}")
    return database

# Function to load an image database from JSON
def load_image_database(file_path='test.json'):
    with open(file_path, 'r') as json_file:
        database = json.load(json_file)
    for entry in database:
        entry['hog_features'] = np.array(entry['hog_features'])
        entry['vgg_features'] = np.array(entry['vgg_features'])
    print(f"Database loaded from {file_path}")
    return database

# Function to compute cosine similarity
def calculate_similarity_2(query_features, database_features):
    return cosine_similarity([query_features], [database_features])[0][0]

# Function to search for similar images
def search_similar_images(query_image_path, database, top_n=80):
    query_hog_features = extract_hog_features(query_image_path)
    query_vgg_features = extract_vgg_features(query_image_path)

    similarities = []
    for entry in database:
        image_path = entry['image_path']
        hog_features = entry['hog_features']
        vgg_features = entry['vgg_features']

        hog_similarity = calculate_similarity_2(query_hog_features, hog_features)
        vgg_similarity = calculate_similarity_2(query_vgg_features, vgg_features)
        combined_similarity = 0.5 * hog_similarity + 0.5 * vgg_similarity
        similarities.append((image_path, combined_similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Function to load and resize images
def load_and_resize_images_from_paths(results):
    images = []
    for path, similarity in results:
        img = cv2.imread(path)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    return images

# Function to display images in a grid
def show_images_in_figure(images, rows=5, cols=6):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
