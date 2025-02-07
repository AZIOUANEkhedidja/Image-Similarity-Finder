import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

class ImageClassifier:
    def __init__(self, data_path="static/images/form"):
        self.data_path = data_path
        self.class_mapping = {
            2: 'horse',     
            3: 'flowers',   
            4: 'cat',       
            5: 'cake',      
            6: 'baby',      
            7: 'nature'     
        }
        self.nature_keywords = {"00", "a", "aa", "aaa", "aze", "ba", "bb", "c", "cc", "d", "e", "f", "g", "h", 
                                "hh", "i", "k", "l", "ll", "m", "mm", "nn", "o", "oo", "p", "pp", "q", "r", "s", "ss", 
                                "t", "u", "v", "y", "yy", "z"}
        self.baby_keywords = {"000001", "000002"}.union({f"{i:06d}" for i in range(5, 45)})
        self.hors_keywords = {f"{i:02d}" for i in range(1,52)}
        self.cat_keywords = {f"{i:04d}" for i in range(1,32)}
        self.flowers_keywords = {f"{i:03d}" for i in range(1,44)}
        self.cake_keywords = {f"{i:05d}" for i in range(1,45)}
        self.image_histograms = {}
        self.train_model()  


    def get_class_from_filename(self, filename):
        filename_without_extension = os.path.splitext(filename)[0]
        if filename_without_extension in self.nature_keywords:
            return 'nature', filename
        elif filename_without_extension in self.cake_keywords:
            return 'cake', filename
        elif filename_without_extension in self.flowers_keywords:
            return 'flowers', filename
        elif filename_without_extension in self.cat_keywords:
            return 'cat', filename
        elif filename_without_extension in self.baby_keywords:
            return 'baby', filename
        elif filename_without_extension in self.hors_keywords:
            return 'horse', filename
        else:
            return None, None

    def extract_histogram_features(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        h_hist = cv2.calcHist([h_channel], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
        v_hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
        histogram_features = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
        return histogram_features

    def train_model(self):
        features = []
        labels = []
        for img_name in os.listdir(self.data_path):
            img_path = os.path.join(self.data_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Impossible de lire l'image : {img_path}")
                continue
            
            class_name, file_name = self.get_class_from_filename(img_name)
            if class_name is None:
                print(f"Class name could not be determined for image: {img_name}")
                continue
            
            class_index = list(self.class_mapping.values()).index(class_name) if class_name != 'nature' else len(self.class_mapping)
            histogram_features = self.extract_histogram_features(image)
            
            if histogram_features.size > 0:
                features.append(histogram_features.astype(np.float64))
                labels.append(class_index)
                if class_name not in self.image_histograms:
                    self.image_histograms[class_name] = []
                self.image_histograms[class_name].append((file_name, histogram_features.tolist()))  # Keep original filename
        
        features = np.array(features)
        labels = np.array(labels)
        self.kmeans = KMeans(n_clusters=len(self.class_mapping) + 1, random_state=42)
        self.kmeans.fit(features)

    def predict_class(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Impossible de lire l'image : {image_path}")
            return
        histogram_features = self.extract_histogram_features(image)
        predicted_class = self.kmeans.predict([histogram_features.astype(np.float64)])[0]
        print(f"La classe prédite pour l'image '{image_path}' est : {list(self.class_mapping.values())[predicted_class] if predicted_class < len(self.class_mapping) else 'nature'}")

        return predicted_class, histogram_features

    def find_similar_images(self, image_path, k=5):
        predicted_class, target_features = self.predict_class(image_path)
        distances = []
        similar_images = []
        for class_name, histograms in self.image_histograms.items():
            for original_filename, histogram in histograms:  # Unpack the original filename
                distance = np.linalg.norm(np.array(histogram) - np.array(target_features))  
                image_full_path = os.path.join(self.data_path, original_filename)  # Use the original filename
                similar_images.append((image_full_path, distance))

        similar_images.sort(key=lambda x: x[1])
        top_similar_images = similar_images[:k]
        list = []
        print(f"\nTop {k} similar images:")
        for i, (image_path, dist) in enumerate(top_similar_images):
            list.append(image_path.replace("\\", "/").replace("static/", ""))
            # print(f"{i + 1}. Path: {list} - Distance: {dist:.2f}")
        return list


# data_path = "static/images/form"  
# classifier = ImageClassifier(data_path)

# test_image_path = "static/images/form/000001.jpg"  
# list = classifier.find_similar_images(test_image_path)
# print(list)
