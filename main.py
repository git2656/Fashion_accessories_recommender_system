import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import faiss # 👈 Import Faiss

# --- 1. Load Data & Build Faiss Index ---
# Load pre-computed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Build the Faiss index for fast similarity search
dimension = feature_list.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(feature_list) # Add all feature vectors to the index

# --- 2. Load the Model ---
# Define the feature extraction model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System 🛍️')

# --- 3. Define Helper Functions ---
def save_uploaded_file(uploaded_file):
    """Saves the uploaded image to the 'uploads' directory."""
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    """Extracts and normalizes features from an image using the ResNet50 model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, index):
    """Recommends similar items using the Faiss index."""
    # The .search method finds the k nearest neighbors
    # We search for 6 neighbors because the first one will be the query image itself
    distances, indices = index.search(np.array([features]), 6)
    return indices

# --- 4. Main Streamlit App Logic ---
# File uploader widget
uploaded_file = st.file_uploader("Choose an image to find similar items")
if uploaded_file is not None:
    # Save the uploaded file
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Your Upload", width=200)
        st.subheader("Here are some similar styles:")

        # Extract features from the uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        
        # Get recommendations using the Faiss index
        indices = recommend(features, index)
        
        # Display the recommended images in columns
        # Note: Using st.columns which is the current standard
        cols = st.columns(5)
        
        # We start from indices[0][1] to skip the most similar item (which is the item itself)
        with cols[0]:
            st.image(filenames[indices[0][1]])
        with cols[1]:
            st.image(filenames[indices[0][2]])
        with cols[2]:
            st.image(filenames[indices[0][3]])
        with cols[3]:
            st.image(filenames[indices[0][4]])
        with cols[4]:
            st.image(filenames[indices[0][5]])
    else:
        st.header("An error occurred during file upload.")
