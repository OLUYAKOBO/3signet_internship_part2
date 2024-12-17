import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd

# Load the trained model
model = load_model('./week12/best_model.keras') 
#to run this locally the code should be model = load_model('best_model.keras')

# Streamlit app title
st.title("Image Classification Application")
st.write("Choose an Image and get the classification along with confidence scores.")

# Dictionary of class names
class_names = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver", 5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly", 15: "camel", 16: "can", 17: "castle", 18: "caterpillar",
    19: "cattle", 20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach", 25: "couch", 26: "crab",
    27: "crocodile", 28: "cup", 29: "dinosaur", 30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox", 35: "girl",
    36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard", 40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain", 50: "mouse", 51: "mushroom", 52: "oak_tree",
    53: "orange", 54: "orchid", 55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum", 65: "rabbit", 66: "raccoon", 67: "ray", 68: "road",
    69: "rocket", 70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew", 75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake",
    79: "spider", 80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
    85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor", 90: "train", 91: "trout",
    92: "tulip", 93: "turtle", 94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm"
}

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Get Image Class"):
        # Preprocess the image
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0) 

        # Get predictions
        predictions = model.predict(image)
        confidence_scores = predictions[0]
        predicted_index = np.argmax(confidence_scores)
        predicted_class = class_names[predicted_index]
        
        # Display the classification result
        st.write(f"The Image displayed is that of A/An **{predicted_class.capitalize()}**")
        st.write("### Confidence Scores for Each Class:")
        
        # Display confidence scores in a table
        confidence_df = pd.DataFrame({
            "Class": [class_names[i] for i in range(len(confidence_scores))],
            "Confidence Score": confidence_scores
        }).sort_values(by="Confidence Score", ascending=False).head(10) # top 10 class

        st.dataframe(confidence_df)
