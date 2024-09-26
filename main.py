apt-get install tesseract-ocr
pip install pytesseract
import os
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import requests
from sklearn.preprocessing import LabelEncoder
import re
from tqdm import tqdm 

# dataset 
try:
    train_data = pd.read_csv('/content/drive/MyDrive/train.csv').head(1000).reset_index(drop=True)
    test_data = pd.read_csv('/content/drive/MyDrive/test.csv').head(500).reset_index(drop=True)
except Exception as e:
    raise RuntimeError(f"Error loading dataset: {e}")

# image feature extraction
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
except Exception as e:
    raise RuntimeError(f"Error initializing ResNet50 model: {e}")

# Entity unit map
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

#extract value and unit
def extractvalueandunit(text, entity):
    pattern = r'([+-]?\d+(?:\.\d+)?)\s{0,1}(cm|ft|in|m|mm|yard|g|kg|ug|mg|oz|pd|ton|kv|mv|v|kw|w|cl|ft\^3|in\^3|c|dl|floz|gal|imp gal|L|ul|ml|pt|qt)'
    matches = re.findall(pattern, text)
    
    if matches:
        for match in matches:
            value, unit = match
            unit = unit.strip().lower()
            if unit in entity_unit_map.get(entity, {}):
                return value, unit
    return None, None

# download and preprocess image
def preprocessimage(url):
    try:
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)
        img = img.resize((224, 224))  # Resizing the image to 224x224 for ResNet input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

#extract image features using ResNet50
def extract_image_features(img_array):
    try:
        features = resnet_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

#extract text from image using Tesseract OCR
def extract_text_from_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        img = Image.open(response.raw)
        img_cv = np.array(img)
        text = pytesseract.image_to_string(img_cv)
        return text
    except Exception as e:
        print(f"Error extracting text from {image_url}: {e}")
        return ""

# process text and extract value and unit
def process_text_for_prediction(text, entity):
    value, unit = extractvalueandunit(text, entity)
    if value and unit:
        return f"{value} {unit}"
    return ""

# Batch processing function 
def batchdatapreparing(data, batch_size=250):
    total_batches = len(data) // batch_size
    features = []
    labels = []
    for batch_num in tqdm(range(total_batches), desc="Processing training batches"):
        batch_data = data[batch_num * batch_size:(batch_num + 1) * batch_size]
        for _, row in batch_data.iterrows():
            image_url = row['image_link']
            entity_value = row['entity_value']
            entity_type = row['entity_name']

            img_array = preprocessimage(image_url)
            if img_array is None:
                continue

            img_features = extract_image_features(img_array)
            if img_features is None:
                continue

            text = extract_text_from_image(image_url)
            processed_text = process_text_for_prediction(text, entity_type)

            combined_features = np.hstack([img_features, len(processed_text)])
            features.append(combined_features)
            labels.append(entity_value)

        print(f"Completed batch {batch_num + 1} of {total_batches}")
    
    return np.array(features), np.array(labels)

try:
    X, y = batchdatapreparing(train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
except Exception as e:
    raise RuntimeError(f"Error preparing training data: {e}")

# Train XGBoost model
try:
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    xgb_model.fit(X_train, y_train_encoded)
except Exception as e:
    raise RuntimeError(f"Error training XGBoost model: {e}")

# Batch processing for test data with predictions
def predictdata(batch_size=250):
    output_file = '/content/drive/MyDrive/test_out.csv'
    
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['index', 'prediction']).to_csv(output_file, mode='w', header=True, index=False)

    total_batches = len(test_data) // batch_size
    predictions = []
    
    for batch_num in tqdm(range(total_batches), desc="Processing test batches"):
        batch_data = test_data[batch_num * batch_size:(batch_num + 1) * batch_size]
        for _, row in batch_data.iterrows():
            index = row['index']
            image_url = row['image_link']
            entity_type = row['entity_name']

            img_array = preprocessimage(image_url)
            if img_array is None:
                predictions.append({'index': index, 'prediction': ''})
                continue

            img_features = extract_image_features(img_array)
            if img_features is None:
                predictions.append({'index': index, 'prediction': ''})
                continue

            text = extract_text_from_image(image_url)
            processed_text = process_text_for_prediction(text, entity_type)

            combined_features = np.hstack([img_features, len(processed_text)])
            prediction = xgb_model.predict([combined_features])[0]

            predicted_value = le.inverse_transform([int(prediction)])[0]
            predicted_text = f"{predicted_value} {processed_text}"
            predictions.append({'index': index, 'prediction': predicted_text})

        pd.DataFrame(predictions).to_csv(output_file, mode='a', header=False, index=False)
        predictions = [] 
        print(f"Completed batch {batch_num + 1} of {total_batches}")

predictdata()
