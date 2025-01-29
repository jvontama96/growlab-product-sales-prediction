import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

# Load banner image
banner_image = Image.open("image.png")

# Define the MLPModel class
class MLPModel(torch.nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(14, 64)  # Input size 14, Output size 64
        self.fc2 = torch.nn.Linear(64, 1)   # Output size 1 (for sales prediction)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
# Load the trained model
model = MLPModel()
try:
    model.load_state_dict(torch.load("sales_prediction_model.pth"))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load your dataset
data = pd.read_csv("final.csv")

# Extract unique Lokasi and Category values
lokasi_options = data["Lokasi"].unique().tolist()
categories = data["Category"].unique().tolist()

# Streamlit app layout
st.image(banner_image, use_container_width=True)  # Display banner image

# Create tabs
tab1, tab2 = st.tabs(["Sales Prediction", "About"])

# Sales Prediction Tab
with tab1:
    st.info("""This app is designed to help Shopee online store user predict future sales based on key product attributes such as price, stock quantity, rating, location, and category.
    #### How It Works:
    1. **Input Parameters**: Provide the required details about your product, including price, stock quantity, rating, location, and category.
    2. **Predict Sales**: Click the **Predict Sales** button to get the predicted sales quantity and value for the next 30 days.
    """)
    
    # Input fields in the main layout
    st.header("Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        harga = st.number_input("Price (Harga)", min_value=0, help="Enter the price of the product.")
        jumlah_stok = st.number_input("Stock Quantity (Jumlah Stok)", min_value=0, help="Enter the current stock quantity.")

    with col2:
        rating = st.number_input("Rating", min_value=0, max_value=5, help="Enter the product rating (0 to 5).")
        lokasi = st.selectbox("Location (Lokasi)", lokasi_options, help="Select the location of the product.")
        category = st.selectbox("Category", categories, help="Select the product category.")

    # Encode Lokasi and Category
    lokasi_category_filter = (data["Lokasi"] == lokasi) & (data["Category"] == category)
    median_harga = data.loc[lokasi_category_filter, "Harga"].median()
    if pd.isna(median_harga):
        median_harga = harga
    harga_encoded = harga / median_harga if median_harga > 0 else harga

    # Ensure the input data has the correct number of features (14)
    category_encoded = np.zeros(len(categories))  # Create a one-hot encoding for categories

    # Filter data for the selected category
    category_data = data[data["Category"] == category]

    # Normalize harga values within the selected category
    if not category_data.empty:
        min_harga = category_data["Harga"].min()
        max_harga = category_data["Harga"].max()
        if max_harga != min_harga:
            normalized_harga = (harga - min_harga) / (max_harga - min_harga)
        else:
            normalized_harga = 1.0  # If all harga values are the same, default to 1
    else:
        normalized_harga = 1.0  # If no data for the category, default to 1

    # Set the encoded value based on normalized harga
    category_encoded[categories.index(category)] = normalized_harga

    # Combine all inputs into a single array of 14 features
    input_data = np.array([harga_encoded, jumlah_stok, rating] + list(category_encoded[:11]))
    if len(input_data) != 14:
        st.error("Input feature size does not match the model's requirements. Please check your data.")

    # Convert to tensor and predict sales
    if st.button("Predict Sales"):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Convert to tensor without scaling
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Calculate predicted sales value
        predicted_sales_quantity = prediction.item()
        predicted_sales_value = predicted_sales_quantity * harga

        # Display results in an emphasized layout
        st.success("### Prediction Results")
        st.markdown(f"**Predicted Sales Quantity for the Next 30 Days:** `{predicted_sales_quantity:.0f}` units")
        st.markdown(f"**Predicted Sales Value for the Next 30 Days:** `Rp {predicted_sales_value:,.2f}`")
        st.markdown(f"**Stock Adjustment Recommendation:** `{jumlah_stok - predicted_sales_quantity:.0f}` units")


# About Tab
with tab2:
    st.title("About This App")
    st.markdown("""
        #### Key Features:
        - **Sales Prediction**: Predicts the number of units likely to be sold in the next 30 days.
        - **Sales Value Calculation**: Calculates the total sales value based on the predicted quantity and product price.
        - **Stock Adjustment**: Provides recommendations for adjusting stock levels based on predicted sales.

        #### Why Use This App?
        - **Data-Driven Decisions**: Make informed decisions about inventory management and pricing strategies.
        - **User-Friendly Interface**: Simple and intuitive design for easy navigation.

        #### About the Model:
        The app uses a **Multi-Layer Perceptron (MLP)** model trained on historical sales data. The model takes into account various factors such as price, stock, rating, location, and category to provide accurate sales predictions.

        For full documentation, please visit [https://github.com/jvontama96/growlab-product-sales-prediction](https://github.com/jvontama96/growlab-product-sales-prediction).
    """)
