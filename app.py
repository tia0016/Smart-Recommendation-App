import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Load dataset
df = pd.read_csv("RetailSaleDataset.csv")

# Pivot table: Customer vs Product Category
pivot_table = df.pivot_table(
    index='Customer ID',
    columns='Product Category',
    values='Total Amount',
    aggfunc='sum',
    fill_value=0
)

# Fit Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(pivot_table)

# Recommendation function
def get_recommendations(customer_id, pivot_df, model, n_neighbors=6):
    if customer_id not in pivot_df.index:
        return None, None

    customer_index = pivot_df.index.tolist().index(customer_id)
    distances, indices = model.kneighbors(pivot_df.iloc[customer_index, :].values.reshape(1, -1), n_neighbors=n_neighbors)

    target_customer = pivot_df.iloc[customer_index]
    similar_customers = indices.flatten()[1:]
    similar_data = pivot_df.iloc[similar_customers]
    mean_purchases = similar_data.mean()
    recommendations = mean_purchases[target_customer == 0].sort_values(ascending=False)

    return target_customer[target_customer > 0], recommendations


# Streamlit UI
st.title("Smart Product Recommendation App")
st.write("Enter a customer ID to get personalized product recommendations.")

# Input: Customer ID
customer_id = st.text_input("Enter Customer ID (e.g., CUST783)", key="customer_input")

if customer_id:
    bought, recommended = get_recommendations(customer_id, pivot_table, model)

    if bought is None:
        st.warning("Customer not found.")
    else:
        st.subheader("🛍️ Products already purchased:")
        st.dataframe(bought)

        st.subheader("✨ Recommended Categories:")
        if recommended.empty:
            st.info("No new categories to recommend.")
        else:
            st.dataframe(recommended)

# Upload image
st.subheader("🖼️ Upload Product Image (Optional)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Preferences section
st.subheader("🎯 Customize Your Preferences")

category = st.selectbox(
    "Choose Category",
    ["Beauty", "Fashion", "Electronics", "Home Decor"],
    key="category_select"
)

style = st.selectbox(
    "Style/Type",
    ["Minimalist", "Trendy", "Tech-Savvy", "Natural", "Luxury"],
    key="style_select"
)

budget = st.slider("Budget Range (₹)", 500, 50000, step=500, key="budget_slider")

# Button to generate recommendations
if st.button("🎁 Recommend Products", key="recommend_button"):
    st.success(f"Showing {category} recommendations in '{style}' style under ₹{budget}.")

    st.subheader("🛒 Recommended Products")

    # Dummy product data based on category
    product_map = {
        "Beauty": [
            {"name": "Trendy Lipstick Set", "price": 999, "rating": "⭐️⭐️⭐️⭐️"},
            {"name": "Skin Glow Serum", "price": 1499, "rating": "⭐️⭐️⭐️⭐️½"},
            {"name": "Compact Beauty Kit", "price": 2299, "rating": "⭐️⭐️⭐️"}
        ],
        "Fashion": [
            {"name": "Minimalist Denim Jacket", "price": 1999, "rating": "⭐️⭐️⭐️⭐️"},
            {"name": "Stylish Turtleneck Sweater", "price": 1799, "rating": "⭐️⭐️⭐️⭐️½"},
            {"name": "Classic White Sneakers", "price": 2599, "rating": "⭐️⭐️⭐️⭐️⭐️"}
        ],
        "Electronics": [
            {"name": "Wireless Earbuds", "price": 2999, "rating": "⭐️⭐️⭐️⭐️"},
            {"name": "Smart Fitness Band", "price": 3999, "rating": "⭐️⭐️⭐️⭐️½"},
            {"name": "Portable Bluetooth Speaker", "price": 1999, "rating": "⭐️⭐️⭐️"}
        ],
        "Home Decor": [
            {"name": "Decorative Wall Clock", "price": 1499, "rating": "⭐️⭐️⭐️⭐️"},
            {"name": "LED Fairy Light Set", "price": 799, "rating": "⭐️⭐️⭐️⭐️½"},
            {"name": "Handcrafted Vase", "price": 1199, "rating": "⭐️⭐️⭐️⭐️"}
        ]
    }

    # Filter products under budget
    filtered = [p for p in product_map[category] if p["price"] <= budget]

    # Show results
    if filtered:
        for p in filtered:
            st.markdown(f"**{p['name']}** — ₹{p['price']} | {p['rating']}")
    else:
        st.info("No products found under the selected budget.")
