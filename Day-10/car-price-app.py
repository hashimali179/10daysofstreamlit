import streamlit as st
# import numpy as np
import pandas as pd
import os
# import pickle
from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import LabelEncoder


# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

st.title('Used Car Price Prediction App')
st.markdown("""
This app predicts used car prices on the basis of historical data from **CarDekho**!

""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:**numpy, pandas, streamlit, scikit-learn, matplotlib, seaborn, Pickle,
* **Data source:** [CarDekho](http://cardekho.com).
* **Rebuiling on streamlit:** This web appliction was already built on django framwork for my college project, now that I am learning streamlit, I have built it using streamlit library.
""")

@st.cache(allow_output_mutation=True)
def load_data():
    # Construct the file path to penguins_cleaned.csv
    file_path = os.path.join(os.path.dirname(__file__), 'final_dataset.csv')
    return pd.read_csv(file_path)

final_dataset = load_data()
final_dataset = final_dataset.drop(columns=['Unnamed: 0'])



# Sidebar 

st.sidebar.header('User Input Parameters')

def user_input_features():
    no_year =  st.sidebar.number_input("Enter the age of the car", min_value=1, value=5, step=1)
    Owner = st.sidebar.number_input("Enter the number of previous owners", min_value=0, max_value=3, value=1, step=1)
    Present_Price = st.sidebar.number_input("Enter the present price of the car(in lakhs)", min_value=1, max_value=92, value=5, step = 1)
    Kms_Driven = st.sidebar.number_input("Enter the kilometers driven by the car", min_value=500, max_value=500000, value=100000, step=100)
    Fuel_Type = st.sidebar.selectbox('Fuel_Type', ["Petrol", "Diesel", "CNG"])
    Seller_Type = st.sidebar.selectbox('Seller_Type', ["Dealer", "Individual"])
    Transmission = st.sidebar.selectbox('Transmission', ["Automatic", "Manual"])

    data = {'Present_Price': Present_Price,
            'Kms_Driven': Kms_Driven,
            'Fuel_Type': Fuel_Type,
            'Seller_Type': Seller_Type,
            'Transmission': Transmission,
            'Owner': Owner,
            'no_year': no_year,           
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


# Displays the user input features
st.subheader('User Input parameters')
st.write(input_df)


# # Encode categorical variables
# le_fuel_type = LabelEncoder()
# le_seller_type = LabelEncoder()
# le_transmission = LabelEncoder()

# # Create a list of all possible categories for each categorical feature
# fuel_type_categories = final_dataset['Fuel_Type'].unique()
# seller_type_categories = final_dataset['Seller_Type'].unique()
# transmission_categories = final_dataset['Transmission'].unique()

# # Fit the label encoders on all possible categories
# le_fuel_type = le_fuel_type.fit(fuel_type_categories)
# le_seller_type = le_seller_type.fit(seller_type_categories)
# le_transmission = le_transmission.fit(transmission_categories)

# # Map the categories to integer values in the DataFrame
# final_dataset['Fuel_Type'] = le_fuel_type.transform(final_dataset['Fuel_Type'])
# final_dataset['Seller_Type'] = le_seller_type.transform(final_dataset['Seller_Type'])
# final_dataset['Transmission'] = le_transmission.transform(final_dataset['Transmission'])

fuel_type_label = {"Petrol":0, "Diesel":1, "CNG":2}
seller_type_label = {"Individual":1, "Dealer":0}
transmission_label = {"Automatic":0,"Manual":1}

final_dataset['Fuel_Type'] = final_dataset['Fuel_Type'].map(fuel_type_label)
final_dataset['Seller_Type'] = final_dataset['Seller_Type'].map(seller_type_label)
final_dataset['Transmission'] = final_dataset['Transmission'].map(transmission_label)

# Split the dataset into features (X) and target variable (y)
X = final_dataset.drop('Selling_Price', axis=1)
y = final_dataset['Selling_Price']

# Create a Random Forest Regression model
rf_model = DecisionTreeRegressor(random_state=42)

# Train the model on the data
rf_model.fit(X, y)

fuel_type_label = {"Petrol":0, "Diesel":1, "CNG":2}
input_df['Fuel_Type']=input_df['Fuel_Type'].map(fuel_type_label)
input_df['Seller_Type']=input_df['Seller_Type'].map(seller_type_label)
input_df['Transmission']=input_df['Transmission'].map(transmission_label)

# encoded_input_df = pd.DataFrame(input_df)
# Displays the encoded user input features
st.subheader('Encoded User Input parameters')
st.write(input_df)



prediction = rf_model.predict(input_df)
if st.button:
    st.write(prediction)