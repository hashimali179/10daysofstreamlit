import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

st.title('Sample App')
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

data = {
    'Age': [25, 30, 40, 22, 35],
    'Income': [50000, 60000, 75000, 45000, 80000],
    'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single'],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
le_education = LabelEncoder()
le_marital_status = LabelEncoder()
le_city = LabelEncoder()
le_gender = LabelEncoder()

# Create a list of all possible categories for each categorical feature
education_categories = df['Education'].unique()
marital_status_categories = df['Marital_Status'].unique()
city_categories = df['City'].unique()
gender_categories = df['Gender'].unique()

# Fit the label encoders on all possible categories
le_education.fit(education_categories)
le_marital_status.fit(marital_status_categories)
le_city.fit(city_categories)
le_gender.fit(gender_categories)

# Map the categories to integer values in the DataFrame
df['Education'] = le_education.transform(df['Education'])
df['Marital_Status'] = le_marital_status.transform(df['Marital_Status'])
df['City'] = le_city.transform(df['City'])
df['Gender'] = le_gender.transform(df['Gender'])

# Split the dataset into features (X) and target variable (y)
X = df.drop('Age', axis=1)
y = df['Age']

# Create a Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the data
rf_model.fit(X, y)

# Sidebar 

st.sidebar.header('User Input Parameters')


def user_input_features():
    Income = st.sidebar.number_input("Enter Income", min_value=0, max_value=100000, value=1, step=1)
    Education = st.sidebar.selectbox('Education', ["Master", "Bachelor", "PhD"])
    Marital_Status = st.sidebar.selectbox('Marital_Status', ["Married", "Single"])
    City = st.sidebar.selectbox('City', ["New York", "Los Angeles", "Chicago", "San Francisco", "Seattle"])
    Gender = st.sidebar.selectbox('Gender', ["Male", "Female"])
    data = {'Income': Income,
            'Education': Education,
            'Marital_Status': Marital_Status,
            'City': City,
            'Gender': Gender,
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.header('User Input')
st.write(input_df)


# Map the categories of sample input to integer values using the fitted label encoders
input_df['Education'] = le_education.transform(input_df['Education'])
input_df['Marital_Status'] = le_marital_status.transform(input_df['Marital_Status'])
input_df['City'] = le_city.transform(input_df['City'])
input_df['Gender'] = le_gender.transform(input_df['Gender'])

encoded_input_df = pd.DataFrame(input_df)

st.header('Encoded User Input')
st.write(encoded_input_df)

predicted_age = rf_model.predict(input_df)
if st.button('click to predict'):
    st.write(predicted_age)