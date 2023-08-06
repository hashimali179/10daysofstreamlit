import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import joblib
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

st.title('Bank Loan Defaulter Classifier App')
st.markdown("""
This app predicts weather a person will be defaulter or not on the basis of Historical Data!

""")
image = os.path.join(os.path.dirname(__file__), 'loandefaulter.jpg')
st.image(image, use_column_width=True)
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** numpy, pandas, streamlit, scikit-learn, xgboost
* **Rebuiling on streamlit:** This was my EDA project. I am expanding this project into a full stack data science project using streamlit.
""")

# Construct the file path to xgboost_model.pkl
pickle_file_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')
xgb_model = joblib.load(pickle_file_path)

# Construct the file path to final_df.csv
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'final_df.csv')
    return pd.read_csv(file_path)

final_dataset = load_data()
final_dataset = final_dataset.drop(columns=['DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH'])
# Sidebar 

st.sidebar.header('User Input Parameters')

def user_input_features():
    NAME_CONTRACT_TYPE =  st.sidebar.selectbox('NAME_CONTRACT_TYPE', ["Cash loans", "Revolving loans"])
    CODE_GENDER = st.sidebar.selectbox('CODE_GENDER', ["Male", "Female"])
    FLAG_OWN_CAR = st.sidebar.selectbox('FLAG_OWN_CAR', ["Yes", "No"])
    FLAG_OWN_REALTY = st.sidebar.selectbox('FLAG_OWN_REALTY', ["Yes", "No"])
    CNT_CHILDREN = st.sidebar.number_input("Enter the number of children", min_value=0, max_value=20, value=0, step=1)
    AMT_INCOME_TOTAL = st.sidebar.number_input("Enter the annual income", min_value=25000, max_value=200000000, value=25000, step=5000)
    NAME_INCOME_TYPE = st.sidebar.selectbox('NAME_INCOME_TYPE', ["Working", "State servant","Commercial associate","Student","Businessman","Pensioner","Maternity leave"])
    NAME_EDUCATION_TYPE = st.sidebar.selectbox('NAME_EDUCATION_TYPE', ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    NAME_FAMILY_STATUS = st.sidebar.selectbox('NAME_FAMILY_STATUS', ["Single / not married", "Married", "Civil marriage", "Widow", "Separated"])
    NAME_HOUSING_TYPE =  st.sidebar.selectbox('NAME_HOUSING_TYPE', ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"])
    OCCUPATION_TYPE = st.sidebar.selectbox('OCCUPATION_TYPE', ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
                                            "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff",
                                            "Security staff", 'High skill tech staff', 'Waiters/barmen staff',
                                            "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff"])
    CNT_FAM_MEMBERS = st.sidebar.number_input("Enter the number of family members", min_value=0, max_value=20, value=0, step=1)
    LIVE_CITY_NOT_WORK_CITY = st.sidebar.selectbox('LIVE_CITY_NOT_WORK_CITY', ["Yes", "No"])
    FLAG_DOCUMENT_6 = st.sidebar.selectbox('FLAG_DOCUMENT_6', ["Yes", "No"])
    FLAG_DOCUMENT_8 = st.sidebar.selectbox('FLAG_DOCUMENT_8', ["Yes", "No"])

    data = {
            'NAME_CONTRACT_TYPE':NAME_CONTRACT_TYPE,
            'CODE_GENDER':CODE_GENDER,
            'FLAG_OWN_CAR':FLAG_OWN_CAR,
            'FLAG_OWN_REALTY':FLAG_OWN_REALTY,
            'CNT_CHILDREN':CNT_CHILDREN,
            'AMT_INCOME_TOTAL':AMT_INCOME_TOTAL,
            'NAME_INCOME_TYPE':NAME_INCOME_TYPE,
            'NAME_EDUCATION_TYPE':NAME_EDUCATION_TYPE,
            'NAME_FAMILY_STATUS':NAME_FAMILY_STATUS,
            'NAME_HOUSING_TYPE':NAME_HOUSING_TYPE,
            'OCCUPATION_TYPE':OCCUPATION_TYPE,
            'CNT_FAM_MEMBERS':CNT_FAM_MEMBERS,
            'LIVE_CITY_NOT_WORK_CITY':LIVE_CITY_NOT_WORK_CITY,
            'FLAG_DOCUMENT_6':FLAG_DOCUMENT_6,
            'FLAG_DOCUMENT_8':FLAG_DOCUMENT_8,           
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


# Displays the user input features
st.subheader('User Input parameters')
if st.button('Show Input DataFrame'):
    st.write(input_df)



# Labelling DATASET
NAME_CONTRACT_TYPE_LABEL = {"Cash loans":0, "Revolving loans":1}
CODE_GENDER_LABEL = {"M":0, "F":1}
FLAG_OWN_CAR_LABEL = {"Y":1, "N":0}
FLAG_OWN_REALTY_LABEL = {"Y":1, "N":0}
NAME_INCOME_TYPE_LABEL = {"Working":0, "State servant":1,"Commercial associate":2,"Student":3,"Businessman":4,"Pensioner":5,"Maternity leave":6}
NAME_EDUCATION_TYPE_LABEL = {"Secondary / secondary special":0, "Higher education":1, "Incomplete higher":2, "Lower secondary":3, "Academic degree":4}
NAME_FAMILY_STATUS_LABEL = {"Single / not married":0, "Married":1, "Civil marriage":2, "Widow":3, "Separated":4}
NAME_HOUSING_TYPE_LABEL = {"House / apartment":0, "Rented apartment":1, "With parents":2, "Municipal apartment":3, "Office apartment":4, "Co-op apartment":5}
OCCUPATION_TYPE_LABEL = {"Laborers":0, "Core staff":1, "Accountants":2, "Managers":3, "Drivers":4, "Sales staff":5,
                                            "Cleaning staff":6, "Cooking staff":7, "Private service staff":8, "Medicine staff":9,
                                            "Security staff":10, 'High skill tech staff':11, 'Waiters/barmen staff':12,
                                            "Low-skill Laborers":13, "Realty agents":14, "Secretaries":15, "IT staff":16, "HR staff":17}
# Labelling INPUT DATA
CODE_GENDER_INPUT_LABEL = {"Male":0, "Female":1}
FLAG_OWN_CAR_INPUT_LABEL = {"Yes":1, "No":0}
FLAG_OWN_REALTY_INPUT_LABEL = {"Yes":1, "No":0}
LIVE_CITY_NOT_WORK_CITY_LABEL = {"Yes":1, "No":0}
FLAG_DOCUMENT_6_LABEL = {"Yes":1, "No":0}
FLAG_DOCUMENT_8_LABEL = {"Yes":1, "No":0}

# Mapping DATASET
final_dataset["NAME_CONTRACT_TYPE"] = final_dataset["NAME_CONTRACT_TYPE"].map(NAME_CONTRACT_TYPE_LABEL)
final_dataset["CODE_GENDER"] = final_dataset["CODE_GENDER"].map(CODE_GENDER_LABEL)
final_dataset["FLAG_OWN_CAR"] = final_dataset["FLAG_OWN_CAR"].map(FLAG_OWN_CAR_LABEL)
final_dataset["FLAG_OWN_REALTY"] = final_dataset["FLAG_OWN_REALTY"].map(FLAG_OWN_REALTY_LABEL)
final_dataset["NAME_INCOME_TYPE"] = final_dataset["NAME_INCOME_TYPE"].map(NAME_INCOME_TYPE_LABEL)
final_dataset["NAME_EDUCATION_TYPE"] = final_dataset["NAME_EDUCATION_TYPE"].map(NAME_EDUCATION_TYPE_LABEL)
final_dataset["NAME_FAMILY_STATUS"] = final_dataset["NAME_FAMILY_STATUS"].map(NAME_FAMILY_STATUS_LABEL)
final_dataset["NAME_HOUSING_TYPE"] = final_dataset["NAME_HOUSING_TYPE"].map(NAME_HOUSING_TYPE_LABEL)
final_dataset["OCCUPATION_TYPE"] = final_dataset["OCCUPATION_TYPE"].map(OCCUPATION_TYPE_LABEL)
# Mapping INPUT DATA
input_df["NAME_CONTRACT_TYPE"] = input_df["NAME_CONTRACT_TYPE"].map(NAME_CONTRACT_TYPE_LABEL)
input_df["CODE_GENDER"] = input_df["CODE_GENDER"].map(CODE_GENDER_INPUT_LABEL)
input_df["FLAG_OWN_CAR"] = input_df["FLAG_OWN_CAR"].map(FLAG_OWN_CAR_INPUT_LABEL)
input_df["FLAG_OWN_REALTY"] = input_df["FLAG_OWN_REALTY"].map(FLAG_OWN_REALTY_INPUT_LABEL)
input_df["NAME_INCOME_TYPE"] = input_df["NAME_INCOME_TYPE"].map(NAME_INCOME_TYPE_LABEL)
input_df["NAME_EDUCATION_TYPE"] = input_df["NAME_EDUCATION_TYPE"].map(NAME_EDUCATION_TYPE_LABEL)
input_df["NAME_FAMILY_STATUS"] = input_df["NAME_FAMILY_STATUS"].map(NAME_FAMILY_STATUS_LABEL)
input_df["NAME_HOUSING_TYPE"] = input_df["NAME_HOUSING_TYPE"].map(NAME_HOUSING_TYPE_LABEL)
input_df["OCCUPATION_TYPE"] = input_df["OCCUPATION_TYPE"].map(OCCUPATION_TYPE_LABEL)
input_df["LIVE_CITY_NOT_WORK_CITY"] = input_df["LIVE_CITY_NOT_WORK_CITY"].map(LIVE_CITY_NOT_WORK_CITY_LABEL)
input_df["FLAG_DOCUMENT_6"] = input_df["FLAG_DOCUMENT_6"].map(FLAG_DOCUMENT_6_LABEL)
input_df["FLAG_DOCUMENT_8"] = input_df["FLAG_DOCUMENT_8"].map(FLAG_DOCUMENT_8_LABEL)



X = final_dataset.drop("TARGET", axis = 1)
y = final_dataset["TARGET"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and fit the XGBoost Classifier model
# xgb_model = XGBClassifier(random_state=43)
# xgb_model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.subheader('Accuracy')
st.write("Accuracy of the model is :",accuracy)

prediction = xgb_model.predict(input_df)
prediction_proba = xgb_model.predict_proba(input_df)


st.subheader('Prediction')
loan_tpye = np.array(['Non-Defaulter','Defaulter'])
st.write(loan_tpye[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns
# Plot the Feature Importance graph
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Subplot 1: Feature Importance
ax[0].barh(feature_names, feature_importance)
ax[0].set_xlabel('Feature Importance')
ax[0].set_ylabel('Features')
ax[0].set_title('XGBoost Feature Importance')

# Subplot 2: Accuracy Score
colors = ['limegreen', 'lightgray']
explode = [0.1, 0]
ax[1].pie([accuracy, 1 - accuracy], labels=['Accuracy', ''], colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
ax[1].set_title('Accuracy Score')

plt.tight_layout()
st.pyplot(fig)



if st.button("Show Encoded  User Input"):
    st.write(input_df)