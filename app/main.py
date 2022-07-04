###############################
# this program lets you       #
# - enter values in Streamlit #
# - get predictios            #
###############################

import joblib
import pandas as pd
import numpy as np
import streamlit as st

df_clean = pd.read_csv('../data/clean_data.csv')

# Loading model
path = '../models/'

# Linear Regression (Clasic)
LinearR_path = path + 'linearRegression.pkl'
linearR_model = joblib.load(LinearR_path)

# RandomForest
randomF_path = path + 'randomForest.pkl'
randomF_model = joblib.load(randomF_path)

# Lasso Regression
lassoR_path = path + 'lassoRegression.pkl'
lassoR_model = joblib.load(lassoR_path)

# Loading Scaler
path = '../scalers/'
scaler_path = path + 'min_max.pkl'
minMax = joblib.load(scaler_path)

#############
# Main Page #
#############
st.write("The Models for Houses Price Prediction in Washington State")


# Get input values - numeric variables
bedrooms = st.number_input("Enter bedrooms numbers: ")
bathrooms = st.number_input("Enter bathrooms numbers: ")
sqft_living = st.number_input("Enter in square foot for living space: ")
sqft_lot = st.number_input("Enter in squared foot for lot space: ")
floors = st.number_input("Enter the floors numbers: ")
waterfront = st.selectbox('Enter if the house have waterfront or not: ', [0, 1])
view = st.number_input("Enter views numbers: ")

condition_options = pd.unique(df_clean['condition'])
condition_options.sort()
condition = st.selectbox("Select in list of how good condition 1 (worn-out) and 5 (excellent): ", 
                        options=condition_options)

grade_options = pd.unique(df_clean['grade'])
grade_options.sort()
grade = st.selectbox("Overall grade given to the housing unit, based on King County grading system. 1 poor,13 excellent: ",
                    grade_options)

sqft_above = st.number_input("Enter in square foot for above space: ")
sqft_basement = st.number_input("Enter in square foot for basement space: ")
lat = st.number_input("Location for latitude: ")
long = st.number_input("Location for longitude: ")

sqft_living15 = sqft_living
sqft_lot15 = sqft_lot

# scaler_selection = st.selectbox("Select of type scaler: ", ['Min Max Scaler', 'Yeo Johnson'])

age_dict = {
    '(0, 10.0]': 0,
    '(10.0, 21.0]': 1,
    '(21.0, 35.0]': 2,
    '(35.0, 47.0]': 3,
    '(47.0, 60.0]': 4,
    '(60.0, 76.0]': 5,
    '(76.0, 115.0]': 6
    }
age_bins_options = ["{}".format(key) for key, item in age_dict.items()]
age_bins = st.selectbox("Selection of age range of house: ", age_bins_options)

renovated_age = {
    '(0, 16.0]': 0,
    '(16.0, 32.0]': 1,
    '(32.0, 48.0]': 2,
    '(48.0, 64.0]': 3,
    '(64.0, 80.0]': 4
    }

renovated_bins_options = ["{}".format(key) for key, item in renovated_age.items()]
renovated_bins = st.selectbox("Selection age of renovation for the house: ", renovated_bins_options)

if st.button("Get your prediction"):

    X = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'grade': [grade],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'lat': [lat],
        'long': [long],
        'sqft_living15': [sqft_living15],
        'sqft_lot15': [sqft_lot15],
        'age_bins': [age_bins],
        'renovated_bins': [renovated_bins]
    })

    X['age_bins'] = X['age_bins'].replace(age_dict)
    X['renovated_bins'] = X['renovated_bins'].replace(renovated_age)

    # # Scaling Data:
    X_scaled = minMax.transform(X)

    # # Making Prediction:
    linearR_prediction = linearR_model.predict(X_scaled)
    randomF_prediction = randomF_model.predict(X_scaled)
    lassoR_prediction = lassoR_model.predict(X_scaled)

    st.success("The Price predict through Linear Regression is: {}".format(round(np.expm1(linearR_prediction[0]), 2)))
    st.success("The Price predict through Random Forest Regression is: {}".format(round(np.expm1(randomF_prediction[0]), 2)))
    st.success("The Price predict through Lasso Regression is: {}".format(round(np.expm1(lassoR_prediction[0]), 2)))