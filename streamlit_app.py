import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Resmi belirli bir genişlikle resim gösterme
st.image('TimeSpend.png', caption='Time Spending of Customers in a Restautrant', width=500)

st.header("Restaurant Bill Estimator")

dt_model = pickle.load(open('final_DT_model.pkl', "rb"))
xgb_model = pickle.load(open('final_XGB_model.pkl', 'rb')) 
ann_model = load_model('final_ANN_model.h5')
scaler = pickle.load(open('final_scaler_saved.pkl', "rb"))

st.sidebar.title("Please enter details about the bill.")

# Örnek özellik seçimleri
meal_type = st.sidebar.selectbox("Meal Type", ['Breakfast', 'Lunch', 'Dinner'])
table_location = st.sidebar.selectbox("Table Location", ['Window', 'Center', 'Patrio'])
weather_condition = st.sidebar.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
day = st.sidebar.selectbox("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
age_group = st.sidebar.selectbox("Age Group", ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
reservation = st.sidebar.radio("Reservation", [1, 0])
live_music =  st.sidebar.radio("Live Music", [1, 0])
number_of_people = st.sidebar.number_input("Number of People", min_value=1, value=1)
time_spent = st.sidebar.number_input("Time Spent (minutes)",  min_value=30)
gender = st.sidebar.radio("Gender", ['Male', 'Female', 'Other'])
customer_satisfaction = st.sidebar.slider('Customer Satisfaction', 1, 5, 3)

# Kullanıcının girdiği verileri bir sözlük olarak kaydetme
user_data = {
    "Number of People": number_of_people,
    "Meal Type": meal_type,
    "Table Location": table_location,
    "Weather Condition": weather_condition,
    "Day": day,
    "Time Spent (minutes)": time_spent,
    "Gender": gender,
    "Reservation": reservation,
    "Age Group": age_group,
    "Live Music": live_music,
    "Customer Satisfaction": customer_satisfaction
}

# Kullanıcının girdiği verileri DataFrame'e dönüştürme
df_user = pd.DataFrame.from_dict([user_data])

st.subheader("Entered Information:")
st.dataframe(df_user)

# Modelinizin beklediği sütunları buraya ekleyin.
columns_expected_by_model = [
    'Number of People', 'Time Spent (minutes)',
    'Customer Satisfaction', 'Live Music_True', 'Reservation_True',
    'Meal Type_Dinner', 'Meal Type_Lunch', 'Day_Monday', 'Day_Saturday',
    'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday',
    'Gender_Male', 'Gender_Other', 'Table Location_Patio',
    'Table Location_Window', 'Age Group_26-35', 'Age Group_36-45',
    'Age Group_46-55', 'Age Group_56-65', 'Age Group_65+',
    'Weather Condition_Rainy', 'Weather Condition_Snowy',
    'Weather Condition_Sunny'
]
df_user = pd.get_dummies(df_user).reindex(columns=columns_expected_by_model, fill_value=0)

# Verileri ölçeklendirme
df_user_scaled = scaler.transform(df_user)

# Karar Ağacı Modeli ile Tahmin Yapma
dt_prediction = dt_model.predict(df_user_scaled)

# XGBoost Modeli ile Tahmin Yapma
xgb_prediction = xgb_model.predict(df_user_scaled)

# Yapay Sinir Ağı Modeli ile Tahmin Yapma
ann_prediction = ann_model.predict(df_user_scaled)

# Tahminleri kullanıcıya gösterme
st.success("Estimated Bill Amount with Decision Tree Model: {:.2f} TL".format(int(dt_prediction[0])))
st.success("Estimated Invoice Amount with XGBoost Model: {:.2f} TL".format(int(xgb_prediction[0])))
st.success("Estimated Invoice Amount with ANN Model: {:.2f} TL".format(int(ann_prediction[0][0])))
# Kodun sonu

