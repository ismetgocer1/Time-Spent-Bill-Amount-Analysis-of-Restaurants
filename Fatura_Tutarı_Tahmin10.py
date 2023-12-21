import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.header("Restoran Fatura Tahmin Uygulaması")

# Model ve ölçeklendirici dosyalarını yükleme
dt_model = load_model('final_DT_model.h5')
ann_model = load_model('final_ANN_model.h5')
scaler = pickle.load(open('final_scaler_saved', "rb"))

st.sidebar.title("Lütfen faturayla ilgili detayları giriniz.")

# Örnek özellik seçimleri
meal_type = st.sidebar.selectbox("Hangi öğünde yemek yediniz?", ['Kahvaltı', 'Öğle Yemeği', 'Akşam Yemeği'])
table_location = st.sidebar.selectbox("Masa konumu neresi?", ['Pencere Kenarı', 'Merkez', 'Bahçe'])
weather_condition = st.sidebar.selectbox("Hava durumu nasıldı?", ['Güneşli', 'Bulutlu', 'Yağmurlu', 'Karlı'])
day = st.sidebar.selectbox("Haftanın günü nedir?", ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'])
age_group = st.sidebar.selectbox("Yaş grubunuz nedir?", ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
reservation = st.sidebar.radio("Rezervasyon yaptırdınız mı?", [1, 0])
live_music =  st.sidebar.radio("Canlı müzik vardı mı?", [1, 0])
number_of_people = st.sidebar.number_input("Yemekte kaç kişi vardı?", min_value=1, value=1)
time_spent = st.sidebar.number_input("Restoranda ne kadar zaman geçirdiniz? (dakika)",  min_value=0)
gender = st.sidebar.radio("Cinsiyetiniz nedir?", ['Kadın', 'Erkek', 'Diğer'])
customer_satisfaction = st.sidebar.slider('Müşteri Memnuniyeti', 1, 5, 3)

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

st.subheader("Girdiğiniz Detaylar:")
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

# Yapay Sinir Ağı Modeli ile Tahmin Yapma
ann_prediction = ann_model.predict(df_user_scaled)

# Tahminleri kullanıcıya gösterme
st.success("Karar Ağacı Modeli ile Tahmini Fatura Tutarınız: {:.2f} TL".format(int(dt_prediction[0])))
st.success("Yapay Sinir Ağı Modeli ile Tahmini Fatura Tutarınız: {:.2f} TL".format(int(ann_prediction[0][0])))


