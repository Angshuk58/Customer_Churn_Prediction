import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model and necessary objects
model = load_model('my_model.h5')
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)
with open("onehotencoder_geo.pkl", "rb") as file:
    onehotencoder_geo = pickle.load(file)
with open("onehotencoder_card.pkl", "rb") as file:
    onehotencoder_card = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Set up the Streamlit app
st.title("Customer Churn Prediction")

# Create input fields for user input
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])  # Assuming 0 for No and 1 for Yes
IsActiveMember = st.selectbox("Is Active Member", [0, 1])  # Assuming 0 for No and 1 for Yes
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
Card_Type = st.selectbox("Card Type", ['Blue', 'Gold', 'Platinum', 'Silver'])  # Assuming 0 for No and 1 for Yes


# Create a button to trigger prediction
if st.button("Predict Churn"):
    # Create input data dictionary
    input_data = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Card Type': Card_Type
    }

    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = label_encoder.transform(input_df['Gender'])
    geo_encoded = onehotencoder_geo.transform(input_df[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
    card_encoded = onehotencoder_card.transform(input_df[['Card Type']]).toarray()
    card_encoded_df = pd.DataFrame(card_encoded, columns=onehotencoder_card.get_feature_names_out(['Card Type']))
    input_df = pd.concat([input_df.drop("Card Type", axis=1), card_encoded_df], axis=1)

    # Get the columns used during training
    training_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
           'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France',
           'Geography_Germany', 'Geography_Spain', 'Card Type_Blue', 'Card Type_Gold',
           'Card Type_Platinum', 'Card Type_Silver']

    # Ensure input_df has the same columns as training data
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # Display the prediction
    if prediction_proba > 0.5:
        st.write("Prediction: The customer is likely to churn.")
    else:
        st.write("Prediction: The customer is not likely to churn.")
