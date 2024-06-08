import streamlit as st
import pandas as pd
import pickle

# Load the logistic regression model
with open('model.pkl', 'rb') as file:
    LR = pickle.load(file)

st.title("Logistic Regression Model Prediction")

st.header("Input Features")

# Input fields for the features
Pclass = st.radio("Pclass", options=[1, 2, 3], index=0)
SibSp = st.slider("SibSp", min_value=0, max_value=5, step=1)
Parch = st.slider("Parch", min_value=0, max_value=5, step=1)
Fare = st.text_input("Fare", value="0.00")
Age = st.text_input("Age", value="0.00")
Cabin = st.radio("Cabin", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
Embarked = st.radio("Embarked", options=['Cherbourg', 'Queenstown', 'Southampton'])
Sex_encoded = st.radio("Sex", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

# Mapping Embarked option to its encoded value
embarked_mapping = {
    'Cherbourg': 0.553571,
    'Queenstown': 0.389610,
    'Southampton': 0.339009
}
Embarked_encoded = embarked_mapping[Embarked]

# Convert text inputs to float
Fare = float(Fare)
Age = float(Age)

# Prepare the input data as a DataFrame
test_input_data = {
    'Pclass': [Pclass],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Age': [Age],
    'Cabin': [Cabin], 
    'Embarked_encoded': [Embarked_encoded],
    'Sex_encoded': [Sex_encoded]
}

test_input_df = pd.DataFrame(test_input_data)

# Predict and display the results when the button is clicked
if st.button("Predict"):
    prediction = LR.predict(test_input_df)
    prediction_proba = LR.predict_proba(test_input_df)
    
    st.subheader("Prediction")
    if int(prediction[0]) == 0:
        st.image("abc.jpg", caption="Passenger is dead!!", use_column_width=True)
        st.write("😢 Passenger is dead!!")
    else:
        st.balloons()
        st.snow()
        st.write("🎉 Passenger survived!!")
        st.markdown("# Passenger Survived")

    st.subheader("Prediction Probability")
    st.write("Probability of each class:", prediction_proba[0])
