import streamlit as st
import pickle
import numpy as np
import sklearn
from packaging import version


import sys
st.write(f"Python version: {sys.version}")
st.write(f"Pickle version: {pickle.format_version}")
st.write(f"Sklearn version: {sklearn.__version__}")


if version.parse(sklearn.__version__) < version.parse("1.2.1"):
    st.error("Sklearn version is too old. Please use version 1.2.1 or newer.")
    st.stop()


try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title('Fifa 2022 Player Prediction')
st.header("Enter The Player's Traits")


with st.form(key='numerical_form'):
    number1 = st.number_input('Potential', value=0.0)
    number2 = st.number_input('Value In Euros', value=0.0)
    number3 = st.number_input('Wage In Euros', value=0.0)
    number4 = st.number_input('Movement_Reactions', value=0.0)
    number5 = st.number_input('Mentality_Composure', value=0.0)
    submit_button = st.form_submit_button(label='Submit')


if submit_button:
    st.subheader('Submitted Numbers')
    st.write(f'potential: {number1}')
    st.write(f'value_eur: {number2}')
    st.write(f'wage_eur: {number3}')
    st.write(f'movement_reactions: {number4}')
    st.write(f'mentality_composure: {number5}')

    
    input_data = np.array([number1, number2, number3, number4, number5]).reshape(1, -1)

    
    try:
        prediction = model.predict(input_data)
        st.write(f'Prediction: {prediction[0]}')
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

    st.markdown('---')
    st.markdown('Created with ❤️ by Brigidi Ackah Blay Thompson')