import streamlit as st
import os
import pickle
import pandas as pd
from xgboost import XGBRegressor
model =XGBRegressor()
model.load_model(r'xgb_model.json') 
encoders= pickle.load(open(r'used_car_evauluation_encoders.pkl', "rb"))
scaler= pickle.load(open(r'used_car_evauluation_scaler.pkl', "rb"))




st.set_page_config(page_title="Used Car Price Evalaution")
st.title("Get the approx price of your used car.")
st.write("Please enter the following details.")

with st.form(key="car_data"):
    brand=st.selectbox("Choose your cars make",['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mitsubishi', 
                                                'Renault', 'Mahindra', 'Ford','Datsun', 'Chevrolet',
    'Skoda', 'Fiat', 'Smart', 'Ambassador', 'Isuzu', 'Force','Audi', 'Nissan', 'Volkswagen', 'Land Rover',
    'Mercedes-Benz', 'BMW', 'Porsche','Jaguar', 'Volvo', 'Mini Cooper','Jeep', 'Bentley', 'Lamborghini'])

    location= st.selectbox("Choose your location!!",['Pune', 'Chennai', 'Coimbatore', 'Jaipur', 'Mumbai', 'Kochi',
        'Kolkata', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad'])

    year=st.slider("Choose the year your car was manufactuered in",min_value=1998,max_value=2024,step=1)

    kilometers_driven=st.slider("Please enter the number of kilometers your car has been driven",
                                min_value=5000,max_value=500000,step=100)

    fuel_type=st.select_slider("Please select the fuel type",['Petrol','Diesel','Electric'])
    transmission=st.selectbox("Please Select the transmision type",["Manual",'Transmission'])
    owner_type=st.selectbox("Please select how many owners this car has had",['First','Second','Third','Fourth & Above'])
    avg_fuel_consumption=st.slider("Please select average fuel consumption of your car per KM",min_value=3,max_value=30)
    engine_capacity=st.number_input("Please enter the engine capacity of your car",min_value=500,max_value=5000)
    power=st.number_input("Please enter the power output of your vehicle",min_value=50,max_value=400)
    seats=st.slider("Please select the number of seats your car has",min_value=1,max_value=10)
    age=2024-year
    submit_button=st.form_submit_button(label="Submit")

data={'Name':brand,'Location':location,'Kilometers_Driven':kilometers_driven,"Fuel_Type":fuel_type,
        'Transmission':transmission,'Owner_Type':owner_type,'Mileage':avg_fuel_consumption,"Engine":engine_capacity,
        "Power":power,"Seats":seats,"age":age}

cols = ['Name', 'Location', 'Fuel_Type', 'Transmission','Owner_Type']
def predict_price(data, encoders,scaler, model): # Add '_' before encoders
  df = pd.DataFrame([data])
  df1=encoders.transform(df[cols])
  df1=pd.DataFrame(df1,columns=encoders.get_feature_names_out(cols))
  df=df.drop(cols,axis=1)
  df=pd.concat([df,df1],axis=1)
  feature_order=scaler.feature_names_in_
  df=df[feature_order]
  df = scaler.transform(df)
  response = model.predict(df)
  return response

if submit_button:
    if all(value is not None for value in data.values()):
        predicted_price=predict_price(data,encoders,scaler,model)
        st.success(f"The estimated price of your vehicle is : ₹. {predicted_price[0].round()}" ) 
    else:
        st.write("Please enter all the details")

