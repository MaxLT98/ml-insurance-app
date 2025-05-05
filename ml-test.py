import joblib
import numpy as np
import sklearn

model = joblib.load("./model/insurance.pkl")
sc_x = joblib.load("./model/scaler_x.pkl")
sc_y = joblib.load("./model/scaler_y.pkl")

edad = int(input("Enter number of year: "))
edad_sc = sc_x.transform(np.array([[edad]]))
#print(f'rooms_sc : {rooms_sc}')

prediction = model.predict(edad_sc)

#print(f'prediction : {prediction}')

prediction_sc = sc_y.inverse_transform(prediction)
print(f'Los gastos medicos para una persona con {edad} años es : $ {prediction_sc[0][0]:.2f}')