import numpy as np
from joblib import dump, load

model=load('house_price_predictor.joblib')

features=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']


features_input_list=[]

print("#housePricePredictor\nEnter features of the house.")
for i in features:
    print(f"Enter {i}: ",end="")
    i=float(input())
    features_input_list.append(i)

features_input_array=np.array([features_input_list])

prediction=model.predict(features_input_array)
print(f"{"="*10}\nPRICE OF THE HOUS IS : {prediction}\n{"="*10}")