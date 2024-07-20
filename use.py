import numpy as np
from joblib import dump,load
model=load('realestate.joblib')
features=np.array([[
    -5.4,4,-1.6,0,-0.6,-1.4,-11,-49,7,-26,-0.5,-0.9,0.41
]])
print(model.predict(features))