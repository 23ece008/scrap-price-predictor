# model_train.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle

materials = ["iron", "plastic", "copper", "aluminum"]
data = []

for _ in range(300):
    material = np.random.choice(materials)
    weight = np.random.randint(1, 50)
    demand = np.random.randint(1, 10)

    base_price = {
        "iron": 30,
        "plastic": 20,
        "copper": 600,
        "aluminum": 150
    }[material]

    price = base_price + (demand * np.random.uniform(2, 8))

    data.append([material, weight, demand, datetime.now().hour, datetime.now().weekday(), price])

df = pd.DataFrame(data, columns=["material", "weight", "demand", "hour", "day", "price"])

le = LabelEncoder()
df["material"] = le.fit_transform(df["material"])

X = df[["material", "weight", "demand", "hour", "day"]]
y = df["price"]

model = XGBRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("✅ Model Ready!")
