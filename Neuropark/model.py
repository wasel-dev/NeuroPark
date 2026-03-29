import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("etv_dataset.csv")

X = data[["age","hydro_type","prior_shunt","ventricle_size","floor_thickness","basilar_distance","needle_depth","needle_angle","entry_x","entry_y"]]
y = data["success"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Accuracy:", model.score(X_test, y_test))