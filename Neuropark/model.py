import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data.csv")

X = data[["pos_x","pos_y","pos_z","rot_x","rot_y","rot_z","depth"]]
y = data["outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Accuracy:", model.score(X_test, y_test))