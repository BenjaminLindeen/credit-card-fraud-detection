import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the dataset
file_path = '/Users/benjaminlindeen/developement/school/ie5533/credit-card-fraud-detection/datatset/card_transdata.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Extracting features and target variable
X = data.drop('fraud', axis=1)
y = data['fraud']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Building the ANN model
model_smote = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_smote.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model_smote.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model_smote.fit(X_train_smote, y_train_smote, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model on test data
loss_smote, accuracy_smote = model_smote.evaluate(X_test_scaled, y_test)
print(f"SMOTE Test Accuracy: {accuracy_smote*100:.2f}%")

# Making predictions on the test data
y_pred_smote = model_smote.predict(X_test_scaled)
y_pred_classes_smote = (y_pred_smote > 0.5).astype("int32")

# Calculating the confusion matrix
conf_matrix_smote = confusion_matrix(y_test, y_pred_classes_smote)

# Calculating precision, recall, F1-score, and ROC-AUC
classification_rep_smote = classification_report(y_test, y_pred_classes_smote)
roc_auc_smote = roc_auc_score(y_test, y_pred_smote)

print("Confusion Matrix:\n", conf_matrix_smote)
print("\nClassification Report:\n", classification_rep_smote)
print(f"ROC-AUC Score: {roc_auc_smote:.2f}")
