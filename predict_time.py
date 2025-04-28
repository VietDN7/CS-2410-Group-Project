import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#data
df = pd.read_csv("Crime_Data_Cleaned.csv")

#Extract hour from TIME OCC
df['Hour'] = df['TIME OCC'] // 100

#Categorize into time windows
def categorize_time(hour):
    if 0 <= hour < 6:
        return 'Early Morning'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening/Night'

df['Time Category'] = df['Hour'].apply(categorize_time)

# Prepare features and target
features = df[['AREA NAME', 'Crm Cd Desc']].copy() 
target = df['Time Category']

# Encode categorical features
le_area = LabelEncoder()
le_crime = LabelEncoder()

features['Area Encoded'] = le_area.fit_transform(features['AREA NAME'])
features['Crime Encoded'] = le_crime.fit_transform(features['Crm Cd Desc'])
X = features[['Area Encoded', 'Crime Encoded']]
y = target

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)
#Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Prediction vs Actual Crime Time')
plt.show()
