import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Check for outliers using IQR method
def detech_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers)

# Load the dataset
data = np.genfromtxt('wine.data', delimiter=',')
columns = ['Class', 'Alcohol', 'Malic_acid', 'Ash',
           'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
           'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
           'Color_intensity', 'Hue', 'OD280_OD315', 'Proline']
df = pd.DataFrame(data, columns=columns)
df['Class'] = df['Class'].astype(int)

# Checking for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Checking for duplicates
print('\nNumer of duplicate rows:')
print(df.duplicated().sum())

# Class Distribution
print('\nClass Distribution:')
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Checking for autliers for each feature
print('\nOutliers Detection:')
for col in df.columns[1:]:  # Exclude target (Class)
    outlier_count = detech_outliers(df, col)
    print(f'Outliers in {col}: {outlier_count}')
    # plt.boxplot(df[col])
    # plt.title(f'Outliers in {col}')
    # plt.show()

# Features and target
x = df.drop('Class', axis=1)
y = df['Class']

# Split and scale
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model: Multinomial (Softmax) - multi_class='multinomial' (default)
model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

# Evaluate model
print("\nMultinomial (Softmax) Model:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}\n")
print("Confusion Matrix:")
print(cm)

# Output: Alcohol, True Class, Predicted Class
results = pd.DataFrame({
    'True_Class': y_test,
    'Predicted_Class': y_pred
})
print("\nTest Set Predictions:")
print(results.to_string(index=False))
