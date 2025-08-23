import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
data = np.genfromtxt('wine.data', delimiter=',', dtype=np.float64)
df = pd.DataFrame(data=data, columns=[
    'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
])

# Step 1: Data Cleaning
print('=== Data Cleaning ===')

# Checking for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Checking for duplicates
print('\nNumer of duplicate rows:')
print(df.duplicated().sum())

# Checking for autliers for each feature
print('\nOutliers Detection:')
for col in df.columns[1:]:
    outlier_count = detech_outliers(df, col)
    print(f'Outliers in {col}: {outlier_count}')
    plt.boxplot(df[col])
    plt.title(f'Outliers in {col}')
    plt.show()
    
# Data types
print('\nData Types:')
print(df.dtypes)
    