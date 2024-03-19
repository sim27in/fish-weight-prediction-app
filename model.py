import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# data split and transformation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
import pickle


# Load and inspect the dataset
data = 'Fish.csv'
fish_df = pd.read_csv(data)
print(fish_df.head(10))


print(fish_df.info())

# Statistical summary
print(fish_df.describe())

null_= pd.DataFrame(fish_df.isna().sum())
null_

# Visualize distributions of numerical features

species_counts = fish_df['Species'].value_counts()

# Create a bar plot for the species distribution
plt.figure(figsize=(4, 3))

species_counts.plot(kind='bar', color='green')
plt.title('Fish Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# checking for correlations
numerical_df = fish_df.select_dtypes(include=['float64', 'int64'])

corr = numerical_df.corr()
corr.style.background_gradient(cmap="ocean")



# checking for outliers and distribution

numerical = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
    
# finding the outliers
for col in numerical:
    df = fish_df[col]
    df_Q1 = df.quantile(0.25)
    df_Q3 = df.quantile(0.75)
    df_IQR = df_Q3 - df_Q1
    df_lowerend = df_Q1 - (1.5 * df_IQR)
    df_upperend = df_Q3 + (1.5 * df_IQR)

    df_outliers = df[(df < df_lowerend) | (df > df_upperend)]
    print(df_outliers)
    
# dropping the outlier at row # 142
fish_data = fish_df.drop([142, 143, 144])

# checking for collinerity between the 3 length variables
fish_data[["Length1", "Length2", "Length3", "Weight"]].corr()

fish_data[["Length1", "Length2", "Length3"]].head(10)



fish_data = fish_data.drop(["Length1"], axis=1)

# Selecting features and target variable
X = fish_data.drop('Weight', axis=1)
y = fish_data['Weight']

# Defining numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = ['Species']  # Based on dataset inspection

# Creating preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols),
    ]
)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Building a pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

# Fitting the model
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Test MSE: {mse}")
print(f"Random Forest Test R^2 Score: {r2}")


# Ensure all necessary imports are present
pickle.dump(pipeline, open('model.pkl','wb'))


# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Sample input data in DataFrame format
data = {
    'Species': ['Bream'],  # Example species
    'Length1': [23.2],  # Vertical length in cm
    'Length2': [25.4],  # Diagonal length in cm
    'Length3': [30.0],  # Cross length in cm
    'Height': [11.52],  # Height in cm
    'Width': [4.02]  # Diagonal width in cm
}

# Convert the input data to a DataFrame
sample_df = pd.DataFrame(data)

# Making a prediction using the DataFrame with the loaded model
prediction = model.predict(sample_df)
print(f"Predicted Weight: {prediction[0]}")