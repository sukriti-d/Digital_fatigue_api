import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv("D:\digital_fatigue_api\digital_fatigue\model\digital_fatigue_profiled_dataset.csv")

# Features and target
X = df.drop('fatigue_level', axis=1)
y = df['fatigue_level']

# Identify categorical and numerical columns
categorical_features = ['profile']
numerical_features = ['screen_time_hours', 'nighttime_use', 'app_switches', 'social_media_ratio', 'unlocks']

# Preprocessing for categorical data (with imputation and encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values in categorical data
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One hot encode categorical variables
])

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values in numerical data
    ('scaler', StandardScaler())  # Normalize numerical data
])

# Combine preprocessing for both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

# Build pipeline with RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'digital_fatigue_model.joblib')

# Optional: Check accuracy
accuracy = pipeline.score(X_test, y_test)
print(f"Model trained successfully! Accuracy: {accuracy:.2f}")
