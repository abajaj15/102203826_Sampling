import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Load the dataset
data = pd.read_csv('/mnt/data/Creditcard_data.csv')
print("Dataset Preview:")
print(data.head())

# Split features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define sampling techniques
samplers = {
    "Original": (X_scaled, y),
    "Random OverSampler": RandomOverSampler(random_state=42).fit_resample(X_scaled, y),
    "SMOTE": SMOTE(random_state=42).fit_resample(X_scaled, y),
    "Random UnderSampler": RandomUnderSampler(random_state=42).fit_resample(X_scaled, y),
    "SMOTEENN": SMOTEENN(random_state=42).fit_resample(X_scaled, y)
}

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Evaluate each combination of sampling technique and model
results = []

for sampler_name, (X_resampled, y_resampled) in samplers.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((sampler_name, model_name, accuracy))

# Create a DataFrame for the results
results_df = pd.DataFrame(results, columns=['Sampling Method', 'Model', 'Accuracy'])

# Display the results
print("\nResults Table:")
print(results_df.to_string(index=False))

# Pivot the results to matrix form
results_matrix = results_df.pivot(index='Sampling Method', columns='Model', values='Accuracy')
print("\nResults Matrix:")
print(results_matrix.to_markdown())

# Determine the best sampling technique and model combination
best_result = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Combination:")
print(f"Sampling Method: {best_result['Sampling Method']}, Model: {best_result['Model']}, Accuracy: {best_result['Accuracy']:.2f}")
