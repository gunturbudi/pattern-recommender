import lightgbm as lgb
import numpy as np
import pandas as pd

base_path="train/"

# Function to read and extract feature indices from LightSVM formatted data
def get_feature_indices(files):
    all_features = set()
    for file_name in files:
        with open(base_path + file_name, 'r') as file:
            for line in file:
                # Remove the comment part of the line if it exists
                line = line.split('#')[0].strip()
                # Skip the label and qid, then extract feature indices
                tokens = line.strip().split()[2:]  # Skip the label and qid
                features = {int(tok.split(':')[0]) for tok in tokens if ':' in tok}
                all_features.update(features)
    return all_features

# Paths to your testing data files
train_files = [
    'test_fold_1.txt'
]

# Get all feature indices
all_feature_indices = get_feature_indices(train_files)
num_total_features = max(all_feature_indices)

# Load your models and calculate feature importances
model_files = [
    'model_fold_1.txt',
    'model_fold_2.txt',
    'model_fold_3.txt',
    'model_fold_4.txt',
    'model_fold_5.txt'
]

# Initialize a dictionary to store the feature importances from all folds
feature_importances = {f'f{i}': [] for i in range(1, num_total_features + 1)}

# Load each model and gather the feature importances
for model_file in model_files:
    bst = lgb.Booster(model_file=base_path + model_file)  # Load the model
    fold_importance = bst.feature_importance(importance_type='gain')
    # Store the feature importances for the fold
    for i, importance in enumerate(fold_importance, start=1):
        feature_importances[f'f{i}'].append(importance)

# Calculate the average importance for each feature
average_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}

# Convert to a DataFrame for easier manipulation and saving to Excel
importance_df = pd.DataFrame.from_dict(average_importances, orient='index', columns=['Average Importance'])
importance_df.index.name = 'Feature'

# Sort the DataFrame by the feature importances
importance_df = importance_df.sort_values(by='Average Importance', ascending=False)

# Save to an Excel file
importance_df.to_excel(base_path + 'feature_importances.xlsx')

print("Feature importances have been calculated and saved to feature_importances.xlsx")
