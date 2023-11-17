import json
import numpy as np
import lightgbm as lgb
from feature_creation import PrivacyPatternFeatures

pp = PrivacyPatternFeatures()

with open("data/patterns.json", 'r') as p:
    patterns = json.loads(p.read())

pattern_name = [pattern["title"].replace(".md", "") for i, pattern in enumerate(patterns)]

with open("data/patterns_new.json", 'r') as p:
    patterns_new = json.loads(p.read())

    for p_new in patterns_new:
        patterns.append(p_new)
        pattern_name.append(p_new["title"])

def process_new_requirement(new_req_text):
    """
    Process a new privacy requirement text to create a feature vector.

    Parameters:
    - new_req_text (str): The new privacy requirement text.
    - patterns (list): The list of patterns (already loaded from the patterns file).
    - pp (PrivacyPatternFeatures): The PrivacyPatternFeatures instance.
    - pattern_name (list): List of pattern names extracted from the patterns file.

    Returns:
    - A list of feature vectors for the new requirement text.
    """
    features = pp.construct_features(new_req_text)
    feature_vectors = []

    for idx, pattern in enumerate(patterns):
        feature_vector = features[idx]  # Assuming features[idx] is already a list of features
        feature_vectors.append(feature_vector)

    return feature_vectors


def predict_new_data(model, new_req_text):
    """
    Predict the ranking for a new privacy requirement text using the trained model.

    Parameters:
    - model: The trained LightGBM model.
    - new_req_text (str): The new privacy requirement text.

    Returns:
    - The predictions and the sorted pattern names based on their rankings.
    """

    # Process the new requirement text
    feature_vectors = process_new_requirement(new_req_text)
    data_matrix = np.array(feature_vectors)
    
    # Make predictions using the model
    predictions = model.predict(data_matrix)
    
    # Rank the pattern names based on predictions
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_patterns = [pattern_name[i] for i in sorted_indices]

    return predictions, sorted_patterns

# Example usage:
new_req_text = """


If I have written a privacy tool as a .NET web application which is to be hosted on a commercial hosting site, other than hosting it in a privacy friendly country, how can I assure users that the application has not been compromised by a third party at the host?

Obviously SSL will be used and the assemblies will be as obfuscated as possible, but these can only go so far.

For example, is there a way I can ensure that my assemblies haven't been wrapped to intercept plain-text user details?

"""
model_file_path = "LTR_resources/model_fold_4_train_3_law.txt"  # Provide the correct path to your trained model

# Load the trained model
bst = lgb.Booster(model_file=model_file_path)

# Assuming 'pp' (PrivacyPatternFeatures instance) and 'pattern_name' list are already defined in your environment.
predictions, sorted_patterns = predict_new_data(bst, new_req_text)

# Print or process the predictions and sorted pattern names as needed
print("Predictions:", predictions)
print("Ranked Patterns:", sorted_patterns)
