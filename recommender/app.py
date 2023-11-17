from flask import Flask, render_template, request, jsonify, url_for
import json
import numpy as np
import lightgbm as lgb
from feature_creation import PrivacyPatternFeatures
import os
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
pp = PrivacyPatternFeatures()

# Load the patterns and the trained model
with open("data/patterns.json", 'r') as p:
    patterns = json.load(p)

pattern_name = [pattern["title"].replace(".md", "") for pattern in patterns]

# Load any additional patterns
with open("data/patterns_new.json", 'r') as p:
    patterns_new = json.load(p)

for p_new in patterns_new:
    patterns.append(p_new)
    pattern_name.append(p_new["title"])

model_file_path = "LTR_resources/model_fold_4_train_3.txt"  # the path to LeToR trained model on LightGBM
if os.path.exists(model_file_path):
    bst = lgb.Booster(model_file=model_file_path)
    bst.params['objective'] = 'lambdarank'
else:
    raise FileNotFoundError("Model file not found.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    new_req_text = request.form['requirement']
    feature_vectors = process_new_requirement(new_req_text)
    data_matrix = np.array(feature_vectors)

    # Get sorted patterns with SHAP plots
    sorted_patterns_with_shap = predict_new_data(bst, data_matrix)

    # Get indices of sorted patterns from original pattern list
    sorted_pattern_indices = [pattern_name.index(pattern[0]) for pattern in sorted_patterns_with_shap]

    # Use these indices to get the correct excerpts and the SHAP image paths
    patterns_with_desc_and_shap = [
        {
            "pattern": sorted_patterns_with_shap[i][0],
            "excerpt": patterns[sorted_pattern_indices[i]]["excerpt"],
            "shap_plot_path": sorted_patterns_with_shap[i][2],  # SHAP plot image path
            "shap_waterfall_plot_path": sorted_patterns_with_shap[i][3]  # SHAP plot image path
        }
        for i in range(len(sorted_patterns_with_shap))
    ]

    # Return JSON response with paths to SHAP plot images
    return jsonify(sorted_patterns=patterns_with_desc_and_shap)

def get_feature_names():
    # Generate feature names for the first 4 features
    feature_names = [
        "Covered Words", "Covered Words Ratio",
        "Length of Query", "IDF of Query"
    ]

    # Generate feature names for TF features (5-14)
    tf_feature_names = [
        "TF Sum", "TF Min", "TF Max", "TF Average", "TF Variance",
        "Normalized TF Sum", "Normalized TF Min", "Normalized TF Max", "Normalized TF Average", "Normalized TF Variance"
    ]
    feature_names.extend(tf_feature_names)

    # Generate feature names for TF-IDF features (15-19)
    tf_idf_feature_names = [
        "TF-IDF Sum", "TF-IDF Min", "TF-IDF Max", "TF-IDF Average", "TF-IDF Variance"
    ]
    feature_names.extend(tf_idf_feature_names)
    
    feature_names.extend([
        "BM25", "Content Similarity #1", "Title Similarity #1", "Excerpt Similarity #1",
        "Content Similarity #2", "Title Similarity #2", "Excerpt Similarity #2",
        "Binary Query", "Multi Query", "Binary Pattern", "Multi Pattern",
        "Content Similarity #1.1", "Title Similarity #1.1", "Excerpt Similarity #1.1",
        "Content Similarity #2.1", "Title Similarity #2.1", "Excerpt Similarity #2.1"
    ])
    
    # return here for train_1
    # return feature_names

    # Define the naming for hadamard product features
    hadamard_feature_sets = [
        "Hadamard Content #1", "Hadamard Title #1", "Hadamard Excerpt #1",
        "Hadamard Content #2", "Hadamard Title #2", "Hadamard Excerpt #2"
    ]
    

    # Generate feature names for the hadamard product features
    for feature_set_name in hadamard_feature_sets:
        for i in range(1, 769):  # Each set has 768 features
            feature_names.append(f"{feature_set_name} {i}")

    # return here for train_2
    # return feature_names

    # Define the naming for concatenation features
    concat_feature_sets = [
        "Concat Content #1", "Concat Title #1", "Concat Excerpt #1",
        "Concat Content #2", "Concat Title #2", "Concat Excerpt #2"
    ]

    # Generate feature names for the concatenation features
    for feature_set_name in concat_feature_sets:
        for i in range(1, 769):  # Each set has 768 features
            feature_names.append(f"{feature_set_name} {i}")
            
    # return here for train_3
    return feature_names

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

def predict_new_data(model, feature_vectors):
    print(len(feature_vectors))
    # Make predictions using the model
    predictions = model.predict(feature_vectors)
    

    # Rank the pattern names based on predictions and select only the top 7
    sorted_indices = np.argsort(predictions)[::-1][:7]
    top_sorted_patterns = [pattern_name[i] for i in sorted_indices]

    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_vectors)

    # Generate SHAP force plot images for each prediction
    shap_image_paths = []
    shap_waterfall_image_paths = []
    for i in sorted_indices:
        print(shap_values[i])
        print(feature_vectors[i])
        
        # FORCE PLOT
        plt.figure()
        shap.force_plot(
            explainer.expected_value, shap_values[i], feature_vectors[i],
            feature_names=get_feature_names(), matplotlib=True, show=False
        )
        image_url = url_for('static', filename=f'shap_plots/pattern_{i}.png')
        image_path = f"static/shap_plots/pattern_{i}.png"
        plt.savefig(image_path)
        shap_image_paths.append(image_url)
        plt.close()
        
        # WATERFALL PLOT
        plt.figure()
        plt.tight_layout()
        # Create an Explanation object
        shap_explanation = shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value, 
            data=feature_vectors[i], 
            feature_names=get_feature_names()
        )
        # Generate a waterfall plot for the i-th prediction
        shap.plots.waterfall(shap_explanation, max_display=14, show=False)
        waterfall_image_path = f"static/shap_plots/waterfall_pattern_{i}.png"
        plt.savefig(waterfall_image_path, bbox_inches='tight')
        shap_waterfall_image_paths.append(url_for('static', filename=f"shap_plots/waterfall_pattern_{i}.png"))
        plt.close()

    # Combine predictions, pattern names, and their corresponding SHAP values
    top_patterns_with_shap = [
        (pattern_name[i], predictions[i], shap_image_paths[j], shap_waterfall_image_paths[j])
        for j, i in enumerate(sorted_indices)
    ]

    return top_patterns_with_shap

if __name__ == '__main__':
    app.run(debug=True)
