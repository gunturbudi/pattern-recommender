import shap
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'model_fold_1.txt' is your model file and 'test_fold_1.txt' is your test data file
PARENT_PATH = "resultfold/train_1/"
model_path = PARENT_PATH + 'model_fold_4.txt'
test_path = PARENT_PATH + 'test_fold_4.txt'


def define_feature_names():
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


feature_names = define_feature_names()

# Define a function to parse the data, as it's used for both training and test data
def parse_data(data):
    labels = []
    features = []
    qids = []
    for line in data.strip().split('\n'):
        tokens = line.split()
        labels.append(float(tokens[0]))
        qids.append(int(tokens[1].split(':')[1]))
        feat_vals = {}
        for tok in tokens[2:]:
            if ':' in tok:
                feat, val = tok.split(':')
                feat_vals[int(feat)] = float(val)
        features.append(feat_vals)
    return labels, features, qids


# Load the trained model
bst = lgb.Booster(model_file=model_path)

# Manually set the objective parameter if it's not present
if 'objective' not in bst.params:
    bst.params['objective'] = 'lambdarank'

# Parse test data
with open(test_path, "r") as file:
    test_labels, test_features, test_qids = parse_data(file.read())

# Prepare test data matrix
max_feature_idx = max([max(feats.keys()) for feats in test_features])
test_data_matrix = np.array([[feat_vals.get(feat, 0) for feat in range(1, max_feature_idx+1)] for feat_vals in test_features])

# Create SHAP explainer
explainer = shap.TreeExplainer(bst)

# Compute SHAP values
shap_values = explainer.shap_values(test_data_matrix)

shap_explanation = shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value, 
                                    data=test_data_matrix[0], feature_names=feature_names)

# Save the SHAP summary bar plot with the top 10 features to a file
shap.summary_plot(shap_values, test_data_matrix, plot_type='bar', feature_names=feature_names, show=False, max_display=10)
plt.savefig('ltr_shap_summary_bar_top10_named.png')
plt.close()

# Save the SHAP beeswarm plot with the top 10 features to a file
shap.summary_plot(shap_values, test_data_matrix, show=False, feature_names=feature_names, max_display=10)
plt.savefig('ltr_shap_summary_beeswarm_top10_named.png')
plt.close()


# Save the SHAP dependence plot for a specific feature (e.g., Feature 1) to a file
shap.dependence_plot(21, shap_values, test_data_matrix, feature_names=feature_names, show=False)
plt.savefig('ltr_shap_dependence_plot_feature_22.png')
plt.close()

# Save the SHAP dependence plot for a specific feature (e.g., Feature 1) to a file
# shap.dependence_plot(597, shap_values, test_data_matrix, feature_names=feature_names, show=False)
# plt.savefig('ltr_shap_dependence_plot_feature_597.png')
# plt.close()

# Save the SHAP waterfall plot for the first prediction to a file
shap_waterfall_plot = plt.figure()
shap.plots.waterfall(shap_explanation, max_display=10)
shap_waterfall_plot.savefig('ltr_shap_waterfall_plot_named.png')
plt.close(shap_waterfall_plot)
    
    
