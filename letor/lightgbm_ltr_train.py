import numpy as np
import lightgbm as lgb
import pandas as pd

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

def create_dataset(features, labels, qids):
    num_features = max([max(feat_vals.keys()) for feat_vals in features])
    data = np.array([[feat_vals.get(feat, 0) for feat in range(1, num_features+1)] for feat_vals in features])
    group = np.unique(qids, return_counts=True)[1]
    return lgb.Dataset(data=data, label=labels, group=group, free_raw_data=False)

# Train and evaluate model
def train_evaluate(train_file, test_file):
    # Parse training data
    with open(train_file, "r") as file:
        train_labels, train_features, train_qids = parse_data(file.read())
    
    # Parse test data
    with open(test_file, "r") as file:
        test_labels, test_features, test_qids = parse_data(file.read())
    
    # Create datasets
    train_dataset = create_dataset(train_features, train_labels, train_qids)
    test_dataset = create_dataset(test_features, test_labels, test_qids)
    test_dataset.reference = train_dataset
    
    # Parameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': list(range(1, 11)),
        'learning_rate': 0.1,
        'num_leaves': 31
    }
    num_round = 1000
    
    # Train model
    bst = lgb.train(params, train_dataset, num_round, valid_sets=[test_dataset], valid_names=['test'])
    
    # Make predictions
    max_feature_idx = max([max(feats.keys()) for feats in train_features])
    test_data_matrix = np.array([[feat_vals.get(feat, 0) for feat in range(1, max_feature_idx+1)] for feat_vals in test_features])
    test_preds = bst.predict(test_data_matrix)
    
    # Results
    results = {f"ndcg@{i+1}": bst.best_score['test'][f'ndcg@{i+1}'] for i in range(10)}
    
    return results, bst

def generate_file_paths(fold_num, base_path="train_3"):
    """Generate file paths for training, testing, and model based on fold number."""
    train_file = f"{base_path}/train_fold_{fold_num}.txt"
    test_file = f"{base_path}/test_fold_{fold_num}.txt"
    model_file = f"{base_path}/model_fold_{fold_num}.txt"
    return train_file, test_file, model_file

def perform_5_fold_train_test():
    """Perform 5-fold training and testing, saving results and models."""
    results_list = []
    for i in range(1, 6):
        train_file, test_file, model_file = generate_file_paths(i)
        
        results, bst = train_evaluate(train_file, test_file)
        results_list.append(results)
        bst.save_model(model_file)
    
    return results_list

# Execute 5-fold training/testing and save results
results_list = perform_5_fold_train_test()

# Store results in Excel
df = pd.DataFrame(results_list)
results_path = "train_results.xlsx"
df.to_excel(results_path, index=False)
