import json, pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

import numpy as np

import json

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_stopword, lemmatize_word

preprocess_functions = [to_lower, remove_stopword, lemmatize_word]

privacy_objectives = {
    "anonymity" : ["Protection-against-tracking", "Location-granularity", "Pseudonymous-messaging", "Onion-routing", "Anonymous-reputation-based-blacklisting", "Attribute-based-credentials", "Anonymity-set"],
    
    "unlinkability" : ["Protection-against-tracking", "Location-granularity", "Pseudonymous-messaging", "Onion-routing", "Anonymous-reputation-based-blacklisting", "Attribute-based-credentials", "Decoupling-[content]-and-location-information-visibility","Active-broadcast-of-presence","Trustworthy-privacy-plugin"],
    
    "confidentiality" : ["Informed-Secure-Passwords", "Encryption-user-managed-keys", "Personal-data-store", "Aggregation-gateway", "Single-Point-of-Contact", "User-data-confinement-pattern", "Selective-Access-Control", "Buddy-List", "Added-noise-measurement-obfuscation", "Trustworthy-privacy-plugin", "Support-Selective-Disclosure", "Private-link", "Active-broadcast-of-presence", "Unusual-activities"],
    
    "plausible_deniability" : ["Location-granularity", "Use-of-dummies", "Onion-routing", "Pseudonymous-identity", "Added-noise-measurement-obfuscation", "Attribute-based-credentials", "Anonymity-set"],
    
    "undetectability" : ["Location-granularity", "Use-of-dummies", "Aggregation-gateway", "Trustworthy-privacy-plugin", "Active-broadcast-of-presence"],
    
    "manageability" : ["Federated-privacy-impact-assessment", "Data-breach-notification-pattern", "Trust-Evaluation-of-Services-Sides", "Sign-an-Agreement-to-Solve-Lack-of-Trust-on-the-Use-of-Private-Data-Context", "Obligation-management", "Privacy-Aware-Wording", "Sticky-policy"],
    
    "intervenability" : ["Minimal-Information-Asymmetry", "Informed-Secure-Passwords", "Awareness-Feed", "Encryption-user-managed-keys", "Whos-Listening", "Discouraging-blanket-strategies", "Outsourcing-[with-consent]", "Personal-data-store", "Single-Point-of-Contact", "Enable-Disable-Functions", "Obtaining-Explicit-Consent", "Decoupling-[content]-and-location-information-visibility", "Selective-Access-Control", "Informed-Credential-Selection", "Reasonable-Level-of-Control", "Masquerade", "Buddy-List", "Lawful-Consent", "Sticky-policy", "Personal-Data-Table", "Informed-Consent-for-Web-based-Transactions", "Support-Selective-Disclosure", "Private-link", "Active-broadcast-of-presence"],
    
    "transparency" : ["Minimal-Information-Asymmetry", "Informed-Secure-Passwords", "Awareness-Feed", "Whos-Listening", "Privacy-Policy-Display", "Layered-policy-design", "Asynchronous-notice", "Abridged-Terms-and-Conditions", "Policy-matching-display", "Ambient-notice", "Dynamic-Privacy-Policy-Display", "Privacy-Labels", "Data-breach-notification-pattern", "Trust-Evaluation-of-Services-Sides", "Appropriate-Privacy-Icons", "Privacy-aware-network-client", "Informed-Implicit-Consent", "Privacy-color-coding", "Icons-for-Privacy-Policies", "Obtaining-Explicit-Consent", "Privacy-Mirrors", "Appropriate-Privacy-Feedback", "Impactful-Information-and-Feedback", "Platform-for-Privacy-Preferences", "Privacy-dashboard", "Preventing-Mistakes-or-Reducing-Their-Impact", "Informed-Credential-Selection", "Privacy-Awareness-Panel", "Lawful-Consent", "Privacy-Aware-Wording", "Sticky-policy", "Personal-Data-Table", "Informed-Consent-for-Web-based-Transactions", "Increasing-Awareness-of-Information-Aggregation", "Unusual-activities"],
}

hard_goal = ["unlinkability","anonymity","pseudonym","undetectability","confidentiality","plausible_deniability"] 
soft_goal = ["transparency","intervenability","content_awareness"]
skip_goal = ["availability", "integrity"]

hard_pattern = []
soft_pattern = []

for g in hard_goal:
    if g not in privacy_objectives:
        continue

    for o in privacy_objectives[g]:
        hard_pattern.append(o)

for g in soft_goal:
    if g not in privacy_objectives:
        continue

    for o in privacy_objectives[g]:
        soft_pattern.append(o)

def get_data(multiclass=False):
    req_path = "../data/requirements.json"

    with open(req_path, 'r') as p:
        requirements = json.loads(p.read())

    text, label = [], []
    for r in requirements["rows"]:
      text.append(preprocess_text(r["req_text"], preprocess_functions))
      lbl = r["req_type"].replace("_3","").replace("_2","").replace("_1","")

      if multiclass:
        label.append(lbl)
      else:
        label.append(1 if lbl in hard_goal else 0)

    return text, label

def append_text_label(filepath):
    text, label, label_unique = [], [], []

    # MAKE LABEL AS INDEX
    with open(filepath,"r",encoding='utf-8') as dd:
        for d in dd:
            if len(d)<=8:
                continue

            l = d.split()[0].replace("__label__","")

            if l in skip_goal:
                continue

            label_unique.append(l)

    label_unique = list(set(label_unique))
    print(label_unique)

    with open(filepath,"r") as dd:
        for d in dd:
            if len(d)<=8:
                continue

            l = d.split()[0].replace("__label__","")

            if l in skip_goal:
                continue

            label.append(label_unique.index(l))
            text.append(" ".join(d.split()[1:]))

            # if d[9].strip() in ['0','1']:
            #     label.append(d[9].strip())
            #     text.append(d[11:].strip())

    return text, label


def get_data_from_file(with_aug=True, combine_test=True, binary="binary"):
    text, label = [], []

    if with_aug:
        text_temp, label_temp = append_text_label("data/privacy_{}_data_train_aug.txt".format(binary))
        text.extend(text_temp)
        label.extend(label_temp)

    else:
        text_temp, label_temp = append_text_label("data/privacy_{}_data_train.txt".format(binary))
        text.extend(text_temp)
        label.extend(label_temp)

    if combine_test:
        text_temp, label_temp = append_text_label("data/privacy_{}_data_test.txt".format(binary))
        text.extend(text_temp)
        label.extend(label_temp)

    return text, label

def test_cross_val():
    text, label = get_data_from_file(with_aug=False, combine_test=True, binary="multi")

    count_vect = CountVectorizer(analyzer="word", ngram_range=(1,1))
    train_data = count_vect.fit_transform(text)


    clf1 = MultinomialNB()

    print ("Naive Bayes:", np.mean(cross_val_score(clf1, train_data, label, scoring='f1_macro',cv=5))) 

def make_classifier():
    text, label = get_data_from_file(with_aug=True, combine_test=False, binary="multi")

    count_vect = CountVectorizer(analyzer="word", ngram_range=(1,1))
    train_data = count_vect.fit_transform(text)

    filename = 'multi_vectorizer_model.sav'
    pickle.dump(count_vect, open(filename, 'wb'))

    clf1 = MultinomialNB()
    clf1.fit(train_data, label)

    filename = 'multi_nb_model.sav'
    pickle.dump(clf1, open(filename, 'wb'))

def predict_class(texts, model, vectorizer):
    loaded_model = pickle.load(open(model, 'rb'))
    loaded_vect = pickle.load(open(vectorizer, 'rb'))

    text = [preprocess_text(t, preprocess_functions) for t in texts]
    v_text = loaded_vect.transform(text)
    
    prediction = loaded_model.predict(v_text)

    return prediction

def test_classifier():
    test_data = "../data/sec_compass.json"

    with open(test_data, 'r', encoding="utf-8-sig") as p:
        requirements = json.loads(p.read())

    req_type = [r["req_type"] for r in requirements["rows"]]
    texts = [r["req_text"] for r in requirements["rows"]]
    prediction = predict_class(texts,'nb_model.sav','vectorizer_model.sav')

    for i,p in enumerate(prediction):
        print(req_type[i], p)

def test_classifier_on_pattern():
    pattern_file = "../data/patterns.json"

    with open(pattern_file, 'r') as p:
        patterns = json.loads(p.read())

    texts = []

    for pattern in patterns:
        pattern_text = []

        pattern_text.append(pattern["excerpt"])

        for heading in pattern["heading"]:
            pattern_text.append(heading["content"].strip())

        texts.append(". ".join(pattern_text))

    # BINARY CLASS
    prediction = predict_class(texts,"binary_nb_model.sav","binary_vectorizer_model.sav")

    for i,p in enumerate(prediction):
        print(patterns[i]["title"], p)

    print("=="*20)

    # MULTI CLASS
    prediction = predict_class(texts,"multi_nb_model.sav","multi_vectorizer_model.sav")

    for i,p in enumerate(prediction):
        print(patterns[i]["title"], p)



def test_classifier_flair():
    from flair.data import Sentence
    from flair.models import TextClassifier

    # only for binary
    # after testing this model, I dont know why it shows all 1
    model = TextClassifier.load('binary/glove-roberta-best-model.pt') # create example sentence

    test_data = "../data/sec_compass.json"

    with open(test_data, 'r', encoding="utf-8-sig") as p:
        reqs = json.loads(p.read())

    for r in reqs["rows"]:
        sentence = Sentence(r["req_text"])
        model.predict(sentence)
        print("True Type:",r["req_type"],r["req_type"])
        # print(sentence.labels)
        print(sentence.labels[0].value, sentence.labels[0].score)
        print()

def reduce_test_data():
    test_data = "../data/propan_patterns_requirements.json"

    with open(test_data, 'r', encoding="utf-8-sig") as p:
        requirements = json.loads(p.read())

    texts = [r["req_text"] for r in requirements]

    prediction = predict_class(texts)

    new_test_data = []
    for i, label in enumerate(prediction):
        reduced_pattern = []
        goal = soft_pattern
        if label == "1":
            goal = hard_pattern

        print(goal)

        for p in requirements[i]["pattern"]:
            if p["name"] in goal:
                reduced_pattern.append(p)

        new_test_data.append({"id":requirements[i]["id"],"req_text":requirements[i]["req_text"],"pattern":reduced_pattern})

    with open("../data/reduced_propan_patterns_requirements.json", "w") as outfile:
        json.dump(new_test_data, outfile, indent=3)


def classify_new_data():
    json_file_path = 'all_inqueries.json'
    
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        
        texts = [d["req_text"] for d in data]
        
        prediction = predict_class(texts, "model/binary_nb_model.sav", "model/binary_vectorizer_model.sav")
        
        # Return data and prediction labels
        return data, prediction

def split_dataset_into_5fold():
    data, labels = classify_new_data()

    # Create a dictionary to hold data grouped by labels
    grouped_data = {
        0: [],
        1: []
    }

    for d, label in zip(data, labels):
        grouped_data[int(label)].append(d)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    folds = []

    for train_idx, test_idx in skf.split(data, labels):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        folds.append((train_data, test_data, train_labels, test_labels))

    return folds

def split_dataset_uniformly():
    data, labels = classify_new_data()

    # Create a dictionary to hold data grouped by labels
    grouped_data = {
        0: [],
        1: []
    }
    
    for d, label in zip(data, labels):
        grouped_data[int(label)].append(d)

    # Split each group into train, dev, and test sets
    train, test = [], []
    for label, items in grouped_data.items():
        train_set, test_set = train_test_split(items, test_size=0.2, random_state=42)  # Splitting 80% train, 20% test
        
        train.extend(train_set)
        test.extend(test_set)

    return train, test

def export_to_json(data, filename):
    """Export data to a JSON file."""
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
folds = split_dataset_into_5fold()

for fold_num, (train_data, test_data, train_labels, test_labels) in enumerate(folds, 1):
    # Exporting train data for each fold to JSON
    train_filename = f'train_patterns_req_v2_fold_{fold_num}.json'
    export_to_json(train_data, train_filename)
    
    # Exporting test data for each fold to JSON
    test_filename = f'test_patterns_req_v2_fold_{fold_num}.json'
    export_to_json(test_data, test_filename)

    
# test_cross_val()
# make_classifier()
# test_classifier()

# test_classifier_on_pattern()

# reduce_test_data()
