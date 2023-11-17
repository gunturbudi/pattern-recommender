import nltk

nltk.download('stopwords')
nltk.download('punkt')

import json, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_stopword, lemmatize_word
from transformers import BertTokenizer, BertModel
import torch
import os
import time

preprocess_functions = [to_lower, remove_stopword, lemmatize_word]

PARENT_FOLDER = ""

'''
Construct features for learning-to-rank
The main function is the construct_features(q) which receives input of a query (which in our case a requirement)
then it is computed for each pattern in privacypatterns.org
'''

class PrivacyPatternFeatures(object):
    def __init__(self):
        self.patterns, self.pattern_titles, self.pattern_excerpts = self.get_corpus_pattern()
        self.initiate_tf_idf()
        self.initiate_bm25(0.75, 1.6)

        print("Loading LTR Embeddings...")

        self.model_sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        self.model_sentence_transformer_overflow = SentenceTransformer('dean-ai/legal_heBERT_ft')

        self.emb_pattern_file = PARENT_FOLDER + 'LTR_resources/emb_pattern.pkl'
        if os.path.isfile(self.emb_pattern_file):
            self.load_pattern_embeddings()
        else:
            self.precompute_pattern_embeddings()

    def construct_features(self, q):
        q_words = word_tokenize(self.remove_stopwords(q))

        # we adapt the representation from MSLR-WEB dataset
        # q is query that represents the requirements
        # pattern is the document
        # each query have the pattern features
        # query level feature = when the parameter only contain q

        len_q = len(self.remove_stopwords(q))
        idf_q = self.get_idf(q_words)
        tf_idf_q = self.tf_idf_features(q)
        bm25 = self.bm25(q)
        binary_q, multi_q = self.class_features([q])
        binary_pattern, multi_pattern = self.class_features(self.patterns)

        cosine_pattern, cosine_title, cosine_excerpt, cosine_pattern_overflow, cosine_title_overflow, cosine_excerpt_overflow = self.semantic_similarity_features(q)
        deep_semantic_features = self.deep_semantic_interaction_features(q)

        features_all = []
        for i, pattern in enumerate(self.patterns):
            features = []
            features.extend(self.number_of_covered_words(q_words, pattern)) # 1, 2
            features.append(len_q) # 3
            features.append(idf_q) # 4
            features.extend(self.tf_features(q_words, pattern)) # 5 - 14
            features.extend(tf_idf_q) # 15 - 19
            features.append(bm25[i]) # 20
            features.append(float(cosine_pattern[0][i])) # 21
            features.append(float(cosine_title[0][i])) # 22
            features.append(float(cosine_excerpt[0][i])) # 23
            features.append(float(cosine_pattern_overflow[0][i])) # 24
            features.append(float(cosine_title_overflow[0][i])) # 25
            features.append(float(cosine_excerpt_overflow[0][i])) # 26

            features.append(binary_q[0])
            features.append(multi_q[0])
            features.append(binary_pattern[i])
            features.append(multi_pattern[i])

            # Append deep semantic interaction features: similarities
            features.append(float(deep_semantic_features["similarities"][0][0][i])) # pattern similarity
            features.append(float(deep_semantic_features["similarities"][1][0][i])) # title similarity
            features.append(float(deep_semantic_features["similarities"][2][0][i])) # excerpt similarity
            features.append(float(deep_semantic_features["similarities"][3][0][i])) # pattern similarity (overflow model)
            features.append(float(deep_semantic_features["similarities"][4][0][i])) # title similarity (overflow model)
            features.append(float(deep_semantic_features["similarities"][5][0][i])) # excerpt similarity (overflow model)

            # Append deep semantic interaction features: hadamard products
            features.extend(deep_semantic_features["hadamard_products"][0][i].tolist()) # pattern
            features.extend(deep_semantic_features["hadamard_products"][1][i].tolist()) # title
            features.extend(deep_semantic_features["hadamard_products"][2][i].tolist()) # excerpt
            features.extend(deep_semantic_features["hadamard_products"][3][i].tolist()) # pattern (overflow model)
            features.extend(deep_semantic_features["hadamard_products"][4][i].tolist()) # title (overflow model)
            features.extend(deep_semantic_features["hadamard_products"][5][i].tolist()) # excerpt (overflow model)

            # # Append deep semantic interaction features: concatenation
            features.extend(deep_semantic_features["concatenations"][0][i].tolist()) # pattern
            features.extend(deep_semantic_features["concatenations"][1][i].tolist()) # title
            features.extend(deep_semantic_features["concatenations"][2][i].tolist()) # excerpt
            features.extend(deep_semantic_features["concatenations"][3][i].tolist()) # pattern (overflow model)
            features.extend(deep_semantic_features["concatenations"][4][i].tolist()) # title (overflow model)
            features.extend(deep_semantic_features["concatenations"][5][i].tolist()) # excerpt (overflow model)

            features_all.append(features)

        return features_all

    def get_corpus_pattern(self):
        pattern_file= PARENT_FOLDER + "patterns.json"
        X = []
        title = []
        excerpt = []
        with open(pattern_file, 'r') as p:
            patterns = json.loads(p.read())

        for pattern in patterns:
            text = ""

            filename = pattern["filename"].replace(".md","").replace("-"," ")

            title.append(filename)
            excerpt.append(pattern["excerpt"].strip())

            text += filename
            if not text.endswith("."):
                text += ". "

            text += pattern["excerpt"].strip()
            if not text.endswith("."):
                text += ". "

            for heading in pattern["heading"]:
                text += heading["content"].strip()
                if not text.endswith("."):
                    text += ". "

            X.append(text)

        X_new, title_new, excerpt_new = self.get_new_patterns()

        X.extend(X_new)
        title.extend(title_new)
        excerpt.extend(excerpt_new)

        return X, title, excerpt


    def get_new_patterns(self):
      pattern_file= PARENT_FOLDER + "patterns_new.json"
      X = []
      title = []
      excerpt = []
      with open(pattern_file, 'r') as p:
          patterns = json.loads(p.read())

          for pattern in patterns:
            X.append(pattern["description"])
            title.append(pattern["title"])
            excerpt.append(pattern["excerpt"])

      return X, title, excerpt

    def remove_stopwords(self, q):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(q)
        filtered_sentence = " ".join([w for w in word_tokens if not w.lower() in stop_words])

        return filtered_sentence


    def initiate_tf_idf(self):
        self.tf_idf_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.tf_idf_vectorizer.fit(self.patterns)
        self.tf_idf_feature_names = self.tf_idf_vectorizer.get_feature_names_out()


    def initiate_bm25(self, b, k1):
        self.b = b
        self.k1 = k1

        y = super(TfidfVectorizer, self.tf_idf_vectorizer).transform(self.patterns)
        self.avdl = y.sum(1).mean()


    def bm25(self, q):
        X = self.patterns
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.tf_idf_vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.tf_idf_vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.tf_idf_vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

    def number_of_covered_words(self, q_words, pattern):
        # How many terms in the user query are covered by the text.
        # ration = Covered query term number divided by the number of query terms.

        n = 0
        for word in q_words:
            if word.lower() in pattern.lower():
                n += 1

        ratio = n/len(q_words)
        return [n, ratio]

    def get_idf(self, q_words):
        # 1 divided by the number of documents containing the query terms.

        n = 0
        word_in_patterns = set()
        for pattern in self.patterns:
            for word in q_words:
                if word.lower() in pattern.lower():
                    word_in_patterns.add(word.lower())

        if len(list(word_in_patterns)) == 0:
          return 0

        idf = 1/len(list(word_in_patterns))

        return idf

    def tf_features(self, q_words, pattern):
        # Sum, Min, Max, Average, Variance of counts of each query term in the document.
        # Normalized version : term counts divided by text length

        pattern_words = word_tokenize(pattern)
        total_len = len(pattern_words)
        n_count_all = [pattern_words.count(word) for word in q_words]

        tf_sum, tf_min, tf_max, tf_avg, tf_var = sum(n_count_all), min(n_count_all), max(n_count_all), np.average(n_count_all), np.var(n_count_all)

        norm_tf_sum, norm_tf_min, norm_tf_max, norm_tf_avg, norm_tf_var = sum(n_count_all)/total_len, min(n_count_all)/total_len, max(n_count_all)/total_len, np.average(n_count_all)/total_len, np.var(n_count_all)/float(total_len)

        return [tf_sum, tf_min, tf_max, tf_avg, tf_var, norm_tf_sum, norm_tf_min, norm_tf_max, norm_tf_avg, norm_tf_var]


    def tf_idf_features(self, q):
        tfidf_matrix= self.tf_idf_vectorizer.transform([q]).todense()
        feature_index = tfidf_matrix[0,:].nonzero()[1]
        tfidf_scores = zip([self.tf_idf_feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])

        word_scores = [score for score in dict(tfidf_scores).values()]

        tfidf_sum, tfidf_min, tfidf_max, tfidf_avg, tfidf_var = sum(word_scores), min(word_scores), max(word_scores), np.average(word_scores), np.var(word_scores)

        return [tfidf_sum, tfidf_min, tfidf_max, tfidf_avg, tfidf_var]

    def load_pattern_embeddings(self):
        # Load the embeddings from the saved files
        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern.pkl', 'rb') as f:
            self.emb_pattern = pickle.load(f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_title.pkl', 'rb') as f:
            self.emb_pattern_title = pickle.load(f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_excerpt.pkl', 'rb') as f:
            self.emb_pattern_excerpt = pickle.load(f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_overflow.pkl', 'rb') as f:
            self.emb_pattern_overflow = pickle.load(f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_title_overflow.pkl', 'rb') as f:
            self.emb_pattern_title_overflow = pickle.load(f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_excerpt_overflow.pkl', 'rb') as f:
            self.emb_pattern_excerpt_overflow = pickle.load(f)

    def precompute_pattern_embeddings(self):
        print("Precompute Pattern Embeddings")
        # Compute embeddings for patterns
        self.emb_pattern = self.model_sentence_transformer.encode(self.patterns, convert_to_tensor=True)
        self.emb_pattern_title = self.model_sentence_transformer.encode(self.pattern_titles, convert_to_tensor=True)
        self.emb_pattern_excerpt = self.model_sentence_transformer.encode(self.pattern_excerpts, convert_to_tensor=True)

        self.emb_pattern_overflow = self.model_sentence_transformer_overflow.encode(self.patterns, convert_to_tensor=True)
        self.emb_pattern_title_overflow = self.model_sentence_transformer_overflow.encode(self.pattern_titles, convert_to_tensor=True)
        self.emb_pattern_excerpt_overflow = self.model_sentence_transformer_overflow.encode(self.pattern_excerpts, convert_to_tensor=True)

        # Save the embeddings for later use
        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern.pkl', 'wb') as f:
            pickle.dump(self.emb_pattern, f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_title.pkl', 'wb') as f:
            pickle.dump(self.emb_pattern_title, f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_excerpt.pkl', 'wb') as f:
            pickle.dump(self.emb_pattern_excerpt, f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_overflow.pkl', 'wb') as f:
            pickle.dump(self.emb_pattern_overflow, f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_title_overflow.pkl', 'wb') as f:
            pickle.dump(self.emb_pattern_title_overflow, f)

        with open(PARENT_FOLDER + 'LTR_resources/emb_pattern_excerpt_overflow.pkl', 'wb') as f:
            pickle.dump(self.emb_pattern_excerpt_overflow, f)

    def semantic_similarity_features(self, q):
        emb_q = self.model_sentence_transformer.encode(q, convert_to_tensor=True)

        cosine_scores_pattern = util.cos_sim(emb_q, self.emb_pattern)
        cosine_scores_title = util.cos_sim(emb_q, self.emb_pattern_title)
        cosine_scores_excerpt = util.cos_sim(emb_q, self.emb_pattern_excerpt)

        emb_q = self.model_sentence_transformer_overflow.encode(q, convert_to_tensor=True)

        cosine_scores_pattern_overflow = util.cos_sim(emb_q, self.emb_pattern_overflow)
        cosine_scores_title_overflow = util.cos_sim(emb_q, self.emb_pattern_title_overflow)
        cosine_scores_excerpt_overflow = util.cos_sim(emb_q, self.emb_pattern_excerpt_overflow)

        return cosine_scores_pattern, cosine_scores_title, cosine_scores_excerpt, cosine_scores_pattern_overflow, cosine_scores_title_overflow, cosine_scores_excerpt_overflow

    def hadamard_product(self, tensor1, tensor2):
        return tensor1 * tensor2

    def deep_semantic_interaction_features(self, q):
        # Encode the query
        emb_q = self.model_sentence_transformer.encode(q, convert_to_tensor=True)
        emb_q_overflow = self.model_sentence_transformer_overflow.encode(q, convert_to_tensor=True)

        # Compute the "ideal" similarity, which is the query with itself
        ideal_emb = self.model_sentence_transformer.encode([q + " [SEP] " + q], convert_to_tensor=True)
        ideal_emb_overflow = self.model_sentence_transformer_overflow.encode([q + " [SEP] " + q], convert_to_tensor=True)

        # Compute Hadamard product between query and pattern embeddings
        hadamard_emb_pattern = self.hadamard_product(emb_q, self.emb_pattern)
        hadamard_emb_pattern_title = self.hadamard_product(emb_q, self.emb_pattern_title)
        hadamard_emb_pattern_excerpt = self.hadamard_product(emb_q, self.emb_pattern_excerpt)

        hadamard_emb_pattern_overflow = self.hadamard_product(emb_q_overflow, self.emb_pattern_overflow)
        hadamard_emb_pattern_title_overflow = self.hadamard_product(emb_q_overflow, self.emb_pattern_title_overflow)
        hadamard_emb_pattern_excerpt_overflow = self.hadamard_product(emb_q_overflow, self.emb_pattern_excerpt_overflow)

        # Compute Concatenation between query and pattern embeddings
        concat_emb_pattern = torch.cat((emb_q.unsqueeze(0), self.emb_pattern), dim=0)
        concat_emb_pattern_title = torch.cat((emb_q.unsqueeze(0), self.emb_pattern_title), dim=0)
        concat_emb_pattern_excerpt = torch.cat((emb_q.unsqueeze(0), self.emb_pattern_excerpt), dim=0)

        concat_emb_pattern_overflow = torch.cat((emb_q_overflow.unsqueeze(0), self.emb_pattern_overflow), dim=0)
        concat_emb_pattern_title_overflow = torch.cat((emb_q_overflow.unsqueeze(0), self.emb_pattern_title_overflow), dim=0)
        concat_emb_pattern_excerpt_overflow = torch.cat((emb_q_overflow.unsqueeze(0), self.emb_pattern_excerpt_overflow), dim=0)

        # Compute similarity between the query embeddings and the precomputed pattern embeddings
        similarities_pattern = util.pytorch_cos_sim(emb_q, self.emb_pattern)
        similarities_title = util.pytorch_cos_sim(emb_q, self.emb_pattern_title)
        similarities_excerpt = util.pytorch_cos_sim(emb_q, self.emb_pattern_excerpt)

        similarities_pattern_overflow = util.pytorch_cos_sim(emb_q_overflow, self.emb_pattern_overflow)
        similarities_title_overflow = util.pytorch_cos_sim(emb_q_overflow, self.emb_pattern_title_overflow)
        similarities_excerpt_overflow = util.pytorch_cos_sim(emb_q_overflow, self.emb_pattern_excerpt_overflow)

        # Returning features which include the similarities, the hadamard product embeddings, and concatenations
        return {
            "similarities": (similarities_pattern, similarities_title, similarities_excerpt,
                            similarities_pattern_overflow, similarities_title_overflow, similarities_excerpt_overflow),
            "hadamard_products": (hadamard_emb_pattern, hadamard_emb_pattern_title, hadamard_emb_pattern_excerpt,
                                  hadamard_emb_pattern_overflow, hadamard_emb_pattern_title_overflow, hadamard_emb_pattern_excerpt_overflow),
            "concatenations": (concat_emb_pattern, concat_emb_pattern_title, concat_emb_pattern_excerpt,
                              concat_emb_pattern_overflow, concat_emb_pattern_title_overflow, concat_emb_pattern_excerpt_overflow)
        }



    def predict_class(self, texts, model, vectorizer):
        loaded_model = pickle.load(open(model, 'rb'))
        loaded_vect = pickle.load(open(vectorizer, 'rb'))

        text = [preprocess_text(t, preprocess_functions) for t in texts]
        v_text = loaded_vect.transform(text)

        prediction = loaded_model.predict(v_text)

        return prediction

    def class_features(self, texts):
      PARENT_FOLDER = ""

      # BINARY CLASS
      binary_prediction = self.predict_class(texts, PARENT_FOLDER + "binary_nb_model.sav", PARENT_FOLDER + "binary_vectorizer_model.sav")

      # MULTI CLASS
      multi_prediction = self.predict_class(texts,PARENT_FOLDER + "multi_nb_model.sav",PARENT_FOLDER + "multi_vectorizer_model.sav")

      return binary_prediction, multi_prediction

def process_fold_data(fold_type, fold_num, pattern_file_path, base_path, cache={}):
    """
    Processes the given fold data (train or test) and writes the output to a file.

    Parameters:
    - fold_type (str): Either 'train' or 'test'.
    - fold_num (int): The fold number (1-5).
    - pattern_file_path (str): Path to the patterns file.
    - base_path (str): Base path for input and output files.
    """
    pp = PrivacyPatternFeatures()

    with open(pattern_file_path, 'r') as p:
        patterns = json.loads(p.read())

    pattern_name = [pattern["title"].replace(".md", "") for i, pattern in enumerate(patterns)]

    with open(base_path + f"{fold_type}_patterns_req_v2_fold_{fold_num}.json", 'r', encoding="utf-8") as p:
        patterns_requirements = json.loads(p.read())

    lines = []
    for pr in patterns_requirements:
        print("Query_Id", pr["id"])

        # Check if features are already calculated for this pr["id"]
        if pr["id"] not in cache:
            cache[pr["id"]] = pp.construct_features(pr["req_text"])

        features = cache[pr["id"]]

        for i_pattern, p in enumerate(pr["pattern"]):
            idx = pattern_name.index(p["name"])

            line = ""
            line += "{} qid:{}".format(p["rating"], pr["id"])

            for i_feature, val in enumerate(features[idx]):
                line += " {}:{}".format(i_feature+1, val)

            line += " #docid={}".format(p["name"])
            lines.append(line)

    # Save the processed lines to a file specific to the current fold and type (train/test)
    with open(base_path + f"{fold_type}_fold_{fold_num}.txt", "w") as f:
        for l in lines:
            f.write(l + "\n")


# Example usage:
base_path = ""
pattern_file = base_path + "patterns.json"

# Initialize a cache dictionary to store features
features_cache = {}

# Process all 5 folds for both training and testing data
for fold_num in range(1, 6):
    process_fold_data('train', fold_num, pattern_file, base_path, features_cache)
    process_fold_data('test', fold_num, pattern_file, base_path, features_cache)
