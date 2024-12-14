import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

class CustomNaiveBayes:
    def __init__(self):
        self.class_summaries = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        data = np.c_[X, y]
        self.class_summaries = self._mean_stddev_per_class(data)

    def _mean_stddev_per_class(self, data):
        info = {}
        for class_value in self.classes:
            instances = data[data[:, -1] == class_value]
            features = instances[:, :-1]
            means = np.mean(features, axis=0)
            stdevs = np.std(features, axis=0)
            info[class_value] = (means, stdevs)
        return info

    def _calculate_gaussian_probability(self, x, means, stdevs):
        exponent = np.exp(-((x - means) ** 2) / (2 * (stdevs ** 2 + 1e-10)))
        result = exponent / (np.sqrt(2 * np.pi) * (stdevs + 1e-10))
        return result

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            x = X[i]
            log_probs = {}
            for class_value in self.classes:
                means, stdevs = self.class_summaries[class_value]
                probs = self._calculate_gaussian_probability(x, means, stdevs)
                log_probs[class_value] = np.sum(np.log(probs + 1e-10))
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

    def predict_proba(self, X):
        probabilities = []
        for i in range(X.shape[0]):
            x = X[i]
            class_probs = {}
            for class_value in self.classes:
                means, stdevs = self.class_summaries[class_value]
                probs = self._calculate_gaussian_probability(x, means, stdevs)
                log_prob = np.sum(np.log(probs + 1e-10))
                class_probs[class_value] = log_prob
            total_log_prob = np.logaddexp.reduce(list(class_probs.values()))
            probs_exp = {k: np.exp(v - total_log_prob) for k, v in class_probs.items()}
            probabilities.append(probs_exp)
        return probabilities

    def accuracy(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return (correct / len(y_true)) * 100.0

    def log_loss(self, y_true, y_pred_proba):
        eps = 1e-15
        n_classes = len(self.classes)
        y_true_one_hot = np.eye(n_classes)[y_true.astype(int)]
        y_pred_proba_array = np.array([[proba.get(c, eps) for c in self.classes] for proba in y_pred_proba])
        y_pred_proba_array = np.clip(y_pred_proba_array, eps, 1 - eps)
        loss = -np.sum(y_true_one_hot * np.log(y_pred_proba_array)) / len(y_true)
        return loss

# Data loading and preprocessing
script_dir = os.path.dirname(os.path.abspath(__file__))

additional_features_path = os.path.join(script_dir, '..', 'dataset', 'train', 'additional_features_train.csv')
basic_features_path = os.path.join(script_dir, '..', 'dataset', 'train', 'basic_features_train.csv')
content_features_path = os.path.join(script_dir, '..', 'dataset', 'train', 'content_features_train.csv')
flow_features_path = os.path.join(script_dir, '..', 'dataset', 'train', 'flow_features_train.csv')
labels_path = os.path.join(script_dir, '..', 'dataset', 'train', 'labels_train.csv')
time_features_path = os.path.join(script_dir, '..', 'dataset', 'train', 'time_features_train.csv')

additional_features_df = pd.read_csv(additional_features_path)
basic_features_df = pd.read_csv(basic_features_path)
content_features_df = pd.read_csv(content_features_path)
flow_features_df = pd.read_csv(flow_features_path)
labels_df = pd.read_csv(labels_path)
time_features_df = pd.read_csv(time_features_path)

data = pd.merge(basic_features_df, additional_features_df, on="id")
data = pd.merge(data, content_features_df, on="id")
data = pd.merge(data, flow_features_df, on="id")
data = pd.merge(data, labels_df, on="id")
data = pd.merge(data, time_features_df, on="id")

##### PREPROCESSING ASAL AEK #####
le = LabelEncoder()
data['attack_cat_encoded'] = le.fit_transform(data['attack_cat'].astype(str))

if 'label' in data.columns:
    data = data.drop(columns=['label'])

columns = list(data.columns)
columns.remove('attack_cat_encoded')
columns.append('attack_cat_encoded')
data = data[columns]

data = data.select_dtypes(include=[np.number])
data = data.dropna()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.astype(int)

ratio = 0.7
train_num = int(len(X) * ratio)
indices = np.random.permutation(len(X))
train_indices = indices[:train_num]
test_indices = indices[train_num:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print('Total number of examples:', len(X))
print('Training examples:', len(X_train))
print('Test examples:', len(X_test))

##### TRAINING MODEL #####
model = CustomNaiveBayes()
model.fit(X_train, y_train)

# Save model to .pkl
with open('trained_naive_bayes.pkl', 'wb') as file:
    pickle.dump(model, file)

print('Model telah dilatih dan disimpan ke trained_naive_bayes.pkl')

# Load model from .pkl
def load_trained_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_trained_model('trained_naive_bayes.pkl')

print('Model telah dimuat dari trained_naive_bayes.pkl')

##### EVALUATING MODEL #####
y_pred = model.predict(X_test)

# Accuracy
accuracy = model.accuracy(y_test, y_pred)
print('Akurasi model:', accuracy)

# Calculate Log Loss
y_pred_proba = model.predict_proba(X_test)
logloss = model.log_loss(y_test, y_pred_proba)
print('Log loss model:', logloss)
