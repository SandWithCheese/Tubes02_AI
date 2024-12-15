import pandas as pd
import sklearn.metrics
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

class KNN:
    def __init__(self, _k : int, _distance_metric : str, _p = 3):
        self.k : int = _k
        self.distance_function = self._set_distance_function(_distance_metric.lower())
        self.p = _p
        self.X_train = None
        self.Y_train = None

        self.i = 0

    def _set_distance_function(self, input_metric):
        distance_functions = {
            "euclidean": self._euclidean,
            "manhattan": self._manhattan,
            "minkowski": self._minkowski
        }
        if input_metric not in distance_functions:
            raise ValueError(f"Unsupported distance metric: {input_metric}")
        return distance_functions[input_metric]
    
    def _euclidean(self, a, b):
        self.i += 1
        print(self.i)
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def _manhattan(self, a, b):
        return np.sum(np.abs(a - b), axis=1)

    def _minkowski(self, a, b):
        return np.sum(np.abs(a - b) ** self.p, axis=1) ** (1 / self.p)
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.Y_train = np.array(y)

    # def predict(self, X):
    #     X = np.array(X)
    #     predictions = [self._predict_single(x) for x in X]
    #     return np.array(predictions)

    # def _predict_single(self, x):
    #     # distances = self.distance_function(self.X_train, x)
    #     distances = np.array([self.distance_function(x, x_train) for x_train in self.X_train])
    #     k_nearest_neighbors_index = np.argsort(distances)[:self.k]
    #     k_nearest_neighbors_labels = self.Y_train[k_nearest_neighbors_index]

    #     self.i += 1
    #     print(self.i)

    #     return np.bincount(k_nearest_neighbors_labels).argmax()

    def predict(self, X):
        X = np.array(X)
        distances = np.apply_along_axis(self.distance_function, 1, X, self.X_train)
        k_nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        k_nearest_labels = self.Y_train[k_nearest_indices]
        predictions = np.apply_along_axis(self.majority_vote, 1, k_nearest_labels)
        return predictions
    
    def majority_vote(self, neighbors):
        return np.bincount(neighbors).argmax()
    
    def accuracy(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return (correct / len(y_true)) * 100.0



##### --- DATA LOADING --- ######
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

##### --- PREPROCESS --- ######
RATIO = 0.8

# le = LabelEncoder()
# data['attack_cat_encoded'] = le.fit_transform(data['attack_cat'].astype(str))
# columns = list(data.columns)
# columns.remove('attack_cat_encoded')
# columns.append('attack_cat_encoded')
# data = data[columns]

label_encoder = LabelEncoder()

# Fit and transform the nominal column
data['attack_cat_encoded'] = label_encoder.fit_transform(data['attack_cat'])

# Drop the original nominal column
data = data.drop(columns=['attack_cat'])

if 'label' in data.columns:
    data = data.drop(columns=['label'])

# data = data.select_dtypes(include=[np.number])
data = data.dropna()

binary_columns = ['is_sm_ips_ports', 'is_ftp_login']
nominal_columns = ['proto', 'state', 'service']
numeral_columns = [c for c in data.columns if c not in binary_columns and c not in nominal_columns and c!='attack_cat_encoded']

# Initialize OneHotEncoder
onehot_encoder = OneHotEncoder(drop='if_binary', sparse_output=False)

# Fit and transform binary and nominal columns
binary_encoded = onehot_encoder.fit_transform(data[binary_columns])
binary_encoded_names = onehot_encoder.get_feature_names_out(binary_columns)

nominal_encoded = onehot_encoder.fit_transform(data[nominal_columns])
nominal_encoded_names = onehot_encoder.get_feature_names_out(nominal_columns)

# Convert to DataFrames
binary_encoded_df = pd.DataFrame(binary_encoded, columns=binary_encoded_names)
nominal_encoded_df = pd.DataFrame(nominal_encoded, columns=nominal_encoded_names)

# 2. Normalize numeral columns
scaler = MinMaxScaler()

# Fit and transform numeral columns
numeral_scaled = scaler.fit_transform(data[numeral_columns])

# Convert to DataFrame
numeral_scaled_df = pd.DataFrame(numeral_scaled, columns=numeral_columns)

# 3. Combine all processed columns
# Drop original columns
data = data.drop(columns=binary_columns + nominal_columns + numeral_columns).reset_index(drop=True)

# Concatenate processed columns
data = pd.concat([data, binary_encoded_df, nominal_encoded_df, numeral_scaled_df], axis=1)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(len(data) * RATIO)
train_df = data[:train_size]
test_df = data[train_size:]

X_train = train_df.drop(columns=['attack_cat_encoded'])
Y_train = train_df['attack_cat_encoded']
X_test = test_df.drop(columns=['attack_cat_encoded'])
Y_test = test_df['attack_cat_encoded']

print('Training examples:', len(X_train))
print('Test examples:', len(X_test))

##### --- TRAINING MODEL --- ######
knn_model = KNN(131, "euclidean")
knn_model.fit(X_train, Y_train)

##### --- EVALUATING MODEL --- #####
Y_prediction = knn_model.predict(X_test)
f1_score = sklearn.metrics.f1_score(Y_test, Y_prediction, average='macro')
# accuracy = knn_model.accuracy(Y_test, Y_prediction)
print('Model Accuracy: ', f1_score)