import random
import pandas as pd
import numpy as np
import os
import pickle

def encode_class(data):
    classes = []
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes.append(data[i][-1])
    for i in range(len(classes)):
        for j in range(len(data)):
            if data[j][-1] == classes[i]:
                data[j][-1] = i
    return data

def split_data(data, ratio):
    train_num = int(len(data) * ratio)
    train = []
    test = list(data)
    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train, test

def group_by_class(data):
    data_dict = {}
    for i in range(len(data)):
        if data[i][-1] not in data_dict:
            data_dict[data[i][-1]] = []
        data_dict[data[i][-1]].append(data[i])
    return data_dict

def mean_std_dev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev

def mean_std_dev_per_class(data):
    info = {}
    data_dict = group_by_class(data)
    for class_value, instances in data_dict.items():
        transposed_instances = list(zip(*instances))
        feature_stats = [mean_std_dev(feature) for feature in transposed_instances[:-1]]
        info[class_value] = feature_stats
    return info

def calculate_gaussian_probability(X, means, stdevs):
    epsilon = 1e-10
    means = np.array(means)
    stdevs = np.array(stdevs)
    probabilities = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        exponent = -((X[:, j] - means[j])**2) / (2 * (stdevs[j] + epsilon)**2)
        probabilities[:, j] = np.exp(exponent) / (np.sqrt(2 * np.pi) * (stdevs[j] + epsilon))
    return probabilities

def get_predictions(info, test_data):
    test_array = np.array(test_data)
    X = test_array[:, :-1]
    log_class_probs = {}
    for class_value, class_summaries in info.items():
        means = [summary[0] for summary in class_summaries]
        stdevs = [summary[1] for summary in class_summaries]
        feature_probs = calculate_gaussian_probability(X, means, stdevs)
        log_class_probs[class_value] = np.sum(np.log(feature_probs + 1e-10), axis=1)
    prob_array = np.array(list(log_class_probs.values())).T
    predictions = [list(info.keys())[np.argmax(probs)] for probs in prob_array]
    return predictions

def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0

# Main Test Driver
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

# buat testing, pake yang numerikal aja dan hapus yang nan
data = data.select_dtypes(include=[np.number])
data = data.dropna()

data_list = data.values.tolist()
for i in range(len(data_list)):
    for j in range(len(data_list[i]) - 1):
        data_list[i][j] = float(data_list[i][j])

ratio = 0.7 # ini ntar atur atur lah ya
train_data, test_data = split_data(data_list, ratio)

print('Total number of examples:', len(data_list))
print('Training examples:', len(train_data))
print('Test examples:', len(test_data))

#* training
info = mean_std_dev_per_class(train_data)

# save model ke .pkl
with open('trained_naive_bayes.pkl', 'wb') as file:
    pickle.dump(info, file)

print('Model telah dilatih dan disimpan ke trained_naive_bayes.pkl')

# load model dari .pkl
def load_trained_model(filename):
    with open(filename, 'rb') as file:
        info = pickle.load(file)
    return info

info = load_trained_model('trained_naive_bayes.pkl')

print('Model telah dimuat dari trained_naive_bayes.pkl')

# prediksi
predictions = get_predictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print('Akurasi model:', accuracy)
