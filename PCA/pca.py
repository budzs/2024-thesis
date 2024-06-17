
import json
import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA


BASE_PATH = "C:\\Users\\MindRove_BZs\\Diploma\\"
TFRECORD_STATS_FILE = Path(os.path.join(BASE_PATH, "tfrecord_stats.pickle"))
TFRECORD_FILE = Path(os.path.join(BASE_PATH, "all_angles.tfrecord"))
MODEL_FILE = "hand_model_all_1.pkl"
BATCH_SIZE = 1000
DATASET_SIZE = 13000000

def parse_tfrecord_fn(example_proto, len=400):
    feature_description = {
        'landmarks': tf.io.FixedLenFeature((60,), tf.float32),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    features = tf.reshape(features["landmarks"], (20,3))

    return features
def load_tfrecord_to_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = dataset.map(parse_tfrecord_fn)
    
    return parsed_dataset

def reshape_sample(sample):
    return tf.reshape(sample, (60,))

def remove_outliers_z_score(data, threshold=1):
    z_scores = np.abs(stats.zscore(data))
    return data[(z_scores < threshold).all(axis=1)]


def load_and_prepare_data(test_size=0.1):
    with TFRECORD_STATS_FILE.open("rb") as f:
        dataset_stats = pickle.load(f)
    dataset = load_tfrecord_to_dataset(TFRECORD_FILE)
    #dataset = dataset.map(lambda sample: normalize_sample(sample, dataset_stats))
    dataset = dataset.map(reshape_sample)
    dataset = dataset.batch(DATASET_SIZE)
    all_data = next(iter(dataset)).numpy()
    #all_data = remove_outliers_z_score(all_data)
    train_data, test_data = train_test_split(all_data, test_size=test_size)
    return train_data, test_data, dataset_stats


def perform_pca(data, n_components):
    pca = IncrementalPCA(n_components=n_components, batch_size=BATCH_SIZE)
    pca.fit(data)
    return pca

def parameter_search(data, param_values, target_variance = 1):
    results = []
    for value in param_values:
        pca = perform_pca(data, value)
        total_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Number of components: {value}, Total explained variance: {total_variance}")
        results.append((value, total_variance))
        if total_variance >= target_variance:
            break
    best_result = next((result for result in results if result[1] >= target_variance), results[-1])
    
    return results, *best_result, perform_pca(data, best_result[0])

def save_model(pca, dataset_stats, mu, n_components):
    model_file = f"hand_model_all_{n_components}.pkl"
    stats = {
        'mu': mu.tolist(),
        'principal_components': pca.components_.tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'singular_values': pca.singular_values_.tolist(),
    }
    pickle.dump({"pca": pca, "norm_factors": dataset_stats, "pca_stats": stats}, open(model_file, "wb"))

    dataset_stats = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in dataset_stats.items()}

    stats = {
        'mu': mu.tolist(),
        'principal_components': pca.components_.tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'singular_values': pca.singular_values_.tolist(),
    }
    with open(model_file.replace('.pkl', '.json'), 'w') as f:
        json.dump({"pca_stats": stats, "norm_factors": dataset_stats}, f)



def reconstruct_sample(pca, all_data, mu, stats, name="train"):
    sample = all_data[:100]
    reduced = pca.transform(sample)
    recon = pca.inverse_transform(reduced)
    diff = np.sum((recon - sample) **2)
    total = np.sum(sample ** 2)
    percent_diff = (diff / total) * 100
    print(f"Percentage difference in {name}: {percent_diff}%")
    mse = np.mean((recon - sample) ** 2)
    print(f"Mean Squared Error in {name}: {mse}")


def plot_data(pca, data):
    reduced = pca.transform(data)
    plt.scatter(reduced[:, 0], reduced[:, 1], c="b")
    plt.title("PCA of the data")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.show()
    plt.bar(list(range(len(pca.explained_variance_ratio_))), pca.explained_variance_)
    plt.show()

def plot_data2(pca):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.title('Scree Plot of PCA')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

def plot_parameter_search(results):
    components, variances = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(components, variances, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.title('Total explained variance vs Number of components')
    plt.grid(True)
    plt.yscale('log')  
    plt.show()

def main():
    print(f"TFRecord file exists: {TFRECORD_FILE.exists()}, size: {TFRECORD_FILE.stat().st_size} bytes")
    train_data, test_data, dataset_stats = load_and_prepare_data()
    print("After load_and_prepare_data:", np.any(np.isnan(train_data)))
    print("Number of NaN values:", np.sum(np.isnan(train_data)))
    target_variance = 1.2
    for n_components in range(2, 61):
        print(f"Number of components: {n_components}")
        pca = perform_pca(train_data, n_components)
        mu = np.mean(train_data, axis=0)
        save_model(pca, dataset_stats, mu, n_components)

        reconstruct_sample(pca, train_data, mu, dataset_stats)
        reconstruct_sample(pca, test_data, mu, dataset_stats, "test")

        if False:
            plot_data(pca, train_data)
            plot_data2(pca)
            plot_data(pca, test_data)



if __name__ == "__main__":
    main()