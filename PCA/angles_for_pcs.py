import numpy as np
import os
import pickle
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = "C:\\Users\\MindRove_BZs\\PipeHand\\tmp_hand_model\\tmp_hand_model\\data\\landmarks\\"
FILE_NAMES = ["landmarks_hunco_21.pickle", "landmarks_rhd.pickle", "landmarks_h2o3d.pickle", "landmarks_contactpose.pickle"]

def rotate_to_align(points, p1_idx=0, p2_idx=11):
    points = points.astype(float)  
    p1 = points[p1_idx]
    p2 = points[p2_idx]
    vec = p2 - p1
    vec /= np.linalg.norm(vec)

    target_vec = np.array([0, 1, 0])
    axis = np.cross(vec, target_vec)
    angle = np.arccos(np.dot(vec, target_vec))
    axis /= np.linalg.norm(axis)

    rotation = R.from_rotvec(axis * angle).as_matrix()
    rotated_points = np.dot(points - p1, rotation) + p1

    return rotated_points


def calculate_invariants(all_points):
    all_return_arr = []

    for points in all_points:
        rotated_points = rotate_to_align(points)
        points = rotated_points

        origin = points[0]
        x_axis = points[1] - origin
        x_axis /= np.linalg.norm(x_axis)

        temp_vec = points[2] - origin
        z_axis = np.cross(x_axis, temp_vec)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis])

        centered_points = points - origin
        local_points = np.dot(centered_points, rotation_matrix.T)

        ref_distance = np.linalg.norm(local_points[1])

        distances = np.linalg.norm(centered_points[1:], axis=1) / ref_distance
        azimuths = []
        elevations = []

        for i in range(1, len(local_points)):
            x, y, z = local_points[i]
            azimuth = np.arctan2(y, x)
            elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
            azimuths.append(azimuth)
            elevations.append(elevation)

        azimuths = np.array(azimuths)[..., np.newaxis]
        elevations = np.array(elevations)[..., np.newaxis]
        distances = distances[..., np.newaxis]

        return_arr = np.concatenate([azimuths, elevations, distances], axis=-1)
        all_return_arr.append(return_arr)

    all_return_arr = np.array(all_return_arr)
    print(f"Shape: {all_return_arr.shape}")

    return all_return_arr


def convert_angles_to_tfrecord(landmark_set, target_filename, per_segment = 5000, stat_fname="tfrecord_stats.pickle"):

    print(f"Input landmarks` shape {landmark_set.shape}")

    min_val = np.ones((1,20,1)) * np.inf
    max_val = np.full((1, 20, 1), -np.inf)

    with tf.io.TFRecordWriter(target_filename) as writer:
        print(landmark_set.shape[0])
        
        for i in range(0, landmark_set.shape[0], per_segment):
            print(min(i+per_segment, landmark_set.shape[0]))
            res_arr = calculate_invariants(landmark_set[i:min(i+per_segment, landmark_set.shape[0])])
            print(f"res_arr shape: {res_arr.shape}")
            if np.sum(np.isnan(res_arr)) > 0:
                print(f"Number of NaNs in res_arr: {np.sum(np.isnan(res_arr))}")
            min_val[:,:,0] = np.min(np.concatenate([min_val[:,:,0], res_arr[:,:,2]], axis=0), axis=0, keepdims=True)
            max_val[:,:,0] = np.max(np.concatenate([max_val[:,:,0], res_arr[:,:,2]], axis=0), axis=0, keepdims=True) 

            for sample in res_arr:
                feature = {
                    'landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=sample.reshape((-1,))))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    min_val  = np.min(min_val, axis=0, keepdims=True)
    max_val  = np.max(max_val, axis=0, keepdims=True)

    dataset_stats = {"min_val" : min_val, "max_val": max_val}

    with open(stat_fname, "wb") as f:
        pickle.dump(dataset_stats, f)

def load_and_process_data(file_name, base_dir):
    print(f"Loading {file_name}...")
    with open(os.path.join(base_dir, file_name), "rb") as f:
        lndmrks = pickle.load(f)
    lndmrks = np.array(lndmrks)
    if file_name == "landmarks_rhd.pickle":
        lndmrks = np.concatenate([lndmrks[:, :21, :], lndmrks[:, 21:, :]], axis=0)
    print(f"{file_name} dataset size: {lndmrks.shape[0]}")
    return lndmrks, [file_name]*lndmrks.shape[0]

def main():
    all_landmarks, all_file_names = zip(*[load_and_process_data(file_name, BASE_DIR) for file_name in FILE_NAMES])
    all_landmarks = np.concatenate(all_landmarks, axis=0)
    all_file_names = np.concatenate(all_file_names, axis=0)

    print(f"All datasets: {all_landmarks.shape}")
    np.random.shuffle(all_landmarks)

    convert_angles_to_tfrecord(all_landmarks, "C:\\Users\\MindRove_BZs\\Diploma\\all_angles.tfrecord")

if __name__ == "__main__":
    main()