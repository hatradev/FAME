import numpy as np
import pandas as pd

from scipy.ndimage import shift
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

## Data preperation functions
def read_x(data_dir, header='close'):
    dataset = pd.read_csv(data_dir)[[header]].dropna()
    return dataset.values

def split_dataset(data, percentages=(0.5, 0.25, 0.25)):
    if not np.isclose(np.sum(percentages), 1.0):
        raise ValueError("Sum of percentages must be 1.0")
    n = len(data)
    idx1 = int(n * percentages[0])
    idx2 = int(n * (percentages[0] + percentages[1]))
    return data[:idx1], data[idx1:idx2], data[idx2:]

def add_partial_column(df, col_name, values, start_index):
    col = pd.Series([np.nan] * len(df))
    col.iloc[start_index:start_index + len(values)] = values
    df[col_name] = col
    return df

def return_data_with_lagged_arrays(array, lag):
    temp=array
    shape=array.shape[0]
    temp=temp.reshape(shape, -1)
    for i in range(lag):
        temp_shifted=shift(array, i+1, cval=np.NaN, prefilter=True)
        temp=np.concatenate((temp,temp_shifted.reshape(shape,-1)), axis=1)
    final=temp[lag:len(temp)]
    return final

def determine_direction(series, after=1):
    direction=[]
    for i in range(len(series)):
        if (i < after):
            direction.append(np.nan)
        elif(series[i]>=series[i-after]):
            direction.append(1)
        else:
            direction.append(0)

    return np.array(direction)

def determine_difference(series, after=1):
    difference=[]
    for i in range(len(series)):
        if (i < after):
            difference.append(np.nan)
        else:
            difference.append(series[i]-series[i-after])
    return np.array(difference)

def create_time_vector(num_samples, fs: float = 1.0, t0: float = 0.0):
    if fs <= 0:
        raise ValueError("Sampling frequency (fs) must be positive.")
    if num_samples <= 0:
        raise ValueError("Number of samples (num_samples) must be positive.")

    dt = 1.0 / fs
    t_end = t0 + (num_samples - 1) * dt
    t = np.linspace(t0, t_end, num_samples)
    return t

def calculate_imfs_metrics(imfs):
    num_imfs, num_samples = imfs.shape
    t = create_time_vector(num_samples)
    dt = t[1] - t[0]
    fs = 1.0 / dt

    metrics = []
    for i, imf in enumerate(imfs):
        try:
            analytic_signal = hilbert(imf)
            envelope = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            inst_freq = np.gradient(phase, dt) / (2 * np.pi)

            valid = (inst_freq >= 0) & (inst_freq <= fs / 2)
            freqs = inst_freq[valid]
            amps = envelope[valid]
            weights = amps**2

            avg_freq = np.average(freqs, weights=weights) if weights.sum() > 1e-12 else np.mean(freqs) if len(freqs) else np.nan
            num_cycles = max((phase[-1] - phase[0]) / (2 * np.pi), 0)
            avg_amp = np.nan_to_num(np.mean(envelope), nan=0.0)

            metrics.append({
                'IMF': i,
                'Average Frequency (Hz)': avg_freq,
                'Number of Cycles': num_cycles,
                'Average Amplitude': avg_amp
            })
        except Exception as e:
            print(f"Error processing IMF {i}: {e}. Assigning NaN.")
            metrics.append({
                'IMF': i,
                'Average Frequency (Hz)': np.nan,
                'Number of Cycles': np.nan,
                'Average Amplitude': np.nan
            })

    return pd.DataFrame(metrics).set_index('IMF')

def label_imf(imfs_metrics, features_name=['Average Frequency (Hz)', 'Number of Cycles'], n_clusters=3, random_state=42):
    if n_clusters != 3:
        print(f"Warning: Current labeling logic (long/mid/short term) is designed for n_clusters=3. You provided {n_clusters}.")

    df = imfs_metrics.copy()
    if 'Original Index' not in df.columns:
        df['Original Index'] = df.index

    features = df[features_name]
    if features.isnull().values.any():
        print("Warning: Features contain NaN values. Filling with mean.")
        features = features.fillna(features.mean())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    indices = np.linspace(0, len(scaled_features) - 1, n_clusters, dtype=int)
    init_points = scaled_features[indices]
    kmeans = KMeans(n_clusters=n_clusters, init=init_points, n_init=1, random_state=random_state)
    cluster_labels = kmeans.fit_predict(scaled_features)
    df['Cluster'] = cluster_labels

    try:
        cluster_avg_freq = df.groupby('Cluster')[features_name[0]].mean()
        sorted_clusters = cluster_avg_freq.sort_values()

        label_map = {
            sorted_clusters.index[0]: "long_term",
            sorted_clusters.index[1]: "mid_term",
            sorted_clusters.index[2]: "short_term"
        }

        result_dict = {
            term_label: df[df['Cluster'] == cluster_idx]['Original Index'].tolist()
            for cluster_idx, term_label in label_map.items()
        }

    except Exception as e:
        print(f"Error during cluster labeling: {e}")
        return None

    return result_dict