import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import slic

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io, color
from scipy import ndimage as ndi
from skimage.feature import greycomatrix, greycoprops
import matplotlib.image as mpimg



import warnings
warnings.filterwarnings("ignore")

class KMeans:
    def __init__(self, n_clusters=3, max_iters=10000):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            self.labels = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = self._calculate_centroids(X, self.labels)

            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        return self.centroids, self.labels

    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _calculate_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
        return centroids

def extract_rgb(img):
    # Extract R, G, B channels
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1] 
    b_channel = img[:, :, 2] 

    # Display or use the individual channels as needed
    rgb_df = pd.DataFrame({'r': r_channel.flatten(),
                            'g': g_channel.flatten(), 'b': b_channel.flatten()})
    return rgb_df, b_channel, g_channel, r_channel

def extract_rgbxy(img):
    height, width, _ = img.shape
    x = np.zeros(height*width)
    y = np.zeros(height*width)

    _, b_channel, g_channel, r_channel = extract_rgb(img)

    for i in range(height):
        for j in range(width):
            x[i*width+j] = i
            y[i*width+j] = j


    b_channel = b_channel.reshape(height*width)
    g_channel = g_channel.reshape(height*width)
    r_channel = r_channel.reshape(height*width)

    df = pd.DataFrame({'x': x, 'y': y, 'b': b_channel, 'g': g_channel, 'r': r_channel})
    df['x'] = ((df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min()))*255
    df['y'] = ((df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min()))*255

    return df

def extract_segments(img, num_segments = 100):
    segments = slic(img, n_segments=num_segments, compactness=10)
    return segments

def extract_superpixel_rgb_mean(img, num_segments):
    height, width, _ = img.shape
    df_bgr, _, _, _ = extract_rgb(img)
    segments = extract_segments(img, num_segments)
    df_bgr["label"] = segments.reshape(height*width)
    
    df_b_mean = df_bgr[["b", "label"]].groupby('label').mean()
    df_g_mean = df_bgr[["g", "label"]].groupby('label').mean()
    df_r_mean = df_bgr[["r", "label"]].groupby('label').mean()

    df_bgr_mean = pd.concat([df_b_mean, df_g_mean, df_r_mean], axis=1)
    df_bgr_mean.columns = ['b_mean', 'g_mean', 'r_mean']

    return df_bgr_mean


def extract_superpixel_rgb_histogram(img, num_segments):
    height, width, _ = img.shape
    segments = extract_segments(img, num_segments)

    _, b_channel, g_channel, r_channel = extract_rgb(img)

    b_channel = b_channel.reshape(height*width)
    g_channel = g_channel.reshape(height*width)
    r_channel = r_channel.reshape(height*width)
    df_bgr = pd.DataFrame({'b': b_channel, 'g': g_channel, 'r': r_channel})
    df_bgr["label"] = segments.reshape(height*width)


    df_hist = pd.DataFrame()
    df_hist['b_hist'] = np.histogram(b_channel, bins=256)[0]
    df_hist['g_hist'] = np.histogram(g_channel, bins=256)[0]
    df_hist['r_hist'] = np.histogram(r_channel, bins=256)[0]

    # Get unique labels
    unique_labels, _ = np.unique(df_bgr['label'], return_counts=True)

    segment_arr = []

    for label in unique_labels:
        b = np.histogram(df_bgr[df_bgr['label'] == label]['b'], bins=256)[0]
        g = np.histogram(df_bgr[df_bgr['label'] == label]['g'], bins=256)[0]
        r = np.histogram(df_bgr[df_bgr['label'] == label]['r'], bins=256)[0]
    
        array_stack = np.vstack((b, g, r))
        transpose_of_array_stack = array_stack.T
        segment_arr.append(transpose_of_array_stack)

    return segment_arr

def extract_superpixel_mean_gabor(img, num_segments):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(img)

    # Perform SLIC superpixel segmentation
    segments = slic(img, n_segments=num_segments, compactness=10)

    # Define Gabor filter parameters
    orientations = 8
    scales = [0.1, 0.5] 

    # Initialize an array to store Gabor filter responses
    gabor_responses = np.zeros((len(scales) * orientations, np.max(segments) + 1))
    # Inside the Gabor filter loop
    for scale_idx, scale in enumerate(scales):
        for orientation in range(orientations):
            gabor_kernel = cv2.getGaborKernel(
                (3, 3), sigma=scale, theta=orientation * np.pi / orientations, lambd=5, gamma=0.5, psi=0)
            gabor_filtered = cv2.filter2D(gray_image, cv2.CV_64F, gabor_kernel)
            gabor_filtered = np.where(gabor_filtered == 0, 1e-6, gabor_filtered)  # Avoid division by zero or very small values
            gabor_responses[scale_idx * orientations + orientation] = ndi.mean(gabor_filtered, labels=segments, index=np.arange(0, np.max(segments) + 1))
            
            
    gabor_responses = gabor_responses[:, 1:]  # Remove the first column
    numpy_gabor_responses_array = np.array(gabor_responses.T)
    return numpy_gabor_responses_array

def rgb_pixel_level_model(img, path, k=3):
    rgb_df, _, _, _ = extract_rgb(img)
    # Converting DataFrame to NumPy array
    numpy_array = rgb_df.values
    kmeans = KMeans(n_clusters=k)
    centroids, labels = kmeans.fit(numpy_array)
    segmented_data = centroids[labels.flatten()]
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))
    
    cv2.imwrite(path, segmented_image)

def rgbxy_pixel_level_model(img, path, k=3):
    df = extract_rgbxy(img)

    # Converting DataFrame to NumPy array
    numpy_array = df.values
    kmeans = KMeans(n_clusters=k)
    centroids, labels = kmeans.fit(numpy_array)

    segmented_data = centroids[labels.flatten()]
    segmented_data = segmented_data[:,:3]
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))
    cv2.imwrite(path, segmented_image)

def superpixel_rgb_mean_model(img, path, k=3, n = 100):

    df_bgr_mean = extract_superpixel_rgb_mean(img, n)
    segments = extract_segments(img, n)
    # Converting DataFrame to NumPy array
    numpy_array = df_bgr_mean.values
    kmeans = KMeans(n_clusters=k)
    _, labels = kmeans.fit(numpy_array)

    num_labels = len(np.unique(labels))
    colors = plt.cm.get_cmap('tab20', num_labels) 
    label_color_mapping = {label: colors(label) for label in np.unique(labels)}

    color_image = np.zeros((segments.shape[0], segments.shape[1], 3))

    for segment_value, label_value in zip(np.unique(segments), labels):
        color_image[segments == segment_value] = label_color_mapping[label_value][:3]  # Extract RGB values
    

    mpimg.imsave(path, color_image)

def superpixel_rgb_histogram_model(img, path, k=3, n = 100):
    segment_arr = extract_superpixel_rgb_histogram(img, n)
    segments = extract_segments(img, n)

    # Reshape the segment histograms into a suitable format for K-means
    # Convert the list to a NumPy array
    segment_arr = np.array(segment_arr)
    segment_number = segments.max()
    print(segment_number)
    reshaped_histograms = segment_arr.reshape(segment_number, -1)  # Reshape to 75 segments

    kmeans = KMeans(n_clusters=k)
    _, labels = kmeans.fit(reshaped_histograms)

    num_labels = len(np.unique(labels))
    colors = plt.cm.get_cmap('tab20', num_labels)  # Using a colormap for distinct colors
    label_color_mapping = {label: colors(label) for label in np.unique(labels)}

    color_image = np.zeros((segments.shape[0], segments.shape[1], 3))

    for segment_value, label_value in zip(np.unique(segments), labels):
        color_image[segments == segment_value] = label_color_mapping[label_value][:3]  # Extract RGB values

    mpimg.imsave(path, color_image)


def superpixel_mean_gabor_model(img, path, k=3, n = 100):
    
    numpy_gabor_responses_array = extract_superpixel_mean_gabor(img, n)
    kmeans = KMeans(n_clusters=k)
    _, labels = kmeans.fit(numpy_gabor_responses_array)

    segments = extract_segments(img, n)

    # Define a color mapping for visualization
    # This assumes you have a color assigned to each label
    # Adjust the colors or add more as needed
    num_labels = len(np.unique(labels))
    colors = plt.cm.get_cmap('tab20', num_labels)  # Using a colormap for distinct colors
    label_color_mapping = {label: colors(label) for label in np.unique(labels)}

    # Create an empty RGB image with the same shape as the segments
    color_image = np.zeros((segments.shape[0], segments.shape[1], 3))

    # Assign colors based on label values associated with segments
    for segment_value, label_value in zip(np.unique(segments), labels):
        color_image[segments == segment_value] = label_color_mapping[label_value][:3]  # Extract RGB values

    mpimg.imsave(path, color_image)


def process_image(image_path, output_directory):
    img = cv2.imread(image_path)
    for k in range (2, 5):
        print(f'{output_directory}/{k}n_clusters/rgb_pixel_level_result.jpg')
        rgb_pixel_level_model(img, f'{output_directory}/{k}n_clusters/rgb_pixel_level_result.jpg', k)
        rgbxy_pixel_level_model(img, f'{output_directory}/{k}n_clusters/rgbxy_pixel_level_result.jpg', k)
        for i in range(25,125,25):
            superpixel_rgb_mean_model(img, f'{output_directory}/{k}n_clusters/{i}superpixel_rgb_mean_result.jpg', k, i)
            superpixel_rgb_histogram_model(img, f'{output_directory}/{k}n_clusters/{i}superpixel_rgb_histogram_result.jpg', k, i)
            superpixel_mean_gabor_model(img, f'{output_directory}/{k}n_clusters/{i}superpixel_mean_gabor_result.jpg', k, i)

def all_images():

    # Call the function for each image
    process_image('horse.jpg', 'horse_results')
    process_image('tiger.jpg', 'tiger_results')
    process_image('tree.jpg', 'tree_results')
    process_image('cow.jpg', 'cow_results')
    process_image('sea.jpg', 'sea_results')


if __name__ == "__main__":
    all_images()
#    horse()
#    print("ok")
#    cow()
#    print("ok")
#    sea()
#    print("ok")
#    tiger()
#    print("ok")
#    tree()
#    print("ok")