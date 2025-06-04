from numba import njit # conda install numba
import numpy as np
import cv2
from scipy import stats


@njit
def histogram_figure_numba(np_img):

    # Initialize 256-bin histograms for each channel
    r_hist = np.zeros(256, dtype=np.int32)
    g_hist = np.zeros(256, dtype=np.int32)
    b_hist = np.zeros(256, dtype=np.int32)

    height, width, _ = np_img.shape

    for y in range(height):
        for x in range(width):
            b = np_img[y, x, 0]
            g = np_img[y, x, 1]
            r = np_img[y, x, 2]

            b_hist[b] += 1
            g_hist[g] += 1
            r_hist[r] += 1

    return r_hist, g_hist, b_hist

####

### All other basic functions

# Filters
def apply_all_filters(frame):
    kernel_sharpen = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

    # Define Gabor kernel once
    ksize = 21
    sigma = 5.0
    theta = np.pi / 4  # 45 degrees
    lambd = 10.0
    gamma = 0.5
    psi = 0
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

    # 1. Blur (Gaussian)
    #frame = cv2.GaussianBlur(frame, (7, 7), 0)
    
    # 2. Sharpen
    frame = cv2.filter2D(frame, -1, kernel_sharpen)
    
    # 3. Sobel edge detection (on grayscale)
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    '''
    
    # 4. Gabor filter (on grayscale)
    '''
    gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
    gabor_bgr = cv2.cvtColor(gabor_response, cv2.COLOR_GRAY2BGR)
    '''
    
    # 5. Combine Sobel and Gabor
    # combined = cv2.addWeighted(sobel_bgr, 0.5, gabor_bgr, 0.5, 0)
    
    # 6. Blend combined edges on sharpened frame
    # frame = cv2.addWeighted(frame, 0.7, combined, 0.3, 0)
    
    return frame

# Mean, Mode, Std, Max, Min
def compute_bgr_statistics(frame):
    channels = cv2.split(frame)  # B, G, R channels
    stats_dict = {'mean': [], 'mode': [], 'std': [], 'min': [], 'max': []}

    for c in channels:
        flat = c.flatten()
        stats_dict['mean'].append(float(np.mean(flat)))
        stats_dict['mode'].append(int(stats.mode(flat, keepdims=False).mode))
        stats_dict['std'].append(float(np.std(flat)))
        stats_dict['min'].append(int(np.min(flat)))
        stats_dict['max'].append(int(np.max(flat)))

    # Convert list to tuple for each
    for key in stats_dict:
        stats_dict[key] = tuple(stats_dict[key])

    return stats_dict

# Entropy
def compute_entropy(frame):
    entropy_values = {}
    for i, color in enumerate(['B', 'G', 'R']):
        channel = frame[:, :, i]
        hist = np.bincount(channel.flatten(), minlength=256).astype(np.float32)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        entropy_values[color] = round(float(entropy), 2)
    return entropy_values

# linear trannsformation
def apply_linear_transformation(frame, alpha=1.2, beta=30):
    transformed = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return transformed

# Histogram Equalization
def apply_histogram_equalization(frame):
    # Convert BGR to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Equalize the histogram of the Y channel (luminance)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    
    # Convert back to BGR
    equalized_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return equalized_frame
####
