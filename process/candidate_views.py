import random, math, os, json
from PIL import Image
import numpy as np


# ------Set parameters-----
NUM_RANDOM = 11             # Number of random viewpoints to generate (excluding the front)
N_TOTAL = NUM_RANDOM + 1
NEAR_FRONT_PROB = 0.4       # Hybrid strategy: proportion of near-front disturbances
MAX_AZ_NEAR = 60.0          # Foreground disturbance azimuth range setting
MAX_EL_NEAR = 30.0          # Foreground disturbance el range setting
SPHERE_EL_MIN = -30.0       # Global sampling EL lower bound
SPHERE_EL_MAX = 60.0        # Global sampling upper bound of el
MIN_MASK_RATIO = 0.05       # Filter threshold (based on mask or alpha)
FIXED_RADIUS = 2.5
FIXED_ROLL = 0.0


# --------Angle generation function-------
def random_view_near_front(max_az=MAX_AZ_NEAR, max_el=MAX_EL_NEAR):
    az = random.uniform(-max_az, max_az)
    el = random.uniform(-max_el, max_el)
    return az, el, FIXED_ROLL


def random_view_sphere(az_min=0.0, az_max=360.0, el_min=SPHERE_EL_MIN, el_max=SPHERE_EL_MAX):
    az = random.uniform(az_min, az_max)
    el = random.uniform(el_min, el_max)
    return az, el, FIXED_ROLL


def generate_candidate_views(num_random=NUM_RANDOM, near_prob=NEAR_FRONT_PROB):
    views = []
    # Main perspective (front)
    views.append({"view_id":0, "az": 0.0, "el": 0.0, "roll": FIXED_ROLL})
    # Other perspectives
    for i in range(num_random):
        if random.random() < near_prob:
            az, el, roll = random_view_near_front()
        else:
            az, el, roll = random_view_sphere()
        views.append({"view_id":i+1 ,"az": az , "el": el, "roll": roll})
    return views


# ----------Mask-based filtering----------
def mask_ratio_filter(image_path, bg_color=(255,255,255), alpha_threshold=250, min_ratio=MIN_MASK_RATIO):
    """
    - If the image has an alpha channel: use the proportion of pixels with alpha > 0 as the mask ratio
    - Otherwise, use the proportion of pixels different from bg_color (simple threshold)
    """
    im = Image.open(image_path)
    arr = np.array(im)
    h, w = arr.shape[:2]
    if arr.shape[2] == 4:   # RGBA
        alpha = arr[:, :, 3]
        mask_ratio = np.count_nonzero(alpha>alpha_threshold) / (h*w)
        return mask_ratio >= min_ratio, mask_ratio
    else:
        # If there is no alpha, use simple background difference statistics
        bg = np.array(bg_color, dtype=np.uint8)
        diff = np.any(np.abs(arr[:, :, 3] - bg) > 30, axis=2)   # 容错阈值
        mask_ratio = np.count_nonzero(diff) / (h*w)
        return mask_ratio >= min_ratio, mask_ratio

if __name__ == "__main__":
    main()