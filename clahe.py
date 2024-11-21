import cv2
import numpy as np
import os
from PIL import Image

ORIGINAL_IMAGE_FOLDER_PATH = "./images"
PROCESSED_IMAGE_FOLDER_PATH = "./images/processed_images"

IMG1_NAME = "bag1-1.jpeg"
IMG2_NAME = "bag1-4.jpg"

CROPPED_IMG1_NAME = "cropped1.jpg"
CROPPED_IMG2_NAME = "cropped2.jpg"

COMBINED_CONTOUR_IMAGES_NAME = "combined_contour.jpg"
COMBINED_CROPPED_IMAGES_NAME = "combined_cropped.jpg"

SIMILARITY_THRESHOLD = 50
DISTANCE_THRESHOLD = 40

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def find_largest_contour(image):
    enhanced_image = apply_clahe(image)  # Applying CLAHE to enhance image contrast
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found! Using full image.")
        full_img_contour = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
        return full_img_contour, image

    half_image_area = 0.5 * image.shape[0] * image.shape[1]
    max_area = 0
    max_contour = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = w * h
        if contour_area > max_area and contour_area >= half_image_area:
            max_area = contour_area
            max_contour = contour

    if max_contour is None:
        print("No contour large enough found! Using full image.")
        full_img_contour = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
        return full_img_contour, image

    contour_image = image.copy()
    cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0), 7)
    return max_contour, contour_image


def crop_region_from_contour(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]

def normalize_image(image):
    image_float = image.astype(np.float32)
    cv2.normalize(image_float, image_float, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(image_float)

def compare_images_for_similarity(image1, image2):
    img1 = normalize_image(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    img2 = normalize_image(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
    
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
    
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < SIMILARITY_THRESHOLD]
    
    print(f"Keypoints in image 1: {len(kp1)}")
    print(f"Keypoints in image 2: {len(kp2)}")
    print(f"Number of good matches: {len(good_matches)}")
    
    if len(good_matches) > SIMILARITY_THRESHOLD:
        distances = [m.distance for m in good_matches]
        avg_distance = sum(distances) / len(distances)
        print("\nImages are similar." if avg_distance < DISTANCE_THRESHOLD else "\nImages are different.")
    else:
        print("\nImages are different.")

def combine_and_save_images(image1, image2, file_path, screen_width=1920, screen_height=1080):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    max_height = max(height1, height2)
    total_width = width1 + width2
    
    background = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
    background[:height1, :width1] = image1
    background[:height2, width1:width1+width2] = image2
    
    frame_thickness = 7
    cv2.rectangle(background, (0, 0), (width1, height1), (255, 0, 0), frame_thickness)
    cv2.rectangle(background, (width1, 0), (width1 + width2, height2), (255, 0, 0), frame_thickness)
    
    aspect_ratio = background.shape[1] / background.shape[0]
    new_width = screen_width
    new_height = int(new_width / aspect_ratio)
    if new_height > screen_height:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        
    resized_image = cv2.resize(background, (new_width, new_height))
    cv2.imwrite(file_path, resized_image)


image1 = cv2.imread(os.path.join(ORIGINAL_IMAGE_FOLDER_PATH, IMG1_NAME))
image2 = cv2.imread(os.path.join(ORIGINAL_IMAGE_FOLDER_PATH, IMG2_NAME))

contour1, contoured_img1 = find_largest_contour(image1)
contour2, contoured_img2 = find_largest_contour(image2)
if contour1 is None or contour2 is None:
    exit()
    
cropped_image1 = crop_region_from_contour(image1, contour1)
cropped_image2 = crop_region_from_contour(image2, contour2)

cv2.imwrite(os.path.join(PROCESSED_IMAGE_FOLDER_PATH, CROPPED_IMG1_NAME), cropped_image1)
cv2.imwrite(os.path.join(PROCESSED_IMAGE_FOLDER_PATH, CROPPED_IMG2_NAME), cropped_image2)

compare_images_for_similarity(cropped_image1, cropped_image2)

combine_and_save_images(contoured_img1, contoured_img2, 
                        os.path.join(PROCESSED_IMAGE_FOLDER_PATH, COMBINED_CONTOUR_IMAGES_NAME))

combine_and_save_images(cropped_image1, cropped_image2, 
                        os.path.join(PROCESSED_IMAGE_FOLDER_PATH, COMBINED_CROPPED_IMAGES_NAME))
