import cv2
import numpy as np

def calculate_histogram(image_path):
    image = cv2.imread(image_path)

    # Convert the image to HSV color space (better for lighting invariance)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Normalize the histograms
    hist_hue /= hist_hue.sum()
    hist_saturation /= hist_saturation.sum()
    hist_value /= hist_value.sum()

    return hist_hue, hist_saturation, hist_value

def compare_histograms(hist1, hist2):
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation


image_path_1 = 'images/woman2-5.jpg'  
image_path_2 = 'images/woman2-2.jpg'

hist1_hue, hist1_saturation, hist1_value = calculate_histogram(image_path_1)
hist2_hue, hist2_saturation, hist2_value = calculate_histogram(image_path_2)

correlation_hue = compare_histograms(hist1_hue, hist2_hue)
correlation_saturation = compare_histograms(hist1_saturation, hist2_saturation)
correlation_value = compare_histograms(hist1_value, hist2_value)

total_correlation = (correlation_hue + correlation_saturation + correlation_value) / 3

print(f"Correlation between Hue histograms: {correlation_hue}")
print(f"Correlation between Saturation histograms: {correlation_saturation}")
print(f"Correlation between Value histograms: {correlation_value}")
print(f"Total similarity (average): {total_correlation}")

if total_correlation > 0.9:
    print("\nThe images are similar.")
else:
    print("\nThe images are different.")