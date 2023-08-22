import cv2
import numpy as np
images = []
for i in range(1, 11):
    image = cv2.imread(f'{i}.png')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images.append(hsv_image)
burnt_grass_lower = np.array([0, 100, 100])
burnt_grass_upper = np.array([10, 255, 255])

green_grass_lower = np.array([0, 100, 100])
green_grass_upper = np.array([10, 255, 255])

output_images = []

for image in images:
    burnt_grass_mask = cv2.inRange(image, burnt_grass_lower, burnt_grass_upper)
    green_grass_mask = cv2.inRange(image, green_grass_lower, green_grass_upper)
    image_with_overlay = cv2.addWeighted(image, 1, cv2.cvtColor(burnt_grass_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    image_with_overlay = cv2.addWeighted(image_with_overlay, 1, cv2.cvtColor(green_grass_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    output_images.append(image_with_overlay)
def detect_houses(image):
    blue_lower = np.array([0, 100, 100])
    blue_upper = np.array([10, 255, 255])
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    blue_mask = cv2.inRange(image, blue_lower, blue_upper)
    red_mask = cv2.inRange(image, red_lower, red_upper)
    house_mask = cv2.bitwise_or(blue_mask, red_mask)
    contours, _ = cv2.findContours(house_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_blue = 0
    count_red = 0
    priority_blue = 0
    priority_red = 0

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3:  
            color_at_centroid = image[approx[0][0][1], approx[0][0][0]]
            if np.array_equal(color_at_centroid, [255, 0, 0]):  
                count_blue += 1
                priority_blue += 2
            elif np.array_equal(color_at_centroid, [0, 0, 255]): 
                count_red += 1
                priority_red += 1

    return count_blue, count_red, priority_blue, priority_red

house_counts = []
house_priorities = []

for image in images:
    count_blue, count_red, priority_blue, priority_red = detect_houses(image)
    house_counts.append((count_blue, count_red))
    house_priorities.append((priority_blue, priority_red))
image_names = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']
rescue_ratios = []

for priority in house_priorities:
    if priority[1] != 0:
        ratio = priority[0] / priority[1]
    else:
        ratio = 0  
    rescue_ratios.append(ratio)


sorted_images = sorted(zip(image_names, rescue_ratios), key=lambda x: x[1], reverse=True)

for img_name, ratio in sorted_images:
    print(f"Image: {img_name}, Rescue Ratio: {ratio}")
for i, output_img in enumerate(output_images):
    cv2.imshow(f'Output Image {i + 1}', output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


