from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

directory = './dataset/'

image_files = os.listdir(directory)

pixels_per_cm = 50

def calculate_foot_area(image_path):
    image = Image.open(image_path)

    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    _, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    foot_contour = contours[0]

    foot_area_pixels = cv2.contourArea(foot_contour)

    foot_area_cm2 = foot_area_pixels / (pixels_per_cm ** 2)

    return foot_area_cm2, foot_contour, image_binary

for filename in image_files:
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(directory, filename)

        foot_area_cm2, foot_contour, image_binary = calculate_foot_area(image_path)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        fig.text(0.5, 0.16, 'Press ctrl + w to close and go to next photo', ha='center', fontsize=12, color='red')
        fig.suptitle('Foot Area Calculation', fontsize=16)

        ax1.imshow(Image.open(image_path))
        ax1.set_title(f'Original Image - {filename}')
        ax1.axis('on')

        ax2.imshow(image_binary, cmap='gray')
        ax2.set_title('Binary Image')
        ax2.axis('on')

        image_contours = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_contours, [foot_contour], -1, (0, 255, 0), 2)
        ax3.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
        ax3.set_title('Contour Image')
        ax3.axis('on')
        ax3.text(10, 10, f"Foot area: {foot_area_cm2:.2f} cm²", color='white', bbox=dict(facecolor='black', alpha=0.5))

        plt.tight_layout()
        plt.show()

        print(f"Foot area in {filename}: {foot_area_cm2:.2f} cm²")
