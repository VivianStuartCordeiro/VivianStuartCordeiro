import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Load the image
img = cv2.imread('car.jpg')
if img is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges
edged = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    print("No contours found.")
    exit()

# Sort contours and find the rectangular contour
screenCnt = None
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    print("No rectangular contour found.")
    exit()

# Draw the contour
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Crop the number plate
x, y, w, h = cv2.boundingRect(screenCnt)
roi = img[y:y + h, x:x + w]

# Apply OCR
text = pytesseract.image_to_string(roi, config='--psm 11')
print(text)

# Show the images for debugging
cv2.imshow('Gray', gray)
cv2.imshow('Edged', edged)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
