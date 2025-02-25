import cv2 as cv
import imutils


def preprocess_brain_tumor(img_path):
    print(f"Preprocessing image: {img_path}")
    
    # Read image
    img = cv.imread(img_path)
    if img is None:
        print("Error: Image not found or could not be loaded!", flush=True)
        return None
    
    print("Image loaded successfully!", flush=True)

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Find contours
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:  # Check if no contours found
        print("No contours found!", flush=True)
        return None

    c = max(cnts, key=cv.contourArea)

    # Find extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop image
    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if new_image.size == 0:
        print("Error: Cropped image is empty!", flush=True)
        return None

    print(f"Cropped image size: {new_image.shape}", flush=True)

    # Resize and normalize
    image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    image = image / 255.0

    # Reshape for model
    image = image.reshape((1, 240, 240, 3))

    print("Preprocessing completed successfully!", flush=True)
    return image  # Return processed image
