import cv2

# to convert images from RGB to Grayscale
def grayscale(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return im
    
# To equalize the intensity of converted grayscale image
def equalize(img):
    im = cv2.equalizeHist(img)
    return im
    
# Preprocess the image and normalization of image
def preprocess(img):
    im = grayscale(img)
    im = equalize(img)
    im = im/255
    return im
