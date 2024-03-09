import cv2

# to convert images from RGB to Grayscale
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
# To equalize the intensity of converted grayscale image
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
    
# Preprocess the image and normalization of image
def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
