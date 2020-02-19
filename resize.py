import cv2
scale =4
def modcrop(img, scale=4):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = int(h / scale) * scale
        w = int(w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int(h / scale) * scale
        w = int(w / scale) * scale
        img = img[0:h, 0:w]
    return img

img = cv2.imread('lenna.png',1)
print(img)
label_ = modcrop(img, scale)

input_ = cv2.resize(label_, None, fx=1.0/scale, fy=1.0/scale,interpolation=cv2.INTER_CUBIC)  # Resize by scaling factor

cv2.imwrite('lennasm.png',input_)