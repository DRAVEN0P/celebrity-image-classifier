from collections import Counter
import cv2
import numpy as np
import pywt
import joblib
MODEL = joblib.load('saved_model.pkl')
path = None  #------------------------------ Enter the path of the image here ------------------------



no_run = 0
cd = [
    # './haarcascades/haarcascades_frontalcalface.xm',
    './haarcascades/haarcascade_frontalface_alt.xml',
    './haarcascades/haarcascade_frontalface_alt2.xml',
    './haarcascades/haarcascade_frontalface_alt_tree.xml',
    './haarcascades/haarcascade_frontalface_default.xml',
    # './haarcascades/haarcascade_frontalface_default_alt.xml',
    './haarcascades/haarcascade_frontalcatface_extended.xml',
    # './haarcascades/haarcascade_upperbody.xml',
    './haarcascades/haarcascade_profileface.xml',
    # './haarcascades/haarcascade_eye.xml',
    # './haarcascades/haarcascade_eye_tree_eyeglasses.xml',
    # './haarcascades/haarcascade_eye_tree_eyeglasses2.xml',
    # './haarcascades/haarcascade_eye_tree_eyeglasses3.xml',
]


def crop_image(path):
    cord_list = []
    img = cv2.imread(path)
    img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in cd:
        a = ()
        face_cascade = cv2.CascadeClassifier(i)
        faces = face_cascade.detectMultiScale(gray)
        if (type(faces) == type(a)):
            pass
        else:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h,x:x+w]
            eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml',)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes)>=1:
                cord_list.append((x, y, w, h))
                global no_run
                no_run = no_run + 1

    return (cord_list)
    # x,y,w,h = cord_list[0]
    # x,y,w,h

    # final = img[y:y+h,x:x+w]


def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def check_image(num, img):
    a = 0
    try:
        (x, y, w, h) = crd_ls[num]
        img = img[y:y+h, x:x+w]

        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(
            32*32*3, 1), scalled_img_har.reshape(32*32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        return (MODEL.predict(final)[0])
    except:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(
            32*32*3, 1), scalled_img_har.reshape(32*32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        global no_run
        no_run += 1

        return (MODEL.predict(final)[0])


celeb_dict = {'Mbappe': 0, 'Ms dhoni': 1, 'Ronaldo': 2, 'Virat Kohli': 3, 'Messi': 4}


img1 = cv2.imread(path)
crd_ls = crop_image(path)


def find_max_repetition(lst):
    count = Counter(lst)
    max_count = max(count.values())
    most_common = [item for item, freq in count.items() if freq == max_count]
    return most_common

# Example usage


res = []
check_image(0, img1)
for i in range(no_run):
    result = check_image(i, img1)
    res.append(result)
print(res)
final = find_max_repetition(res)


print(list(celeb_dict.keys())[list(celeb_dict.values()).index(final[0])])
