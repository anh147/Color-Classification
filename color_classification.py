import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def crop_image(img):
    """c
    This is a function that crops extra white background
    around product.
    Src:
        https://stackoverflow.com/questions/64046602/how-can-i-crop-an-object-from-surrounding-white-background-in-python-numpy
    """
    mask = img!=255
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    colstart, colend = mask0.argmax(), len(mask0)-mask0[::-1].argmax()+1
    rowstart, rowend = mask1.argmax(), len(mask1)-mask1[::-1].argmax()+1
    return img[rowstart:rowend, colstart:colend]

# tinh histogram cua kenh H tu anh 
def hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    return np.bincount(h.ravel(), minlength=256)


# imageBGR = cv2.imread("digikala/train/blue/105190813.jpg")
# cv2.imshow("raw", imageBGR)
# cv2.waitKey(0)

# for i in range(2):
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('Horizontally stacked subplots')
#     imageBGR = cv2.imread("digikala/train/blue/105190813.jpg")
    
#     imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
#     ax1.imshow(imageRGB)
#     img = crop_image(imageRGB)
#     ax2.imshow(img)
#     plt.show()

# hist = hsv_histogram(img)
# print("hist", hist.shape)
# plt.plot(hist[1:])
# plt.title('blue') # do image lay o folder blue
# plt.show()


classes = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']
X = []
Y = []
count = 1
for label in classes:

    for img_dir in glob.glob('digikala/train/'+label+'/*.jpg'):
        # print("count", count, "label", label, img_dir)
        img = cv2.imread(img_dir)
        hist = hsv_histogram(img)
        # gray.reshape(1,-1)
        X.append(hist)
        Y.append(label)
        count += 1
X = np.array(X)

#splitting the dataset into 80% training data and 20% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#building KNN model
KNN_model = KNeighborsClassifier(n_neighbors=5, p = 1, weights = 'distance')
KNN_model.fit(X_train, Y_train)

#prediction on testing data
Y_predict = KNN_model.predict(X_test)

# print("Y predict", len(Y_predict))

count = 0
for i in range(len(Y_predict)):
    if Y_predict[i] == Y_test[i]:
        count += 1

print("arc", count/len(Y_predict))