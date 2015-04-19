import gzip
import cPickle
import cv
import numpy as np
import cv2
import root



def load_mnist():
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    path = str(root.path()) + '/res/mnist.pkl.gz'
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def createImage(arrImg, fileName = None):
    arrImg = np.reshape(arrImg, (28,28))
    arrImg = np.array(arrImg * 255, dtype = np.uint8)
    arrImg = cv2.threshold(arrImg, 0, 255, cv2.THRESH_BINARY)[1]
    arrImg = cv2.resize(arrImg, (10, 10))
    return arrImg

#get total train set
#if instance = True, then return just one instance of each class
def get_train_set(instance = False, number_of_instances = 1):
    trainSet = load_mnist()[0]
    numberOfInstances = number_of_instances
    if instance:
        train_img_set = [[0 for x in range(2)] for x in range(10*numberOfInstances)]
    else:
        train_img_set = [[0 for x in range(2)] for x in range(50000)]
    count = np.zeros(10)
    count[0] = 0
    count[1] = 0
    count[2] = 0
    count[3] = 0
    count[4] = 0
    count[5] = 0
    count[6] = 0
    count[7] = 0
    count[8] = 0
    count[9] = 0

    if instance == False:
        for i in range(50000):
            img = createImage(trainSet[0][i])
            arrImage = np.zeros(100)
            for j in range(10):
                for k in range(10):
                    arrImage[j*10+k] = img[j][k]
            train_img_set[i][0] = arrImage
            train_img_set[i][1] = trainSet[1][i]
    else:
        n = 0
        for i in range(10000):
            img = createImage(trainSet[0][i])
            arrImage = np.zeros(100)
            for j in range(10):
                for k in range(10):
                    arrImage[j*10+k] = img[j][k]

            if count[trainSet[1][i]] < numberOfInstances:
                train_img_set[n][0] = arrImage
                train_img_set[n][1] = trainSet[1][i]
                count[trainSet[1][i]] = count[trainSet[1][i]] + 1
                n += 1
            may_continue = True
            for j in range(10):
                if count[j] < numberOfInstances:
                    may_continue = False
                    break

    return train_img_set



# img = cv2.imread(root.path()+"/images/welcome_friends.jpg")
# img = cv2.resize(img,(280, 280), interpolation = cv2.INTER_CUBIC)
# cv2.imwrite(root.path()+"/images/proc.jpg", img)