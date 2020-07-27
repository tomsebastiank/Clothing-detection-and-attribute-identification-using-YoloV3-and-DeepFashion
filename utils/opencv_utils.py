import cv2
import numpy as np


def find_main_rgb_from_roi(img_roi):
    """
    This function will find out the major color component for a given image ROI
    :param img_roi: sub image
    :return: main color component in RGB tuple (R, G, B)
    """
    Z = img_roi.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img_roi.shape)

    max_hist = -1
    index_max_hist = -1
    for i in range(K):
        if sum(label==i)>max_hist:
            max_hist = int(sum(label==i))
            index_max_hist = i
    # taking the main RGB component by selecting the maximum number of occurance
    rgb = center[index_max_hist]
    main_rgb =tuple((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return main_rgb

