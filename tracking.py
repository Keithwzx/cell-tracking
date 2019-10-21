from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import numpy as np
import cv2


def iou_contours(contour1,contour2):
    """
    :param contour1: Output of cv2.findcontours() function
    :param contour2: Output of cv2.findcontours() function
    :return: iou overlap between two contour
    """
    blank = np.zeros((1024, 1024))
    img1 = cv2.fillPoly(blank.copy(), pts=[contour1], color= 1)
    img2 = cv2.fillPoly(blank.copy(), pts=[contour2], color= 1)
    intersection = len(np.nonzero(cv2.bitwise_and(img2, img1))[0])
    union = len(np.nonzero(cv2.bitwise_or(img2, img1))[0])
    return intersection/union

def contours_to_z(contour):
    """

    :param contour: Output of cv2.findcontours() function
    :return: z = [x,y] observables
    """
    return np.mean(contour, axis=0)

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as origin of the contour.
    """
    count = 0

    def __init__(self, cont):
        """
        Initialises a tracker using initial contour.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]])

        self.kf.R *= 10.
        self.kf.P *= 10.
        self.kf.P[2:, 2:] *= 100.  # give high uncertainty to the unobservable initial velocities
        self.kf.Q *= 0.01

        self.kf.x[:2] = contours_to_z(cont)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, cont):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(contours_to_z(cont))

    def predict(self):
        """
        Advances the state vector and returns the predicted state
        """
        if ((self.kf.x[1] + self.kf.x[3]) <= 0):
            self.kf.x[3] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x

