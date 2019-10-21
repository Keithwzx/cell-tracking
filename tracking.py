from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import numpy as np
import cv2
import random



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
    return np.transpose(np.mean(contour, axis=0))

class KalmanBoxTracker():
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
        self.cont = cont
        color = np.uint8(np.random.uniform(100, 255, 3))
        self.display_color = tuple(map(int, color)) #for displaying

    def update(self, cont):
        """
        Updates the state vector with observed contour.
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
        Returns the current mean estimate
        """
        return self.kf.x

class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.trackers_dict = {}

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        ret = []
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                print(d[0])
                trk.update(dets[d[0]])
                self.trackers_dict[trk.id] = trk

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers_dict[trk.id] = trk
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[:2]
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.append(d, trk.id).reshape(1, -1))  # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                del self.trackers_dict[trk.id]
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if (len(self.trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)
        iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = iou_contours(det, trk.cont)
        matched_indices = linear_assignment(-iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def display_trackers(self,image,ret):
        if len(self.trackers) > 0:
            updated_tracks = ret[:, 2].astype(np.int)
            for id in updated_tracks:
                cv2.drawContours(image, self.trackers_dict[id].cont, -1, self.trackers_dict[id].display_color, 3)
                cv2.putText(image, str(self.trackers_dict[id].id), (self.trackers_dict[id].kf.x[0], self.trackers_dict[id].kf.x[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(255,255,255),thickness=1)

        return image
