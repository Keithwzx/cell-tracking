import os
import nibabel as nib
from nilearn import plotting
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import warnings
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import glob
import random
warnings.simplefilter("ignore")


def save_tiff(path,tiff_to_be_saved):
    im = Image.fromarray(tiff_to_be_saved.astype(np.int16))
    im.save(path)


def generate_tracking_txt(tracking_results, sort_algorithm):
    """
    Generating res_track.txt for Cell Tracking Benchmark
    """
    all_tracks =  {**sort_algorithm.deleted_trackers, **sort_algorithm.trackers_dict}
    file_path = os.path.join(os.path.join(tracking_results, "results", "man_track.txt"))
    for id in sorted(all_tracks.keys()):
        line = f"{id} {all_tracks[id].begin_frame} {all_tracks[id].end_frame} 0"
        with open(file_path, 'a') as file:
            file.write(line)
            file.write("\n")

def gif_create_in_same_folder(tracking_results):
    # filepaths
    fp_in = os.path.join(tracking_results,"*.png")
    try:
        os.mkdir(os.path.join(tracking_results, "gif"))
    except:
        pass

    fp_out = os.path.join(tracking_results, "gif", tracking_results.split("/")[-2] + "_" + tracking_results.split("/")[-1] + ".gif")

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)



def detector(image):
    """
    :param image: .png file.
    :return:
    """
    def connected_components(image):
        """

        :param image:
        :return:
        """
        image2 = np.array(image, dtype= np.uint8)
        image2 = np.copy(image2)
        image2 = np.ascontiguousarray(image2, dtype=np.uint8)
        _, contours, _ = cv2.findContours(image2,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected_contours = []
        for contour in contours:
            if len(contour) > 20:
                selected_contours.append(contour)
        # cv2.drawContours(image2, selected_contours, -1, (0, 255, 0), 3)
        # plt.imshow(image)
        return selected_contours
    return connected_components(image)



def iou_contours(contour1,contour2):
    '''
        :param contour1: Output of cv2.findcontours() function
        :param contour2: Output of cv2.findcontours() function
        :return: IOU with rectangular assumption. It is to reduce computational power when two contours is far away.
        Also,
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        '''
    bb1 = {}
    bb2 = {}
    bb1["x1"] = np.min(contour1[:, 0], axis=0)[0]
    bb1["y1"] = np.min(contour1[:, 0], axis=0)[1]
    bb1["x2"] = np.max(contour1[:, 0], axis=0)[0]
    bb1["y2"] = np.max(contour1[:, 0], axis=0)[1]
    bb2["x1"] = np.min(contour2[:, 0], axis=0)[0]
    bb2["y1"] = np.min(contour2[:, 0], axis=0)[1]
    bb2["x2"] = np.max(contour2[:, 0], axis=0)[0]
    bb2["y2"] = np.max(contour2[:, 0], axis=0)[1]
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    else:
        blank = np.zeros((1024, 1024))
        img1 = cv2.fillConvexPoly(blank.copy(), contour1, color=1)
        img2 = cv2.fillConvexPoly(blank.copy(), contour2, color=1)
        # plt.imshow(img2)
        intersection = len(np.nonzero(cv2.bitwise_and(img2, img1))[0])
        union = len(np.nonzero(cv2.bitwise_or(img2, img1))[0])
        if union == 0:
            return 0.0
        else:
            return intersection / union


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

    def __init__(self, cont, frame_count):
        """
        Initialises a tracker using initial contour.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R *= 2.
        self.kf.P *= 20.
        self.kf.P[2:, 2:] *= 100.  # give high uncertainty to the unobservable initial velocities
        self.kf.Q = Q_discrete_white_noise(4, 1, 20)
        self.kf.x[:2] = contours_to_z(cont)
        self.kf.x[2:] = 3


        # Evaluation
        self.begin_frame = frame_count
        self.end_frame = frame_count
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.cont = cont
        color = np.uint8(np.random.uniform(100, 255, 3))
        self.display_color = tuple(map(int, color)) #for displaying
        self.trajectory = [self.kf.x]
        self.measurements = [cont]

    def update(self, cont):
        """
        Updates the state vector with observed contour.
        """

        self.time_since_update = 0
        self.end_frame += 1
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = contours_to_z(cont)
        self.kf.update(z)
        self.cont = cont
        self.trajectory.append(self.kf.x)
        self.measurements.append(z)

    def predict(self):
        """
        Advances the state vector and returns the predicted state
        """
        # if ((self.kf.x[1] + self.kf.x[3]) <= 0):
        #     self.kf.x[3] *= 0.0
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

class Associator(object):
    def __init__(self, max_age=1, min_hits=1):
        """
        Sets key parameters for Associator
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.trackers_dict = {}
        self.deleted_trackers = {}


    def update(self, dets):
        """
        Params:
          dets - List of output of cv2.findcontours() function.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        ret = []
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                # print(d[0])
                trk.predict()
                trk.update(dets[d[0]])
                self.trackers_dict[trk.id] = trk
            else:
                # print("sa")
                trk.time_since_update += 1
                self.trackers_dict[trk.id] = trk

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i], self.frame_count)
            # print(f"olusturdum: {trk.id}")
            self.trackers_dict[trk.id] = trk
            self.trackers.append(trk)
        i = len(self.trackers)


        for trk in reversed(self.trackers):
            d = trk.get_state()[:2]
            if ((trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.append(d, trk.id).reshape(1, -1))  # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                # print(f"siliyorum: {trk.id}")
                self.trackers.pop(i)
                self.deleted_trackers[trk.id] = trk
                del self.trackers_dict[trk.id]
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, iou_threshold=0.15):
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
        """
        Each cell contour is drawed with transparency and id_number. Also, we generated .tif images for evaluation in this part.


        """
        if len(self.trackers) > 0:
            updated_tracks = ret[:, 2].astype(np.int)
            overlay = image.copy()
            output = image.copy()
            blank = np.zeros((image.shape)).astype(np.uint8)
            for id in updated_tracks:
                cv2.drawContours(blank, [self.trackers_dict[id].cont], -1, (int(id), int(id), int(id)), thickness=-1)
                try:
                    cv2.drawContours(overlay, [self.trackers_dict[id].cont], -1, self.trackers_dict[id].display_color, thickness = -1)
                    cv2.putText(overlay, str(self.trackers_dict[id].id),
                                (self.trackers_dict[id].kf.x[0]-5, self.trackers_dict[id].kf.x[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                except:
                    pass

            tiff_to_be_saved = blank[:,:,0]

            cv2.addWeighted(overlay, 0.2, output, 1 - 0.2,
                            0, output)
            for id in updated_tracks:
                try:
                    cv2.putText(output, str(self.trackers_dict[id].id),
                                (self.trackers_dict[id].kf.x[0]-5, self.trackers_dict[id].kf.x[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                except:
                    pass

        return tiff_to_be_saved, output


def main():   
    experiments= ["PhC-C2DH-U373", "Fluo-N2DL-HeLa" ]
    path = r"C:\Users\mete\Documents\Github-Rep\medical-tracking\cell-tracking-master"
    datasets = r"C:\Users\mete\Documents\Github-Rep\medical-tracking\data\u-net"
    tracking_results_path = os.path.join(path,"tracked")

    for experiment in experiments:

        output_path = os.path.join(path, "outputs",experiment)
        photo_path = os.path.join(datasets,experiment,"niftynet_data")
        try:
            os.mkdir(os.path.join(path, "tracked", experiment))
        except:
            pass
        for pack in ["01", "02"]:
            KalmanBoxTracker.count = 0
            sort_algorithm = Associator()

            try:
                os.mkdir(os.path.join(path, "tracked", experiment, pack))
            except:
                pass
            tracking_results = os.path.join(path,"tracked", experiment, pack)
            try:
                os.mkdir(os.path.join(tracking_results, "results"))
            except:
                pass


            detections = []
            ids = []
            files = []
            for f in os.listdir(output_path):
                if f.split("_")[3] == pack:
                    img = nib.load(os.path.join(output_path,f))
                    data = img.get_fdata()
                    files.append(f)
                    ids.append(int(f.split("_")[2]))

            sorted_files = [files_1 for _,files_1 in sorted(zip(ids,files))]

            ## tiff
            def read_tiff(path):
                im = Image.open(os.path.join(photo_path, path))
                return np.array(((np.array(im) - np.min(np.array(im))) / (np.max(np.array(im)) - np.min(np.array(im))) * 255),
                         dtype=np.uint8)


            for i in range(len(sorted_files)):
                img = nib.load(os.path.join(output_path,sorted_files[i])).get_fdata()[:, :, 0, 0, 0]
                dets = detector(nib.load(os.path.join(output_path,sorted_files[i])).get_fdata()[:, :, :, 0, 0])
                real_image = read_tiff("img_" + sorted_files[i].split("_")[2] + "_" + sorted_files[i].split("_")[3] + ".tif")
                three_channel = cv2.merge([real_image,real_image,real_image])
                cv2.imwrite(os.path.join(tracking_results, sorted_files[i].split("_")[2] + "_" + sorted_files[i].split("_")[3] + ".png"),real_image)
                tiff_to_be_saved, drawed_img = sort_algorithm.display_trackers(three_channel, sort_algorithm.update(dets))
                save_tiff(os.path.join(tracking_results, "results", "mask" + sorted_files[i].split("_")[2] + ".tif"),tiff_to_be_saved)
                name = sorted_files[i].split("_")[2] + "_" + sorted_files[i].split("_")[3] + "tra.png"
                print(name)
                cv2.imwrite(os.path.join(tracking_results, name),drawed_img)
            generate_tracking_txt(tracking_results, sort_algorithm)

            del sort_algorithm

if __name__ == "__main__": 
    main()


