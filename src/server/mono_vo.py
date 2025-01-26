# Taken from/Inspired by https://github.com/alishobeiri/Monocular-Video-Odometery
# and https://github.com/uoip/monoVO-python

import cv2
import numpy as np
from server.camera import Camera, Frame
from server.loop import Loop


class MonoVOMixin:
    def __init__(
            self,
            camera: Camera,
            lk_params=dict(
                winSize=(21,21),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            ),
            **kwargs
        ):
        super().__init__(**kwargs)
        self.camera = camera

        self.frame = None
        self.feat_pts = None
        self.min_num_feature = 10
        self.lk_params = lk_params

        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.v = np.zeros(3)

        self.total_t = np.zeros(3)
        self.total_R = np.eye(3)

        self.detector = cv2.FastFeatureDetector_create(
            threshold=25,
            nonmaxSuppression=True
        )

        self.frame_loop = Loop(
            interval=0.05,
            func=self._update_frame
        )
        self.frame_loop.start()

    def process_first_frame(self, frame: Frame):
        self.frame = frame
        feat_pts = self.detector.detect(frame.data)
        self.feat_pts = np.array([x.pt for x in feat_pts], dtype=np.float32)

    def get_features(self, frame: Frame):
        if(self.feat_pts.shape[0] < self.min_num_feature):
            self.feat_pts = self.detector.detect(frame.data)
            self.feat_pts = np.array([x.pt for x in self.feat_pts], dtype=np.float32)

        features, status, err = cv2.calcOpticalFlowPyrLK(
            self.frame.data,
            frame.data,
            self.feat_pts,
            None,
            **self.lk_params
        )
        status = status.reshape(status.shape[0])
        img_1_feat = self.feat_pts[status == 1]
        img_2_feat = features[status == 1]
        
        self.feat_pts = img_2_feat
        self.frame = frame

        return img_1_feat, img_2_feat

    def compute_Rt(self, img_1_feat, img_2_feat):
        E, mask = cv2.findEssentialMat(
            img_1_feat,
            img_2_feat,
            focal=self.camera.fx,
            pp=(self.camera.cx, self.camera.cy),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        _, R, t, mask = cv2.recoverPose(
            E,
            img_1_feat,
            img_2_feat,
            focal=self.camera.fx,
            pp=(self.camera.cx, self.camera.cy),
        )

        self.total_R = R.dot(self.total_R)
        self.total_t = self.total_R.dot(t) + self.total_t
        self.R = R
        self.t = t

    def _update_frame(self):
        frame = self.camera.get_frame()
        if frame is not None:
            if self.last_frame is None:
                self.process_first_frame(frame)
            else:
                img_1_feat, img_2_feat = self.get_features(frame)
                self.compute_Rt(img_1_feat, img_2_feat)


    def get_vo_data(self):
        return self.total_t.tolist()
    
    def deinit_vo(self):
        self.frame_loop.stop()
        self.camera.close()
