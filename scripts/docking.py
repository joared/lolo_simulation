import numpy as np
import time
import sys

sys.path.append("../../simulation/scripts")
from feature import FeatureModel
from camera import Camera
from auv import AUV
sys.path.append("../../perception/scripts")
from feature_extraction import ThresholdFeatureExtractor, featureAssociation
from pose_estimation import DSPoseEstimator
sys.path.append("../../control/scripts")
from control import IBVSController, PBVSController

class StateEstimator:
    def __init__(self, auv, dockingStation):
        self.auv = auv
        self.ds = dockingStation

    def propagate(self, dt, controlCommand=None):
        pass

    def update(self, dsTranslation, dsRotation):
        pass

class Docker:
    def __init__(self):
        featureModel = FeatureModel(0.3, 4, shift=True, centerPointDist=0.3, zShift=0)

        camHz = 1 # camera update rate
        pixelWidth = 0.0000028
        pixelHeight = 0.0000028
        cameraMatrix = np.array([[812.2540283203125,   0,    		    329.864062734141 ],
                                [   0,               814.7816162109375, 239.0201541966089], 
                                [   0,     		       0,   		      1             ]], dtype=np.float32)

        cameraMatrix[0, :] *= pixelWidth
        cameraMatrix[1, :] *= pixelHeight

        camera = Camera(cameraMatrix, 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(480, 640), 
                        pixelWidth=pixelWidth, 
                        pixelHeight=pixelHeight, 
                        hz=15)

        auv = AUV(camera, 
                placement="back", 
                size=(3, 10), # Arbitrary, currently only for simulation
                translation=(0, 0, 0),
                euler=(0, 0, 0)) # XYZ

        dockingStation = AUV(featureModel, 
                            placement="front", 
                            size=(10, 10), # Arbitrary, currently only for simulation
                            translation=(0, 0, 0),
                            euler=(0, 0, 0)) # XYZ

        self.featureExtractor = ThresholdFeatureExtractor(nFeatures=len(featureModel.features))
        self.poseEstimator = DSPoseEstimator(camera, featureModel)#, auv, dockingStation)
        self.controller = PBVSController(auv, dockingStation)
        self.stateEstimator = StateEstimator(auv, dockingStation)

        self.controlCommand = None

    def captureImage(self, img):
        self.img = img

    def requestRendezvous(self, req):
        print("New rendezvous received")
        pass

    def control(self, auvDeltaTranslation, auvDeltaRot, dt=None):
        if dt is None:
            currT = time.time()
            dt = currT - self.lastControlT

        if self.controlCommand:
            self.stateEstimator.propagate(dt, self.controlCommand)
        else:
            self.stateEstimator.propagate(dt)

        # projected image features
        if self.img is not None:
            gray = self.img
            imgColor = self.img
            res, pointsUV = featureExtractor(gray, imgColor)
            # points are in pixels, convert points to meters
            points = self.camera.uvToMeters(pointsUV)
            points = np.array(pointsUV, dtype=np.float32)
            associatedPoints, featurePointsScaled, (cx,cy) = featureAssociation(self.featureModel.features, points)

            dsTranslation, dsRotation = self.poseEstimator.update(associatedPoints)
            self.stateEstimator.update(dsTranslation, dsRotation)
            self.img = None

        controlCommand, err = self.controller.control(self.stateEstimator.ds.translation, 
                                           self.stateEstimator.ds.rotation)
        self.controlCommand = controlCommand
        return controlCommand

if __name__ == "__main__":
    pbvs = True
    featureModel = FeatureModel()
    featureExtractor = FeatureExtractor()
    poseEstimator = PoseEstimator()
    controller = PBVSController() if pbvs else IBVSController()

    vs = Docker(featureModel, featureExtractor, poseEstimator, controller)