#!/usr/bin/env python
import time
import numpy as np
import rospy
import tf
import tf.msg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from smarc_msgs.msg import ThrusterRPM, FloatStamped
from std_srvs.srv import Trigger, TriggerResponse

from lolo_perception.perception_utils import projectPoints
from lolo_perception.perception_ros_utils import vectorToPose, vectorQuatToTransform, vectorToTransform, featurePointsToMsg, readCameraYaml
from lolo_simulation.auv import AUV, DockingStation
from lolo_control.control_ros_utils import twistToVel, stateToOdometry
from scipy.spatial.transform import Rotation as R

import cv2 as cv

class ROSSimulator:
    def __init__(self, 
                 camera,
                 cameraInfo,
                 featureModel,
                 velocityMode=False):

        self.camera = camera
        self.cameraInfo = cameraInfo
        
        self.featureModel = featureModel

        self.auv = AUV(relativeCameraTranslation=np.array([-2.5, 0, -0.33]),
                       relativeCameraRotation=R.from_euler("XYZ", (0, -np.pi/2, -np.pi/2)).as_rotvec(),
                       translationVector=np.array([.5, 0., -2]),
                       rotationVector=np.array([0., 0., np.pi/2]))

        self.dockingStation = DockingStation(translationVector=np.array([0., -12., -1.8]),
                                             rotationVector=np.array([0., 0., np.pi/2]))

        self.hz = 10 # publish rate
        self.dt = 1./self.hz
        self.pause = False
        # TODO: currently camera rate is set in rviz, change so its done here instead
        #self.camHz = 10. # camera update rate
        #self.imageUpdateIndex = self.hz / self.camHz # image capture update index

        self._velDockingStationInit = np.array([0., 0., 0., 0., 0., 0.])
        self._velAUVInit = np.array([0., 0., 0., 0., 0., 0.])
        self._velDockingStation = self._velDockingStationInit.copy()
        self._velAUV = self._velAUVInit.copy()

        self._controlCommandDockingStationInit = np.array([400., 0., 0.]) # [n, deltaR, deltaE]
        self._controlCommandAUVInit = np.array([400., 0., 0.])
        self._controlCommandDockingStation = self._controlCommandDockingStationInit.copy()
        self._controlCommandAUV = self._controlCommandAUVInit.copy()

        self.cameraInfoPublisher = rospy.Publisher("lolo_camera/camera_info", CameraInfo, queue_size=1)

        self.cvBridge = CvBridge()
        self.imagePublisher = rospy.Publisher("lolo_camera/image_raw", Image, queue_size=1)
        self.imageSubscriber = rospy.Subscriber("/cam_image", Image, self._camImageCallback)
        self.camImgMsg = None

        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)

        self.auvFrame = "lolo"
        self.auvPosePublisher = rospy.Publisher(self.auvFrame + "/pose", PoseWithCovarianceStamped, queue_size=1)

        self.auvCameraPosePublisher = rospy.Publisher(self.auvFrame + "/camera/pose", PoseWithCovarianceStamped, queue_size=1)

        self.dockingStationFrame = "docking_station"
        self.dockingStationPosePublisher = rospy.Publisher(self.dockingStationFrame + "/pose", PoseWithCovarianceStamped, queue_size=1)
        self.featurePosesPublisher = rospy.Publisher(self.dockingStationFrame + "/feature_model/poses", PoseArray, queue_size=1)

        self.listener = tf.TransformListener()

        self.auvOdometryPub = rospy.Publisher("core/odometry", Odometry, queue_size=1)
        self.rudderSub = rospy.Subscriber('core/rudder_cmd', Float32, self._rudderCallback)
        self.elevatorSub = rospy.Subscriber('core/elevator_cmd', Float32, self._elevatorCallback)
        #self.elevon_stbd_angle = rospy.Publisher('core/elevon_strb_cmd', Float32, queue_size=1)
        #self.elevon_port_angle = rospy.Publisher('core/elevon_port_cmd', Float32, queue_size=1)
        self.thrusterSub = rospy.Subscriber('core/thruster_cmd', ThrusterRPM, self._thrusterCallback)

        self._twistControlMsg = None
        self.twistControlCallbackSubscriber = rospy.Subscriber("lolo/twist_command", TwistStamped, self._twistControlCallback)

        self.resetService = rospy.Service("sim/reset", Trigger, self._resetCallback)

        self.velocityMode = velocityMode # False: control mode, True: velocity mode
        self.auvToControl = self.dockingStation # start by manual control input to be sent to docking station 

    def _resetCallback(self, req):
        self.auv.reset()
        self.dockingStation.reset()
        self._velDockingStation = self._velDockingStationInit.copy()
        self._velAUV = self._velAUVInit.copy()
        self._controlCommandDockingStation = self._controlCommandDockingStationInit.copy()
        self._controlCommandAUV = self._controlCommandAUVInit.copy()
        return TriggerResponse(
            success=True,
            message="Resetting simulation"
        )

    def _camImageCallback(self, msg):
        self.camImgMsg = msg

    def _twistControlCallback(self, msg):
        self._twistControlMsg = msg

    def _rudderCallback(self, msg):
        self._controlCommandAUV[1] = msg.data

    def _elevatorCallback(self, msg):
        self._controlCommandAUV[2] = msg.data

    def _thrusterCallback(self, msg):
        self._controlCommandAUV[0] = msg.rpm

    def _publish(self):
        timeStamp = rospy.Time.now()

        self.auvPosePublisher.publish(
                                      vectorToPose("ned",
                                      self.auv.translationVector, 
                                      self.auv.rotationVector, 
                                      np.zeros((6,6)),
                                      timeStamp=timeStamp)
                                    )

        rAUV = R.from_rotvec(self.auv.rotationVector)
        rCam = rAUV*R.from_rotvec(self.auv.relativeCameraRotation)
        self.auvCameraPosePublisher.publish(
                            vectorToPose("ned",
                                         self.auv.translationVector + rAUV.apply(self.auv.relativeCameraTranslation),
                                         rCam.as_rotvec(),
                                         np.zeros((6,6)),
                                         timeStamp=timeStamp,
                                        )
        )

        self.dockingStationPosePublisher.publish(
                                                vectorToPose("ned",
                                                             self.dockingStation.translationVector, 
                                                             self.dockingStation.rotationVector, 
                                                             np.zeros((6,6)),
                                                             timeStamp=timeStamp)
                                                )
        
        nedToOdom = vectorToTransform("odom",
                                      "ned",
                                      np.array([0, 0, 0]),
                                      np.array([np.pi, 0, 0]),
                                      timeStamp=timeStamp)

        auvTransformNED = vectorQuatToTransform("ned", 
                                         self.auvFrame + "/base_link_ned", 
                                         self.auv.translationVector, 
                                         self.auv.q, 
                                         timeStamp=timeStamp)

        auvTransform = vectorToTransform(self.auvFrame + "/base_link_ned", 
                                         self.auvFrame + "/base_link", 
                                         np.array([0., 0., 0.]), 
                                         np.array([np.pi, 0, 0]), 
                                         timeStamp=timeStamp)

        cameraTransform = vectorToTransform(self.auvFrame + "/base_link_ned", 
                                            self.auvFrame + "_camera_link", 
                                            self.auv.relativeCameraTranslation, 
                                            self.auv.relativeCameraRotation, 
                                            timeStamp=timeStamp)
                                                

        dockingStationTransformNED = vectorQuatToTransform("ned", 
                                                    self.dockingStationFrame + "/base_link_ned", 
                                                    self.dockingStation.translationVector, 
                                                    self.dockingStation.q, 
                                                    timeStamp=timeStamp)

        dockingStationTransform = vectorToTransform(self.dockingStationFrame + "/base_link_ned", 
                                                    self.dockingStationFrame + "/base_link", 
                                                    np.array([0., 0., 0.]), 
                                                    np.array([np.pi, 0, 0]),
                                                    timeStamp=timeStamp)

        featureModelTransform = vectorToTransform(self.dockingStationFrame + "/base_link_ned", 
                                            self.dockingStationFrame + "/feature_model_link", 
                                            self.dockingStation.relativeFeatureModelTranslation, 
                                            self.dockingStation.relativeFeatureModelRotation, 
                                            timeStamp=timeStamp)

        pArray = featurePointsToMsg(self.dockingStationFrame + "/feature_model_link", 
                                    self.featureModel.features, 
                                    timeStamp=timeStamp)
        self.featurePosesPublisher.publish(pArray)

        self._publishAUVOdometry(timeStamp)

        self.transformPublisher.publish(tf.msg.tfMessage([nedToOdom, 
                                                          dockingStationTransformNED,
                                                          dockingStationTransform, 
                                                          featureModelTransform, 
                                                          auvTransformNED,
                                                          auvTransform, 
                                                          cameraTransform]))

        self._publishImage(timeStamp)


    def _publishAUVOdometry(self, timeStamp):
        state = self.auv.state()
        msg = stateToOdometry("ned", 
                              self.auvFrame + "/base_link_ned", 
                              state, 
                              timeStamp=timeStamp)
        self.auvOdometryPub.publish(msg)

    def _publishImage(self, timeStamp):
        try:
            trueTrans, trueRot = self.listener.lookupTransform(self.auvFrame + "_camera_link", 
                                                               self.dockingStationFrame + "/feature_model_link", 
                                                               rospy.Time(0))
        except:
            return

        self.cameraInfo.header.frame_id = self.auvFrame + "_camera_link"
        self.cameraInfo.header.stamp = timeStamp
        self.cameraInfoPublisher.publish(self.cameraInfo)

        trueTrans = np.array(trueTrans, dtype=np.float32)
        trueRot = np.array(trueRot, dtype=np.float32)
        trueRot = R.from_quat(trueRot).as_rotvec()
        imagePoints = projectPoints(trueTrans, trueRot, self.camera, self.featureModel.features)

        minDist = np.inf
        for i, imgP in enumerate(imagePoints):
            for imgPOther in imagePoints[i+1:]:
                dist = np.linalg.norm([imgP[0]-imgPOther[0], imgP[1]-imgPOther[1]])
                if dist < minDist:
                    minDist = dist

        imgColor = np.zeros((self.camera.resolution[0], self.camera.resolution[1], 3), dtype=np.uint8)
        
        dist = np.linalg.norm(trueTrans)
        k = 13.
        m=10
        I = 255*np.exp(-dist/(k+m))
        size = 70*np.exp(-dist/k)
        size = int(size)
        size = min(size, int(minDist/2))
        size = max(size, 1)
        for p in imagePoints:
            p = int(round(p[0])), int(round(p[1]))
            
            for r in range(size, 0, -1):
                intensity = I*np.exp(-r/k)
                cv.circle(imgColor, p, r, (intensity, intensity, intensity), -1)

        if self.camImgMsg:
            imgColor = imgColor.astype(np.float32) + self.cvBridge.imgmsg_to_cv2(self.camImgMsg, 'bgr8').astype(np.float32)
            imgColor[imgColor > 255] = 255
            imgColor = imgColor.astype(np.uint8)

        imgMsg = self.cvBridge.cv2_to_imgmsg(imgColor, 'bgr8')
        imgMsg.header.stamp = timeStamp
        self.imagePublisher.publish(imgMsg)

    def plotControlInfo(self, controlImg):

        modeText = "Velocity" if self.velocityMode else "Control"
        cv.putText(controlImg, 
                   "Mode: {}".format(modeText), 
                   (5, 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, 
                   thickness=1, 
                   color=(0,255,0))

        auvColor = (150, 150, 150)
        dsColor = (0, 255, 0)
        if self.auvToControl == self.auv:
            auvColor = (0, 255, 0)
            dsColor = (150, 150, 150)

        cv.putText(controlImg, 
                    "Mothership", 
                    (5, 40), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1, 
                    color=dsColor)
        cv.putText(controlImg, 
                    "AUV", 
                    (105, 40), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1, 
                    color=auvColor)
        org = (5, 60)
        for symbol, vDS, vAUV in zip(["n", "dR", "dE"], self._controlCommandDockingStation, self._controlCommandAUV):
            cv.putText(controlImg, 
                        "{} - {}".format(symbol, round(vDS, 2)), 
                        org, 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, 
                        thickness=1, 
                        color=dsColor)

            cv.putText(controlImg, 
                        "{} - {}".format(symbol, round(vAUV, 2)), 
                        (org[0]+100, org[1]), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, 
                        thickness=1, 
                        color=auvColor)

            org = (org[0], org[1]+15)

        auvVelState = self.auv.state()[6:]
        dsVelState = self.dockingStation.state()[6:]
        stateStr = ["Vx", "Vy", "Vz", "Wx", "Wy", "Wz"]
        for auvVal, dsVal, state in zip(auvVelState, dsVelState, stateStr):
            cv.putText(controlImg, 
                    "{} - {}".format(state, round(dsVal, 2)), 
                    org, 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1, 
                    color=dsColor)
            cv.putText(controlImg, 
                    "{} - {}".format(state, round(auvVal, 2)), 
                    (org[0]+100, org[1]), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1, 
                    color=auvColor)
            org = (org[0], org[1]+15)

    def _update(self, controlCommand, i, dt):

        if self.velocityMode:
            self.dockingStation.move(self._velDockingStation, dt)

            vel = self._velAUV.copy()
            if self._twistControlMsg:
                # TODO: This might be werid (adding vel), but useful when testing PBVS and IBVS
                vel += twistToVel(self._twistControlMsg)
                self._twistControlMsg = None
            self.auv.move(vel, dt)
        else:
            n, deltaR, deltaE = self._controlCommandDockingStation
            self.dockingStation.controlIntegrate(n, deltaR, deltaE, dt)

            n, deltaR, deltaE = self._controlCommandAUV
            self.auv.controlIntegrate(n, deltaR, deltaE, dt)

    def update(self, i):

        ###############################
        self._update(None, i, self.dt)
        self._publish()

    def run(self):
        rate = rospy.Rate(self.hz)
        i = 0
        while not rospy.is_shutdown():
            controlImg = np.zeros((200,200,3), dtype=np.uint8)

            self.plotControlInfo(controlImg)

            cv.imshow("control", controlImg)
            
            key = cv.waitKey(1)

            if self.auvToControl == self.auv:
                controlCommandArray = self._controlCommandAUV
                velocityCommandArray = self._velAUV
            else:
                controlCommandArray = self._controlCommandDockingStation
                velocityCommandArray = self._velDockingStation

            w = 0.05
            if key == ord("w"):
                velocityCommandArray[0] += 0.1
                controlCommandArray[0] += 10
            elif key == ord("s"):
                velocityCommandArray[0] -= 0.1
                controlCommandArray[0] = 0
            elif key == ord("a"):
                velocityCommandArray[5] = -w
                controlCommandArray[1] -= 0.02
            elif key == ord("d"):
                velocityCommandArray[5] = w
                controlCommandArray[1] += 0.02
            elif key == ord("i"):
                velocityCommandArray[4] = -w
                controlCommandArray[2] -= 0.02
            elif key == ord("k"):
                velocityCommandArray[4] = w
                controlCommandArray[2] += 0.02   
            elif key == ord("j"):
                velocityCommandArray[3] = -w
            elif key == ord("l"):
                velocityCommandArray[3] = w    
            elif key == ord("p"):
                self.pause = not self.pause
            elif key == ord("r"):
                self._resetCallback(None)
            elif key == ord("m"):
                self.velocityMode = not self.velocityMode
            elif key == ord("z"):
                if self.auvToControl == self.auv:
                    self.auvToControl = self.dockingStation
                else:
                    self.auvToControl = self.auv
            else:
                velocityCommandArray[1:] *= 0
            if not self.pause:
                self.update(i)
                i += 1
            rate.sleep()

if __name__ == "__main__":
    
    from lolo_perception.feature_model import FeatureModel
    from lolo_perception.camera_model import Camera
    import os
    import rospkg

    rospy.init_node("simulation_node")

    featureModelYaml = rospy.get_param("~feature_model_yaml")
    featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    featureModel = FeatureModel.fromYaml(featureModelYamlPath)

    cameraYaml = rospy.get_param("~camera_yaml")
    cameraYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "camera_calibration_data/{}".format(cameraYaml))
    camera = Camera.fromYaml(cameraYamlPath)
    cameraInfo = readCameraYaml(cameraYamlPath)

    velocityMode = rospy.get_param("~velocity_mode")

    sim = ROSSimulator(camera, cameraInfo, featureModel, velocityMode)
    sim.run()
