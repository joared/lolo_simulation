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
from smarc_msgs.msg import ThrusterRPM
from std_srvs.srv import Trigger, TriggerResponse

from lolo_perception.perception_utils import projectPoints
from lolo_perception.perception_ros_utils import vectorToPose, vectorQuatToTransform, vectorToTransform, featurePointsToMsg, readCameraYaml
from lolo_simulation.auv import AUV, DockingStation
from lolo_control.control_utils import twistToVel, stateToOdometry
from scipy.spatial.transform import Rotation as R

import cv2 as cv

class Simulator:
    def __init__(self, 
                 camera,
                 cameraInfo,
                 featureModel):

        self.camera = camera
        self.cameraInfo = cameraInfo
        
        self.featureModel = featureModel

        """
        self.auv = AUV(relativeCameraTranslation=np.array([-2.5, 0, 0.33]),
                       relativeCameraRotation=R.from_euler("XYZ", (-np.pi/2, -np.pi/2, 0)).as_rotvec(),
                       translationVector=np.array([4., -1., 1]),
                       rotationVector=np.array([0., 0., 0.]))
        self.dockingStation = DockingStation(translationVector=np.array([-4., 0., 1.5]),
                                             rotationVector=np.array([0., 0., -0.02]))
        """
        self.auv = AUV(relativeCameraTranslation=np.array([-2.5, 0, -0.33]),
                       relativeCameraRotation=R.from_euler("XYZ", (0, -np.pi/2, -np.pi/2)).as_rotvec(),
                       translationVector=np.array([1., 0., -1]),
                       rotationVector=np.array([0., 0., np.pi/2]))
        self.dockingStation = DockingStation(translationVector=np.array([0., -10., -1.2]),
                                             rotationVector=np.array([0., 0., np.pi/2]))

        # Move the camera towards the detected lights (yaw)
        self.controlCamera = False

        self.hz = 10 # publish rate
        self.dt = 1./self.hz
        self.pause = False
        # TODO: currently camera rate is set in rviz, change so its done here instead
        #self.camHz = 10. # camera update rate
        #self.imageUpdateIndex = self.hz / self.camHz # image capture update index

        self._velDockingStation = np.array([0., 0, 0, 0, 0, 0])
        self._velAUV = np.array([0., 0, 0, 0, 0, 0])

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
        #self.controlCallbackSubscriber = rospy.Subscriber("lolo/twist_command", TwistStamped, self._controlCallback)

        self.resetService = rospy.Service("sim/reset", Trigger, self._resetCallback)

    def _resetCallback(self, req):
        self.auv.reset()
        self.dockingStation.reset()
        self._velDockingStation = np.array([1., 0, 0, 0, 0, 0])
        self._velAUV = np.array([1., 0, 0, 0, 0, 0])
        self._controlCommandDockingStation = self._controlCommandDockingStationInit.copy()
        self._controlCommandAUV = self._controlCommandAUVInit.copy()
        return TriggerResponse(
            success=True,
            message="Resetting simulation"
        )

    def _camImageCallback(self, msg):
        self.camImgMsg = msg

    def _controlCallback(self, msg):
        self._velAUV = twistToVel(msg)

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

        pArray = featurePointsToMsg(self.dockingStationFrame + "/feature_model_link", self.featureModel.features, timeStamp=timeStamp)
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

        imgColor = np.zeros((self.camera.resolution[0], self.camera.resolution[1], 3), dtype=np.uint8)
        
        dist = np.linalg.norm(trueTrans)
        k = 13.
        m=10
        I = 255*np.exp(-dist/(k+m))
        size = 70*np.exp(-dist/k)
        size = int(size)
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
        cv.putText(controlImg, 
                    "Mothership", 
                    (5, 10), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1, 
                    color=(0,255,0))
        cv.putText(controlImg, 
                    "AUV", 
                    (105, 10), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1, 
                    color=(0,255,0))
        org = (5, 30)
        for symbol, vDS, vAUV in zip(["n", "dR", "dE"], self._controlCommandDockingStation, self._controlCommandAUV):
            cv.putText(controlImg, 
                        "{} - {}".format(symbol, round(vDS, 2)), 
                        org, 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, 
                        thickness=1, 
                        color=(0,255,0))

            cv.putText(controlImg, 
                        "{} - {}".format(symbol, round(vAUV, 2)), 
                        (org[0]+100, org[1]), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, 
                        thickness=1, 
                        color=(0,255,0))

            org = (org[0], org[1]+15)

        cv.putText(controlImg, 
                   "Vx - {}".format(round(self.dockingStation.state()[6], 2)), 
                   org, 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, 
                   thickness=1, 
                   color=(0,255,0))
        cv.putText(controlImg, 
                   "Vx - {}".format(round(self.auv.state()[6], 2)), 
                   (org[0]+100, org[1]), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, 
                   thickness=1, 
                   color=(0,255,0))

    def _update(self, controlCommand, i, dt):
        #self.dockingStation.moveMotionModel(self._velDockingStation, dt)
        #self.auv.moveMotionModel(self._velAUV, dt)
        n, deltaR, deltaE = self._controlCommandDockingStation
        #n, deltaR, deltaE = self._controlCommandAUV
        self.dockingStation.controlIntegrate(n, deltaR, deltaE, dt)
        #self.dockingStation.move(self._velDockingStation, dt)

        n, deltaR, deltaE = self._controlCommandAUV
        #n, deltaR, deltaE = self._controlCommandDockingStation
        self.auv.controlIntegrate(n, deltaR, deltaE, dt)
        #self.auv.move(self._velDockingStation, dt)
        #self.auv.move(self._velAUV, dt)

        if self.controlCamera:
            #Rotating the camera towards the detected feature model
            try:
                trueTrans, trueRot = self.listener.lookupTransform(self.auvFrame + "_camera_link", 
                                                                self.dockingStationFrame + "/feature_model_estimated_link", 
                                                                rospy.Time(0))
            except:
                #self.auv.controlCamera(0, P=0.01)
                return
            

            deltaYaw = np.arctan(trueTrans[0]/trueTrans[2])
            self.auv.controlCameraDelta(deltaYaw, P=0.01)

    def update(self, i):

        ###############################
        self._update(None, i, self.dt)
        self._publish()

    def run(self):
        rate = rospy.Rate(self.hz)

        i = 0
        while not rospy.is_shutdown():
            controlImg = np.zeros((100,200,3), dtype=np.uint8)

            self.plotControlInfo(controlImg)

            cv.imshow("control", controlImg)
            
            key = cv.waitKey(1)

            w = 0.05
            if key == ord("w"):
                self._velDockingStation[0] += 0.1
                self._controlCommandDockingStation[0] += 10
            elif key == ord("s"):
                self._velDockingStation[0] -= 0.1
                self._controlCommandDockingStation[0] = 0
            elif key == ord("a"):
                self._velDockingStation[5] = -w
                self._controlCommandDockingStation[1] -= 0.02
            elif key == ord("d"):
                self._velDockingStation[5] = w
                self._controlCommandDockingStation[1] += 0.02
            elif key == ord("i"):
                self._velDockingStation[4] = -w
                self._controlCommandDockingStation[2] -= 0.02
            elif key == ord("k"):
                self._velDockingStation[4] = w
                self._controlCommandDockingStation[2] += 0.02   
            elif key == ord("j"):
                self._velDockingStation[3] = -w
            elif key == ord("l"):
                self._velDockingStation[3] = w    
            elif key == ord("p"):
                self.pause = not self.pause
            elif key == ord("r"):
                self._resetCallback(None)
            else:
                self._velDockingStation = np.array([self._velDockingStation[0], 0, 0, 0, 0, 0])
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

    sim = Simulator(camera, cameraInfo, featureModel)
    sim.run()
