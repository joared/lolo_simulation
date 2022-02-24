import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

from camera import Camera, CameraArtist, CameraImageCaptureArtist
from auv import AUV, AUVArtist
from feature import smallPrototype5, FeatureModel, FeatureModelArtist3D

from scipy.spatial.transform import Rotation as R

import cv2 as cv

class Simulator:
    def __init__(self, 
                 requestRendezvousCB, 
                 captureImageCB, 
                 controlCB, 
                 dockerArtist, 
                 featureModel):

        self.requestRendezvousCB = requestRendezvousCB
        self.captureImageCB = captureImageCB
        self.controlCB = controlCB
        self.dockerArtist = dockerArtist

        self.rendezvousReq = None
        self.controlCommand = None

        self.hz = 20 # control update rate
        self.dt = 1/self.hz # changing this affects the camera behaviour
        camHz = 10 # camera update rate

        pixelWidth = 0.0000028
        pixelHeight = 0.0000028
        px = 329.864062734141
        py = 239.0201541966089
        cameraMatrix = np.array([[812.2540283203125,   0,    		        px],
                                [   0,               814.7816162109375, 	py], 
                                [   0,     		     0,   		       	    1]], dtype=np.float32)

        self.camera = Camera(cameraMatrix, 
                             resolution=(480, 640), 
                             distCoeffs=np.zeros((4,1), dtype=np.float32),
                             pixelWidth=0.0000028, 
                             pixelHeight=0.0000028, 
                             hz=camHz)
        self.imageUpdateIndex = self.hz / self.camera.hz # image capture update index
        
        self.auv = AUV(self.camera, 
                       "back",
                       size=(3, 10),
                       euler=(0, 0, 0))

        self.feature = featureModel
        self.dockingStation = AUV(self.feature,
                                  "front",
                                  size=(10, 10),
                                  translation=[-15, -0.5, 0.5],
                                  euler=(0, 0, 0))

        self.cameraArtist = CameraArtist(self.camera, self.feature)
        self.cameraImageCaptureArtist = CameraImageCaptureArtist(self.camera, self.feature, self.imageUpdateIndex)
        self.featureArtist = FeatureModelArtist3D(self.feature)
        self.auvArtist = AUVArtist(self.auv)
        self.dockingStationArtist = AUVArtist(self.dockingStation, color="navy")

        self.focusCoord = self.auv
        self.blit = False
        self.showAxis = True
        self.updateCenterNr = 200

    def on_press(self, event):
        print('press', event.key)
        if event.key == 'x':
            self.showAxis = not self.showAxis
        elif event.key == "c":
            if self.focusCoord == self.auv:
                self.focusCoord = self.camera
            elif self.focusCoord == self.camera:
                self.focusCoord = self.dockingStation
            elif self.focusCoord == self.dockingStation:
                self.focusCoord = self.feature
            else:
                self.focusCoord = self.auv
            self.setAxisLims()
            
        elif event.key == "up":
            if self.size <= 1:
                self.size -= .1
            else:
                self.size -= 1
            self.size = max(self.size, .1)
            self.setAxisLims()
            self.fig.canvas.draw()
        elif event.key == "down":
            if self.size < 1:
                self.size += .1
            else:
                self.size += 1
            self.setAxisLims()
            self.fig.canvas.draw()

    def setAxisLims(self):
        center = self.focusCoord.translation
        if self.centerAxis:
            center = (0, 0, 0)
        
        self.ax.set_xlim3d(-self.size + center[0], self.size + center[0])
        self.ax.set_ylim3d(-self.size + center[1], self.size + center[1])
        self.ax.set_zlim3d(-self.size + center[2], self.size + center[2])            
        
        center = (0, 0, 0)
        if self.centerAxis:
            center = self.focusCoord.translation
        
        if self.blit is True and not self.centerAxis: # is this correct?
        #if self.blit is True:
            self.fig.canvas.draw()

        return center

    def animate(self, anim=True, blit=False, centerAxis=False):
        
        self.blit = blit
        self.centerAxis = centerAxis

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        gs = self.fig.add_gridspec(2, 2)
        self.ax = self.fig.add_subplot(gs[:, :], projection="3d")

        self.ax.set_title("Docking simulator")
        self.size = 15
        # set_aspect("equal") doesn't exist for 3D axes yet
        self.ax.set_xlim3d(-self.size, self.size)
        self.ax.set_ylim3d(-self.size, self.size)
        self.ax.set_zlim3d(-self.size, self.size)

        #self.ax2 = self.fig.add_subplot(gs[:, 1])
        #self.im = self.ax2.imshow(np.zeros(self.camera.resolution), cmap="gray", vmin=0, vmax=255, animated=True)

        self.startTime = 0
        self.elaspsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        if anim:
            self.anim = animation.FuncAnimation(self.fig, self.update, frames=self.timeGen, init_func=self.init,
                                                interval=self.dt*1000, blit=blit)
        
        self.fig.canvas.draw() # bug fix

    def show(self):
        plt.show()

    def timeGen(self):
        #for i in range(1500):
        i = 0
        while True:
            yield i
            i += 1

    def init(self):
        return self.cameraArtist.init(self.ax) \
               + self.cameraImageCaptureArtist.init(self.ax) \
               + self.auvArtist.init(self.ax) \
               + self.featureArtist.init(self.ax) \
               + self.dockingStationArtist.init(self.ax)
               #+ self.visualServoArtist.init(self.ax)

    def _update(self, controlCommand, i, dt):
        """
        Assumes that trajectory planning, perception and controll classes have been defined
        self.trajectoryPlanner
        self.perceptionEstimator
        self.controller
        """
        prevTranslation = self.auv.translation
        prevRot = self.auv.rotation
        # Control first, then capture image
        if self.controlCommand:
            noiseVel = np.random.normal(0, 0.0, (3,))
            vel = np.array(controlCommand[:3]) + noiseVel
            vxBias = 0
            vel[0] += vxBias
            #vel[1] += -0.5
            print("Commandvel:", controlCommand[:3])
            noiseAngularVel = np.random.normal(0, 0.0, (3,))
            angVel = np.array(controlCommand[3:]) + noiseAngularVel
            #angVel[2] += 0.5
            self.auv.move(np.concatenate((vel, angVel)), dt)
            
        #deltaTranslation = 
        #deltaRotation = 

        vel = [1.5, 0, 0, 0, 0, 0.006]
        self.dockingStation.move(vel, dt)

        if i % self.imageUpdateIndex == 0:
            self.cameraImageCaptureArtist.captureImage()
        else:
            self.cameraImageCaptureArtist.releaseImage()

    def update(self, i):
        if i == 0:
            self.startTime = time.time()
        self.elapsed = time.time() - self.startTime

        ###############################
        self._update(self.controlCommand, i, self.dt)

        if self.rendezvousReq:
            self.requestRendezvousCB(self.rendezvousReq)
        img = self.cameraImageCaptureArtist.getImage(scatter=False)
        if img is not None:
            self.captureImageCB(img) # returns None if not captured
        self.controlCommand = self.controlCB("deltaTranslation", "deltaRotation", self.dt)

        if img is not None:
            cv.imshow("Captured image", img)
            #cv.waitKey(1)

        center = (0, 0, 0)
        if self.centerAxis:
            center = self.focusCoord.translation
        if i % self.updateCenterNr == 0:
            self.setAxisLims()

        # bug fix, title/info text not updating otherwise
        # however, really slows down simulation (doesn't work with blitting)
        # self.fig.canvas.draw()
        showAxis = self.showAxis
        return self.cameraArtist.update(showAxis, center) + \
               self.cameraImageCaptureArtist.update(i, center) + \
               self.auvArtist.update(showAxis, center) + \
               self.featureArtist.update(showAxis, center) + \
               self.dockingStationArtist.update(showAxis, center)

class DockerDummy:
    def __init__(self):
        self.errPrev = 0
        self.velAUV = [0]*6
        self.angleErrPrev = [0]*3

    def requestRendezvous(self, req):
        pass

    def captureImage(self, img):
        pass

    def control(self, dum, dummy, dt):
        velAUV, errPrev, angleErrPrev = self.velAUV, self.errPrev, self.angleErrPrev

        refPoint = np.array(sim.feature.transformedPoints([[0, 0, 5]])[0])
        trans = refPoint - np.array(sim.auv.translation)

        err = np.dot(np.array(sim.auv.rotation)[:, 0], trans)
        #print("Error:", err)
        accelAUV = 1*err
        #accelAUV = min(accelAUV, 0.1)
        #accelAUV = 0.005*l
        
        accelAUV = accelAUV + 40*(err-errPrev)
        accelAUVMax = 0.1
        accelAUV = min(accelAUV, accelAUVMax)
        accelAUV = max(accelAUV, -accelAUVMax)
        #print(accelAUV)
        errPrev = err
        vMax = 100#2
        vx = min(velAUV[0] + accelAUV, vMax)
        vMin = 0.5
        vMax = 3
        vx = max(vx, vMin)
        vx = min(vx, vMax)
        velAUV[0] = vx

        l = np.linalg.norm(trans)
        rotRefLength = 5
        featToAUV = np.array(sim.auv.translation).transpose()-np.array(sim.feature.translation).transpose()
        projAUVOnFeatureLine = np.dot(featToAUV.transpose(), -np.array(sim.feature.rotation)[:, 2])
        xDir = -(projAUVOnFeatureLine+rotRefLength)*np.array(sim.feature.rotation)[:, 2] + np.array(sim.feature.translation).transpose() - np.array(sim.auv.translation).transpose()
        xDir = xDir/np.linalg.norm(xDir)
        xDir = xDir.transpose()
        zDir = np.array([0, 0, 1])
        yDir = np.cross(zDir, xDir)
        yDir /= np.linalg.norm(yDir)
        zDir = np.cross(xDir, yDir)
        Rp = np.column_stack( (xDir.transpose(), yDir.transpose(), zDir.transpose()) )
        Rc = np.array(sim.auv.rotation)
        w = R.from_matrix(np.matmul(np.linalg.inv(Rc), Rp)).as_rotvec()
        Rw = R.from_rotvec(w).as_matrix()
        Rerrprev = R.from_rotvec(angleErrPrev).as_matrix()

        wdt = R.from_matrix(np.matmul(np.linalg.inv(Rerrprev), Rw)).as_rotvec() # think this is the one, but both acts similar
        #wdt = R.from_matrix(np.matmul(Rw, np.linalg.inv(Rerrprev))).as_rotvec()

        angleErrPrev = w
        angMax = 0.2
        angVel = w*0.2 + wdt*20
        if np.linalg.norm(angVel) > angMax:
            angVel = angVel/np.linalg.norm(angVel)*angMax
        velAUV[3:] = list(angVel)

        return velAUV


if __name__ == "__main__":
    import sys
    sys.path.append("../../docking/scripts")
    from docking import Docker#, DockerArtist
    from camera import usbCamera480p

    sys.path.append("../../perception/scripts")
    from feature_extraction import ThresholdFeatureExtractor
    from pose_estimation import DSPoseEstimator
    

    featureModel = smallPrototype5#FeatureModel([0.3], [4], [True], [0])
    featureExtractor = ThresholdFeatureExtractor(featureModel, usbCamera480p)

    #docker = Docker()
    docker = DockerDummy()

    sim = Simulator(docker.requestRendezvous, 
                    docker.captureImage, 
                    docker.control, 
                    "dockerArtist", 
                    featureModel)
    velAUV = [1.5, 0, 0, 0, 0, 0]
    sim.animate(anim=True, blit=True, centerAxis=True)
    sim.show()
    exit()

    sim.animate(None, anim=False, blit=True, centerAxis=False)
    sim.init()
    
    for i in range(1000):
        
        ##################################
        control(i, dt)

        ###################################
        sim.update(i)
        plt.pause(dt)

    #sim.animate()
    sim.show()
