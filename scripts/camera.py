from numpy.core.defchararray import index
from scipy.spatial.transform import Rotation as R
import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist
import sys
sys.path.append("../../perception/scripts")
from feature_simulation import scatterPoint

class Camera(CoordinateSystem):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, distCoeffs, resolution, pixelWidth, pixelHeight, hz, *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        self.cameraMatrixPixel = cameraMatrix
        self._cameraMatrix = cameraMatrix.copy()
        self._cameraMatrix[0, :] *= pixelWidth
        self._cameraMatrix[1, :] *= pixelHeight
        
        self.distCoeffs = distCoeffs
        self.resolution = resolution # maybe change resolution to be (width, height) instead of (height, width)
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.imWidth = self.resolution[1]*self.pixelWidth
        self.imHeight = self.resolution[0]*self.pixelHeight
        self.cx = self.cameraMatrix[0][2]
        self.cy = self.cameraMatrix[1][2]
        self.hz = hz # assume that perception algorithm can perform at this rate

        # Points
        f = self.cameraMatrix[1][1] # focal length (take average between fx and fy?)
        self.f = f
        self.imSize = 1

    @property
    def cameraMatrix(self):
        return self._cameraMatrix.copy()

    def project(self, point):
        """
        TODO: Handle singular matrix
        """
        l0 = np.array(point) # position vector for line
        vl = l0 - np.array(self.translation) # direction vector for line
        p0 = np.array(self.translation) + np.array(self.rotation)[:, 2]*self.f # center of plane
        mat = np.column_stack((np.array(self.rotation)[:, 0], np.array(self.rotation)[:, 1], -vl))
        x = np.linalg.solve(mat, l0-p0)

        # Check if points are within image plane
        if not self.pointWithinImagePlane(x[0], x[1]):
            print("Projection: Point not within image plane")
            return False

        # Check if the projection is from the "front" of the camera/plane
        if np.dot(np.array(self.rotation)[:, 2], vl) < self.f:
            print("Projection: Point behind image plane")
            return False

        return l0 + x[2]*vl

    def projectLocal3D(self, point):
        """
        TODO: Handle singular matrix
        Use open cv project instead?
        """
        l0 = np.array(point) # position vector for line
        vl = l0
        p0 = np.array([0, 0, self.f]) # center of plane
        mat = np.column_stack((np.array([1, 0, 0]).transpose(), np.array([0, 1, 0]).transpose(), -vl))
        x = np.linalg.solve(mat, l0-p0)

        if not self.pointWithinImagePlane(x[0], x[1]):
            return False

        # Check if the projection is from the "front" of the camera/plane
        #if np.dot(np.array([0, 0, 1]), vl) < self.f:

        return l0 + x[2]*vl

    def pointWithinImagePlane(self, x, y):
        x = x + self.cx
        y = y + self.cy
        return x > 0 and x < self.imWidth and y > 0 and y < self.imHeight

    def uvToMeters(self, points):
        points[:, 0] *= self.pixelWidth
        points[:, 1] *= self.pixelHeight
        return points

    def metersToUV(self, points):
        points = points.copy()
        points[:, 0] /= self.pixelWidth
        points[:, 1] /= self.pixelHeight
        return points

class CameraArtist(CoordinateSystemArtist):
    def __init__(self, camera, featureModel, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, camera, *args, **kwargs)

        self.camera = camera
        self.nFeaturesToBeDetected = len(featureModel.features)

        imSize = self.camera.imSize
        f = self.camera.f
        self._topLine = [[0, 0, 0,], [0, -imSize/2, f]]
        self._focalLine = [[0, 0, 0], [0, 0, f]]
        self._imagePoints = [[imSize/2, -imSize/2, f], 
                             [-imSize/2, -imSize/2, f], 
                             [-imSize/2, imSize/2, f], 
                             [imSize/2, imSize/2, f]]

        self.imagePoints = None
        self.focalLine = None
        self.topLine = None

    def artists(self):
        return [self.focalLine, 
                self.topLine, 
                self.imagePoints] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.focalLine = ax.plot3D([], [], [], color="black")[0]
        self.topLine = ax.plot3D([], [], [], color="black")[0]
        self.imagePoints = ax.plot3D([], [], [], color="r")[0]

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)
        points = self.camera.transformedPoints(self._imagePoints, referenceTranslation)
        topLine = self.camera.transformedPoints(self._topLine, referenceTranslation)
        focalLine = self.camera.transformedPoints(self._focalLine, referenceTranslation)

        self.focalLine.set_data_3d(*zip(*focalLine))
        self.topLine.set_data_3d(*zip(*topLine))
        self.imagePoints.set_data_3d(*zip(*points + [points[0]]))  

        return self.artists()

class CameraImageCaptureArtist:
    def __init__(self, camera, featureModel, updateIndex):
        self.camera = camera
        self.featureModel = featureModel

        self._featurePoints3D = []
        self._detectedFeaturePoints = []

        self.projectedFeaturePoints = None
        self.featurePoints3D = None
        self.referenceLines = []

        self.updateIndex = updateIndex

    def artists(self):
        return [self.projectedFeaturePoints, 
                self.featurePoints3D] + self.referenceLines

    def init(self, ax):
        self.projectedFeaturePoints = ax.plot3D([], [], [], color="cornflowerblue", marker=".")[0] # plot3D
        self.featurePoints3D = ax.plot3D([], [], [], color="red", marker="o")[0] # plot3D
        #self.projectedFeaturePoints = ax.scatter([], [], [], color="cornflowerblue", marker=".") # scattering
        self.referenceLines = []
        for _ in range(len(self.featureModel.features)): # hard coded
            self.referenceLines.append(ax.plot3D([], [], [], color="darkslategray")[0])

        return self.artists()

    def update(self, i, referenceTranslation=(0,0,0)):
        #if i % self.updateIndex == 0:
        projPoints = []
        featurePoints3D = []
        for pp, f in zip(self._detectedFeaturePoints, self._featurePoints3D):
            if pp is not False:
                projPoints.append(pp)
                featurePoints3D.append(f)

        if projPoints:
            projPoints = self.camera.transformedPoints(projPoints, referenceTranslation) # local to global
            self.projectedFeaturePoints.set_data_3d(*zip(*projPoints + [projPoints[0]]))
            featurePoints3D = self.camera.transformedPoints(featurePoints3D, referenceTranslation) # local to global
            self.featurePoints3D.set_data_3d(*zip(*featurePoints3D + [featurePoints3D[0]]))
        else:
            self.projectedFeaturePoints.set_data_3d([], [], [])
            self.featurePoints3D.set_data_3d([], [], [])   
        
        if not self._featurePoints3D:
            for l in self.referenceLines:
                l.set_data_3d([], [], [])
        for l, pp, f in zip(self.referenceLines, self._detectedFeaturePoints, self._featurePoints3D):
            if pp is not False:
                pp = self.camera.transformedPoints([pp], referenceTranslation)[0]
                f = self.camera.transformedPoints([f], referenceTranslation)[0]
                l.set_data_3d(*zip(pp, f))
            else:
                l.set_data_3d([], [], [])

        return self.artists()

    def _captureImage(self, featurePoints3D, detectedPoints):

        self._featurePoints3D = featurePoints3D
        self._detectedFeaturePoints = detectedPoints

    def captureImage(self):
        featurePoints3D = self.featureModel.transformedPoints(self.featureModel.features)
        featurePoints3D = self.camera.transformedPointsInv(list(featurePoints3D))
        detectedPoints = []
        for feature in featurePoints3D:
            detectedPoint = self.camera.projectLocal3D(feature)
            if detectedPoint is not False:
                sigmaX = 0
                sigmaY = 0
                noisedPoint = (detectedPoint[0] + np.random.normal(0, sigmaX*self.camera.pixelWidth),
                                detectedPoint[1] + np.random.normal(0, sigmaY*self.camera.pixelHeight),
                                detectedPoint[2])
                if self.camera.pointWithinImagePlane(noisedPoint[0], noisedPoint[1]):
                    detectedPoints.append(noisedPoint)
                else:
                    detectedPoints.append(False)
            else:
                detectedPoints.append(False)

        self._featurePoints3D = featurePoints3D
        self._detectedFeaturePoints = detectedPoints

    def releaseImage(self):
        self._featurePoints3D = []
        self._detectedFeaturePoints = []

    def getImage(self, scatter=True):
        if self._detectedFeaturePoints == []:
            return None

        resolution = self.camera.resolution
        img = np.zeros(resolution)
        imgCoordinates = []
        for p in self._detectedFeaturePoints:
            if p is not False:
                x = p[0] + self.camera.cx
                y = p[1] + self.camera.cy
                u = int(x / self.camera.pixelWidth)
                v = int(y / self.camera.pixelHeight)
                imgCoordinates.append((v, u))
                try:
                    img[v, u] = 255
                except:
                    print("Err:", u, v)
                    print("x,y: {}({}), {}({})".format(x, self.camera.imWidth, y, self.camera.imHeight))
                    raise Exception("This shouldnt happen")
            else:
                imgCoordinates.append(False)

        if scatter:
            for vu, p in zip(imgCoordinates, self._featurePoints3D):
                if vu is not False:
                    r = np.linalg.norm(p)
                    scatterPoint(img, vu, r, 25)
        return img

# this camera matrix is wrong, should use the projection matrix instead since images are already rectified
usbCamera480p = Camera(cameraMatrix=np.array([[812.2540283203125,   0,    		    329.864062734141 ],
                                         [   0,               814.7816162109375, 239.0201541966089], 
                                         [   0,     		     0,   		       1             ]], dtype=np.float32), 
                   distCoeffs=np.zeros((4,1), dtype=np.float32),
                   resolution=(480, 640), 
                   pixelWidth=2.796875e-6, 
                   pixelHeight=2.8055555555e-6, 
                   hz=15)

contourCamera1080p = Camera(cameraMatrix=np.array([[ 884.36572,    0.     ,  994.04928],
                                             [    0.     , 1096.93066,  567.01791],
                                             [    0.     ,    0.     ,    1.     ]], dtype=np.float32), 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(1080, 1920), 
                        pixelWidth=2.8e-6,  # not known
                        pixelHeight=2.8e-6, # not known
                        hz=15)

usbCamera720p = Camera(cameraMatrix=np.array([[1607.87793,    0.     ,  649.18249],
                                              [   0.     , 1609.64954,  293.20127],
                                              [   0.     ,    0.     ,    1.     ]], dtype=np.float32), 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(720, 1280), 
                        pixelWidth=2.796875e-6, 
                        pixelHeight=2.8055555555e-6, 
                        hz=15)

if __name__ == "__main__":
    pass
    


