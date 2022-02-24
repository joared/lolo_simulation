import numpy as np
from scipy.spatial.transform import Rotation as R
from coordinate_system import CoordinateSystem, CoordinateSystemArtist

def polygon(rad, n, shift=False, zShift=0):
    """
    Creates points in the xy-plane
    """
    theta = 2*np.pi/n
    if shift is True:
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

    return points

def polygons(rads, ns, shifts, zShifts):
    assert len(rads) == len(ns) == len(shifts) == len(zShifts), "All args need to be the same length"
    points = None
    for r, n, s, z in zip(rads, ns, shifts, zShifts):
        if points is None:
            points = polygon(r, n, s, z)
        else:
            points = np.append(points, polygon(r, n, s, z), axis=0)
    return points


class FeatureModel(CoordinateSystem):
    def __init__(self, features, euler=(0, 0, 0), *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        
        self.features = features
        """
        for r, n, s, z in zip(rads, ns, shifts, zShifts):
            if self.features is None:
                self.features = polygon(r, n, s, z)
            else:
                self.features = np.append(self.features, polygon(r, n, s, z), axis=0)
        """
        if self.features is not None:
            rotMat = R.from_euler("XYZ", euler).as_dcm()
            self.features = np.matmul(rotMat, self.features[:, :3].transpose()).transpose()
            self.features = self.features[:, :3].copy() # Don't need homogenious

        self.maxRad = max([np.linalg.norm(f) for f in self.features])


class FeatureModelArtist3D(CoordinateSystemArtist):
    def __init__(self, feature, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, feature, *args, **kwargs)
        self.feature = feature
        
        self.features3D = None
        #self.target3D = None    
        #self.featuresProj3D = None
        #self.referenceLines = []

    def artists(self):
        return [self.features3D] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.features3D = ax.plot3D([], [], [], marker="o", color="m")[0]
        #self.features3D = ax.scatter([], [], [], color="navy")

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)
        points = self.feature.transformedPoints(self.feature.features, referenceTranslation)
        self.features3D.set_data_3d(*zip(*points + [points[0]])) # line
        #self.features3D._offsets3d = [*zip(*points + [points[0]])] # scatter

        return self.artists()


smallPrototype5 = FeatureModel(polygons([0, 0.06], 
                                        [1, 4], 
                                        [False, True], 
                                        [0.043, 0]))

smallPrototype9 = FeatureModel(polygons([0, 0.06], 
                                        [1, 8], 
                                        [False, True], 
                                        [0.043, 0]))

bigPrototype5 = FeatureModel(np.array([[0, 0, 0.465], 
                                       [-0.33, -0.2575, 0], 
                                       [0.33, -0.2575, 0], 
                                       [0.33, 0.2575, 0], 
                                       [-0.33, 0.2575, 0]]))

bigPrototype52 = FeatureModel(np.array([[0.04, -0.2575-0.05, 0], 
                                       [-0.33, -0.2575, 0], 
                                       [0.33, -0.2575, 0], 
                                       [0.33, 0.2575, 0], 
                                       [-0.33, 0.2575, 0]]))
        
idealModel = FeatureModel(polygons([0, 1], 
                                   [1, 4], 
                                   [False, True], 
                                   [0.7167, 0]))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fm = FeatureModel([0.06, 0], [4, 1], [False, False], [0, 0.043])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(*zip(*fm.features))
    size = 0.1
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    plt.show()