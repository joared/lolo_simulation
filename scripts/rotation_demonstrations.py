
from scipy.spatial.transform import Rotation as R
import scipy
import matplotlib.pyplot as plt
import numpy as np
from coordinate_system import CoordinateSystem, CoordinateSystemArtist

def howRollCanBeAffectedByAngleAxis():
    eulerOrder = "ZYX"
    euler = (-np.pi/2, -np.pi/4, 0)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    s = 2
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_zlim(-s, s)

    c = CoordinateSystem(translation=(-1,0,0))
    cArt = CoordinateSystemArtist(c)

    c2 = CoordinateSystem(translation=(1,0,0))
    cArt2 = CoordinateSystemArtist(c2)

    r = R.from_euler(eulerOrder, euler)
    rotation = r.as_rotvec().transpose()
    rotation2 = rotation.copy()
    nIter = 100

    yPositions = []
    yPositions2 = []
    cArt.init(ax)
    cArt2.init(ax)
    for i in range(nIter):
        # Transform and project estimated measured points

        rot = (i*rotation/nIter)
        r = R.from_rotvec(rot.transpose())

        if i < nIter/2:
            plt.title("Rotating around Z")
            #rot2 = [0, 0, (2*i*euler[0]/nIter)]
            rot2 = [0, 0, (2*i*rotation2[2]/nIter)]
            r2 = R.from_rotvec(rot2)
        else:
            plt.title("Rotating around Y")
            rot2 = [0, 0, rotation2[2]]
            r2 = R.from_rotvec(rot2)
            #rot2 = [0, ((i-nIter/2)*euler[1]/(nIter-nIter/2)), 0]
            rot2 = [0, ((i-nIter/2)*rotation2[1]/(nIter-nIter/2)), 0]
            r2 = r2 * R.from_rotvec(rot2)

        #yPositions.append(r.as_matrix()[:, 1])
        ls = c.transformedPoints([[0, 0, 0], list(rot)])
        ax.plot3D(*zip(*ls), color="m")
        yPositions.append(c.transformedPoints([[0, 1, 0]])[0])
        ax.plot3D(*zip(*yPositions), color="g")

        ls2 = c2.transformedPoints([[0, 0, 0], list(rot2)])
        ax.plot3D(*zip(*ls2), color="m")
        yPositions2.append(c2.transformedPoints([[0, 1, 0]])[0])
        ax.plot3D(*zip(*yPositions2), color="g")

        c.rotation = list(r.as_matrix())
        c2.rotation = list(r2.as_matrix())
        cArt.update()
        cArt2.update()
        plt.pause(0.01)
    plt.show()

def howLeftRightMultiplicationAffectsRotation():
    """
    Multiplying from the left causes "fixed axis" rotation
    Multiplying from the left causes "relative axis" rotation
    http://web.cse.ohio-state.edu/~wang.3602/courses/cse5542-2013-spring/6-Transformation_II.pdf
    """
    c = CoordinateSystem(translation=(-1,0,0))
    cArt = CoordinateSystemArtist(c)

    c2 = CoordinateSystem(translation=(1,0,0))
    cArt2 = CoordinateSystemArtist(c2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    cArt.init(ax)
    cArt2.init(ax)
    for i in range(400):
        if i < 100:
            plt.title("Rotating around Z")
            w = [0, 0, np.pi/200]
        elif i < 200:
            plt.title("Rotating around X")
            w = [np.pi/200, 0, 0]
        elif i < 300:
            plt.title("Rotating around Z")
            w = [0, 0, np.pi/200]
        else:
            plt.title("Rotating around Y")
            w = [0, np.pi/200, 0]
        
        rotMat = R.from_rotvec(w).as_matrix()
        
        s1 = np.matmul(np.array(c.rotation), rotMat) # left coordinatesystem
        s2 = np.matmul(rotMat, np.array(c2.rotation))# right coordinatesystem
        
        # Equivalent: r = r * exp(skew(r))
        #s1 = np.matmul(np.array(c.rotation), scipy.linalg.expm(skew(w)))
        #s2 = np.matmul(scipy.linalg.expm(skew(w)), np.array(c2.rotation))
        
        c.rotation = list(s1)
        c2.rotation = list(s2)
        cArt.update()
        cArt2.update()
        plt.pause(0.01)

def skew(m):
    return [[   0, -m[2],  m[1]], 
            [ m[2],    0, -m[0]], 
            [-m[1], m[0],     0]]

if __name__ == "__main__":
    howRollCanBeAffectedByAngleAxis()
    #howLeftRightMultiplicationAffectsRotation()

