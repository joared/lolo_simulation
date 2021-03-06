from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.integrate import solve_ivp

class Submarine:

    def __init__(self, translationVector, rotationVector):
        #self.state = state
        self.translationVector = np.array(translationVector, dtype=np.float32)
        self.rotationVector = np.array(rotationVector, dtype=np.float32)

        self.initTranslationVector = self.translationVector.copy()
        self.initRotationVector = self.rotationVector.copy()

        self.q = R.from_rotvec(self.rotationVector).as_quat()

        self.vel = np.array([0., 0., 0., 0., 0., 0.]) # [u, v, w, p, q, r]

    def state(self):
        state = np.array([0.]*12)
        state[:3] = self.translationVector
        az, ay, ax = R.from_rotvec(self.rotationVector).as_euler("ZYX")
        state[3:6] = ax, ay, az
        state[6:12] = self.vel.copy()
        return state

    def _motionModel(self):
        yaw, pitch, roll = R.from_rotvec(self.rotationVector).as_euler("ZYX")

        cYaw = np.cos(yaw)
        sYaw = np.sin(yaw)
        cPitch = np.cos(pitch)
        sPitch = np.sin(pitch)
        cRoll = np.cos(roll)
        sRoll = np.sin(roll)
        J1 = np.array([[cYaw*cPitch, -sYaw*cRoll+cYaw*sPitch*sRoll, sYaw*sRoll+cYaw*cRoll*sPitch],
                       [sYaw*cPitch, cYaw*cRoll+sRoll*sPitch*sYaw, -cYaw*sRoll+sPitch*sYaw*cRoll],
                       [-sPitch,     cPitch*sRoll,                  cPitch*cRoll]])

        tPitch = np.tan(pitch)
        J2 = np.array([[1, sRoll*tPitch, cRoll*tPitch],
                       [0, cRoll, -sRoll],
                       [0, sRoll/cPitch, cRoll/cPitch]])

        return J1, J2

    def _dynamicModel(self, vel, n, deltaR, deltaE):
        C1 = 1.768*10e-2
        C2 = -0.6
        C3 = 2.974*10e-7
        C4 = -0.4
        C5 = -4.1
        C6 = -0.19
        C7 = 0.22
        C8 = -1.51
        C9 = -1.3
        C10 = 0.21
        C11 = -0.4
        C12 = 2.0
        C13 = 0.4
        C14 = 1.0

        """
        auvHeight = 0.5
        m = 300#0.03
        g = 9.82
        totVolume = 0.5#.04
        V = totVolume/2 + totVolume/auvHeight*self.translationVector[2]
        V = max(min(V, totVolume), 0)
        print(V)
        rho = 1026
        deltaMG = m*g-V*rho*g
        print("deltaMG", deltaMG)
        """
        
        deltaMG = 0.0

        u, v, w, p, q, r = vel

        _, pitch, roll = R.from_rotvec(self.rotationVector).as_euler("ZYX")

        return np.array([C1*u + C2*abs(u)*u + C3*abs(n)*n - deltaMG*np.sin(pitch),
                         deltaMG*np.cos(pitch)*np.sin(roll) + C11*v,
                         deltaMG*np.cos(pitch)*np.cos(roll) + C11*w,
                         -C13*p-C14*np.cos(pitch)*np.sin(roll),
                         C4*q + C5*abs(q)*q + C6*np.sin(pitch) + C7*u**2*deltaE,
                         C8*r + C9*abs(r)*r + C10*u**2*deltaR])

    def controlSimple(self, n, deltaR, deltaE, dt):
        self.moveMotionModel(self.vel, dt)
        vDot = self._dynamicModel(self.vel, n, deltaR, deltaE)
        self.vel += vDot*dt

    def controlIntegrate(self, n, deltaR, deltaE, dt):
        # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html

        # propagate translation and rotation based on current velocity
        J1, J2 = self._motionModel()
        deltaTrans = np.matmul(J1, self.vel[:3]*dt)
        deltaAngle = np.matmul(J2, self.vel[3:]*dt)
        deltaRot = R.from_euler("ZYX", [deltaAngle[2], deltaAngle[1], deltaAngle[0]])
        newRot = R.from_rotvec(self.rotationVector)*deltaRot

        # propagate velocity based on current velocity, using dynamic model
        F = lambda t, vel: self._dynamicModel(vel, n, deltaR, deltaE)
        dtProp = 50 # propagate least this many times/sec
        ratio = np.ceil(dt/dtProp)
        dtProp = dt/ratio
        t_eval = np.arange(0, dt+dtProp, dtProp)
        sol = solve_ivp(F, [0, dt], self.vel, t_eval=t_eval)
        newVel = sol.y[:, -1]
        
        self.translationVector += deltaTrans
        self.rotationVector = newRot.as_rotvec()
        self.q = R.from_rotvec(self.rotationVector).as_quat()
        self.vel = newVel
        
    def move(self, vel, dt):

        """
        # using rotvec [wx, wy, wz]
        deltaTrans = R.from_rotvec(self.rotationVector).apply(vel[:3]*dt)
        self.translationVector += deltaTrans
        newRot = R.from_rotvec(self.rotationVector)*R.from_rotvec(vel[3:]*dt)
        self.rotationVector = newRot.as_rotvec()
        self.q = newRot.as_quat()
        self.vel *= 0
        """
                
        # using NED euler convention [wx, wy, wz]
        J1, J2 = self._motionModel()
        deltaTrans = np.matmul(J1, vel[:3]*dt)
        self.translationVector += deltaTrans

        deltaAngle = np.matmul(J2, vel[3:]*dt)
        deltaRot = R.from_euler("ZYX", [deltaAngle[2], deltaAngle[1], deltaAngle[0]])
        newRot = R.from_rotvec(self.rotationVector)*deltaRot
        self.rotationVector = newRot.as_rotvec()
        self.q = R.from_rotvec(self.rotationVector).as_quat()

    def reset(self):
        self.translationVector = self.initTranslationVector.copy()
        self.rotationVector = self.initRotationVector.copy()
        self.vel *= 0
        self.q = R.from_rotvec(self.initRotationVector).as_quat()

class AUV(Submarine):
    def __init__(self,
                 relativeCameraTranslation=np.array([-3.0/2, 0, 0]),
                 relativeCameraRotation=R.from_euler("XYZ", (-np.pi/2, -np.pi/2, 0)).as_rotvec(), 
                 *args, 
                 **kwargs):
        Submarine.__init__(self, *args, **kwargs)
        self.relativeCameraTranslationInit = relativeCameraTranslation
        self.relativeCameraRotationInit = relativeCameraRotation
        self.relativeCameraTranslation = relativeCameraTranslation
        self.relativeCameraRotation = relativeCameraRotation

class DockingStation(Submarine):
    def __init__(self,
                 relativeFeatureModelTrans=np.array([1, 0, 0]),
                 relativeFeatureModelRot=R.from_euler("XYZ", (0, -np.pi/2, -np.pi/2)).as_rotvec(), 
                 *args, 
                 **kwargs):
        Submarine.__init__(self, *args, **kwargs)
        self.relativeFeatureModelTranslation = relativeFeatureModelTrans
        self.relativeFeatureModelRotation = relativeFeatureModelRot

if __name__ == "__main__":
    pass
    


