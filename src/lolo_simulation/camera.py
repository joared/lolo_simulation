from scipy.spatial.transform import Rotation as R
import numpy as np

class SimCamera:
    def __init__(self, translationVector, rotationVector):
        self.translationVector = translationVector
        self.rotationVector = rotationVector

        self.initTranslationVector = self.translationVector.copy()
        self.initRotationVector = self.rotationVector.copy()

if __name__ == "__main__":
    pass
    


