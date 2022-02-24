import numpy as np
from camera import Camera, FeatureSet
from animation import CameraAnimator

def triangle(controlRule):
    camera = Camera(translation=(-5, -0.9, -1.5), 
                    euler=(-.5, -0.1, 0.1), 
                    controlRule=controlRule)
    featureSet = FeatureSet([[-1, 0, 0], [0, 0, 1], [1, 0, 0]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/2, 0))
    targets = [[0.3, 0], [0, 0.3], [-0.3, 0]]
    return camera, featureSet, targets

def stressTest(controlRule):
    camera = Camera(translation=(-6, -1, -0.4), 
                    euler=(-.55, 0.2, 0.2), 
    #camera = Camera(translation=(-5, -1.2, -1.5),
    #                euler=(-.5, -0.1, 0.1), 
                    controlRule=controlRule)
    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/2, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    return camera, featureSet, targets

def rotationTest(controlRule):
    camera = Camera(translation=(0, -1/0.3, 0), 
                    euler=(0, 0, np.pi/2),
                    controlRule=controlRule)
    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/4, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]
    return camera, featureSet, targets

if __name__ == "__main__":
    # features: feature positions
    # targets: desired position of feature in image plane (y [left], z [up])
    # instable for LeStar
    #camera = Camera(translation=(-5, -1, -2), euler=(-0.3, -0.2, 0.1))
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--controlrule', '-c', type=str, default="Le",
                        help='control rule for camera motion (Le, LeStar or LeLeStar)')
    parser.add_argument('--scenario', '-s', default="stress",
                        help='choose camera and feature scenario')

    args = parser.parse_args()
    print(args)

    controlRule = args.controlrule
    if args.scenario == "rot":
        camera, featureSet, targets = rotationTest(controlRule)
    elif args.scenario == "triangle":
        camera, featureSet, targets = triangle(controlRule)
    elif args.scenario == "stress":
        camera, featureSet, targets = stressTest(controlRule)
    else:
        raise Exception("Invalid scenario '{}'".format(args.scenario))
    
    cameraAnimator = CameraAnimator(camera, featureSet.transformedFeatures(), targets, None)
    cameraAnimator.animate()
    #cameraAnimator.anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    cameraAnimator.show()