import camera
import numpy as np
from nose.tools import *
import cv2
import os


def opencv_test():
    np.set_printoptions(precision=1, suppress=True, linewidth=150)

    cam = camera.Camera()
    cam.load('test/camera_opencv.yaml')
    
    if not os.path.exists('test/out'):
        os.mkdir('test/out')

    img = cv2.imread('test/opencv_distorted.jpg')
    assert isinstance(img, np.ndarray)
    img_und = cam.undistort_image(img)
    cv2.imwrite('test/out/test_undistorted.png', img_und)

    img_points = np.array(
        [[1036.,   198.],
         [1557.,   203.],
         [644.,   623.],
         [1938.,   649.],
         [880.,   630.],
         [1667.,   640.],
         [864.,  1238.],
         [1648.,  1272.],
         [630.,  1227.],
         [1915.,  1278.],
         [996.,  1668.],
         [1511.,  1694.]]).T

    print 'Distorted (manually marked)'
    print img_points

    img_points_und = cam.undistort(img_points)
    print 'Undistorted'
    print img_points_und

    for point in img_points.T.astype(int):
        cv2.circle(img, tuple(point), 3, (0, 255, 0), -1)
    cv2.imwrite('test/out/test_points.png', img)

    for point in img_points_und.T.astype(int):
        cv2.circle(img_und, tuple(point), 3, (0, 255, 0), -1)
    cv2.imwrite('test/out/test_points_und.png', img_und)

    img_points_dist = cam.distort(img_points_und)
    print 'Distorted back:'
    print img_points_dist

    print 'Undistortion / distortion errors:'
    print np.linalg.norm(img_points - img_points_dist, axis=0)

    image_coords = cam.world_to_image(np.array([[0., 0., 0.]]).T)
    print image_coords

    print cam.image_to_world(image_coords, 0.)







