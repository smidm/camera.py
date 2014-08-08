import camera
import numpy as np
from nose.tools import *


def p2e_test():
    proj2d = np.array([[1., 2., 3.]]).T
    eucl2d = camera.p2e(proj2d)
    eq_(eucl2d.size, 2)
    eq_(eucl2d[0], 1./3)

    eucl2d_2x = camera.p2e(np.hstack((proj2d, proj2d)))
    eq_(eucl2d_2x.shape, (2, 2))

    proj3d = np.array([[1., 2., 3., 4.]]).T
    eucl3d = camera.p2e(proj3d)
    eq_(eucl3d.size, 3)
    eq_(eucl3d[0], 1./4)

    eucl3d_2x = camera.p2e(np.hstack((proj3d, proj3d)))
    eq_(eucl3d_2x.shape, (3, 2))


def p3e_test():
    euclid_2d = np.array([[1., 2.]]).T
    proj_2d = camera.e2p(euclid_2d)
    eq_(proj_2d.shape, (3, 1))
    ok_(np.all(proj_2d == np.vstack((euclid_2d, 1))))

    euclid_3d = np.array([[1., 2., 3.]]).T
    proj_3d = camera.e2p(euclid_3d)
    eq_(proj_3d.shape, (4, 1))
    ok_(np.all(proj_3d == np.vstack((euclid_3d, 1))))


def load_test():
    c = camera.Camera(1)
    c.load('camera_01.yaml')
    # pitch dimensions [-20, -10, 19, 9.5]  # xmin, ymin, xmax, ymax
    points = np.array([[-20, -10, 0], [-20, 9.5, 0], [19, 9.5, 0], [19, -10, 0], [-20, -10, 0]]).T
    c.plot_world_points(points, 'r-', solve_visibility=False)
    points = np.array([[0, -10, 0], [0, 9.5, 0]]).T
    c.plot_world_points(points, 'y-', solve_visibility=False)
    import matplotlib.pylab as plt
    plt.imshow(plt.imread('cam01.png'))
    # plt.show()
    plt.imsave()
    plt.savefig('camera_load_test.png', dpi=150)








