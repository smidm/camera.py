import camera
import numpy as np
from nose.tools import *
import math
import matplotlib.pylab as plt


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
    c.load('test/camera_01.yaml')
    # pitch dimensions [-20, -10, 19, 9.5]  # xmin, ymin, xmax, ymax
    points = np.array([[-20, -10, 0], [-20, 9.5, 0], [19, 9.5, 0], [19, -10, 0], [-20, -10, 0]]).T
    c.plot_world_points(points, 'r-', solve_visibility=False)
    points = np.array([[0, -10, 0], [0, 9.5, 0]]).T
    c.plot_world_points(points, 'y-', solve_visibility=False)
    import matplotlib.pylab as plt
    plt.imshow(plt.imread('test/cam01.png'))
    # plt.show()
    plt.savefig('camera_load_test.png', dpi=150)
    plt.close()



def init_test():
    c = camera.Camera(1)
    c.set_K_elements(480, 384, 1225.0)
    R = np.array(
        [[-0.9316877145365, -0.3608289515885, 0.002545329627547],
         [-0.1725273110187, 0.4247524018287, -0.8888909933995],
         [0.3296724908378, -0.8263880720441, -0.4579894432589]])
    c.set_R(R)
    c.set_t(np.array([[-1.365061486465], [3.431608806127], [17.74182159488]]))
    eq_(c.get_focal_length(), 1225.)
    assert np.array_equal(c.get_principal_point_px(), np.array([[480., 384.]]))
    c.world_to_image(np.array([[0., 0., 0.]]).T)


def pilab_annotate_load(filename):
    """

    :param filename:
    :return: :rtype: np.ndarray, shape=(n, 2)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    root = tree.getroot()
    query_results = root.findall("./objects[@type='points']")
    assert len(query_results) == 1
    objects = query_results[0]

    npoints = int(objects.get('count'))
    points = np.zeros((npoints, 2))
    xy_string = objects.text.strip().split('\n')
    for i in xrange(npoints):
        xy = xy_string[i].split(' ')
        points[i, 0] = float(xy[0])
        points[i, 1] = float(xy[1])

    return points


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def calibrate_division_model_test():
    img = rgb2gray(plt.imread('test/kamera2.png'))
    y0 = np.array(img.shape)[::-1][np.newaxis].T / 2.
    z_n = np.linalg.norm(np.array(img.shape) / 2.)
    points = pilab_annotate_load('test/kamera2_lines.xml')
    points_per_line = 5
    num_lines = points.shape[0] / points_per_line
    lines_coords = np.array([points[i * points_per_line:i * points_per_line + points_per_line] for i in xrange(num_lines)])
    c = camera.calibrate_division_model(lines_coords, y0, z_n)

    import matplotlib.cm as cm
    plt.figure()
    plt.imshow(img, cmap=cm.gray)
    for line in xrange(num_lines):
        x = lines_coords[line, :, 0]
        plt.plot(x, lines_coords[line, :, 1], 'g')
        mc = camera.fit_line(lines_coords[line].T)
        plt.plot(x, mc[0] * x + mc[1], 'y')
        xy = c.undistort(lines_coords[line].T)
        plt.plot(xy[0, :], xy[1, :], 'r')
    plt.show()
    plt.close()


