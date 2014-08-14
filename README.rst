Python Projective Camera Model
==============================

Features
--------
- camera intrinsic and extrinsic parameters handling
- various lens distortion models
- model persistence
- projection of camera coordinates to an image
- conversion of image coordinates on a plane to camera coordinates
- visibility handling

Installation
------------
:: 

    pip install git+https://github.com/smidm/camera.py.git
    
Usage
-----

Setup camera and project point [0, 0, 0]::

    import camera
    import numpy as np
    import math
    c = camera.Camera()
    c.set_K_elements(1225.0, math.pi / 2, 1, 480, 384)
    R = np.array(
        [[-0.9316877145365, -0.3608289515885, 0.002545329627547],
         [-0.1725273110187, 0.4247524018287, -0.8888909933995],
         [0.3296724908378, -0.8263880720441, -0.4579894432589]])
    c.set_R(R)
    c.set_t(np.array([[-1.365061486465], [3.431608806127], [17.74182159488]]))
    
    print c.world_to_image(np.array([[0., 0., 0.]]).T)
    
Load camera parameters from a file::

    import camera
    import numpy as np
    c = camera.Camera()
    c.load('camera_01.yaml')
    c.world_to_image(np.array([[0., 0., 0.]]).T)

Documentation
-------------

http://python-projective-camera-model.readthedocs.org

Development
-----------

To test the package run::

    $ ./run_tests.sh
    
The module is missing some (essential) functionality (search for TODO comments). I appreciate pull requests that fill the gaps. 

License
-------

MIT
