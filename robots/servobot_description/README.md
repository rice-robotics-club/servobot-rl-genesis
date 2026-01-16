# servobot_description

This repository contains a ROS2 package for the URDF files of the Servobot robot.
The urdf file is located at `urdf/servobot.urdf`, with the CAD files located in `meshes/`.

Additionally, this package provides the github action workflow `.github/workflows/update-urdf.yaml`,
which updates the contents of the package from the OnShape assembly specified in `config.json`. This
is done with the [onshape-to-robot](https://onshape-to-robot.readthedocs.io/en/latest/index.html) tool.

A joint name array constant is included in Python, which can be imported via:
```Python
from servobot_description import JOINT_NAMES
```

Lastly, the URDF model can be previewed in rviz2 by running this launch file:
```bash
ros2 launch servobot_description visualize.launch.py
```
