# Exercise 1

In this exercise, you will implement a kinematic chain for basic robotic joints.

Check `e1/joints.py`. It contains three classes:
- `Joint3D`: **Base class** defining the required methods.
- `RevoluteJoint3D` and `PrismaticJoint3D`: **Implementations** for rotational and translational joints.

Each joint has:
- `axis_of_rotation`: Unit vector for the rotation/translation axis.
- `length_mm`: Link length (radius for revolute, arm length for prismatic).
- `parent`: The parent joint or `None` if it is the base.

## Tasks
Implement the following methods:

- **`get_transformation_matrix`**:  
  Returns the **local** homogeneous transformation.  
  Input: rotation angle (radians) or translation distance (mm).

- **`get_cumulative_transformation`**:  
  Returns the **global** transformation from the base to this joint.  
  Base joint defaults to identity matrix.
  Input: List of joint configurations.

- **`get_global_position`**:  
  Returns the **global** 3D position of the joint end-effector.
  Input: List of joint configurations.

Joint configurations (angles or translations) are provided as a list:
```python
joint_configuration[0]    # First joint
joint_configuration[1:-1] # Intermediate joints
joint_configuration[-1]   # Current joint
```

## Tips
- `get_cumulative_transformation` and `get_global_position` are joint-type independent.
- Both methods must be implemented for the animation to run
- Feel free to modify the joint chain in the `_main()` method `kinematics.py` to test and explore your solutions.