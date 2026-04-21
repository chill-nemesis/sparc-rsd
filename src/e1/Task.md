# Exercise 1

In this exercise, you will implement a kinematic chain for basic robotic joints.

A joint can be thought of as a rigid connection between two links, the parent link and the child link.
The links are the connection points between two joints, and the parent link of the "next" joint always connects with child link of the "previous" joint.
In other words, the parent link is the one closer to the base, and the child link is closer to the end effector.

Each link has its own, dedicated local coordinate system (here it is called a coordinate frame).
Most of the work is done in the joints parent local coordinate frame.
Here we define the axis of our rotation, and where the child coordinate frame is supposed to be.
The reason we do it like this is so we do not have to think about where exactly the global position of the joint is.
Instead, we can simply say that the child frame is supposed to be in that direction, and we would like to rotate around our local Z axis, thank you very much.

Now we only need to link up the previous joint with the next joint.
Most commonly, the [Denavit-Hartenber notation](https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters) is used.
However, that is way out of scope for this exercise, so we use a simplified version instead.
We just assume that the child coordinate frame of the previous joint is the next joints parent frame.

## Coding task

Check `e1/joint.py`. It contains three classes:

- `Joint3D`: **Base class** defining the required methods.
- `RevoluteJoint3D` and `PrismaticJoint3D`: **Implementations** for rotational and translational joints.

Each joint has:

- `actuation_axis`: Unit vector for the rotation/translation axis. This is defined in the joint's local parent frame.
- `link_offset_mm`: The fixed local translation from the current joint parent frame to the child frame.
- `parent`: The parent joint or `None` if it is the base. Careful, this parent here does not refer to a link within a joint, but rather means "To which joint is this joint connected".

## Tasks

Implement the following methods.
First, start with `math.py` to define a few basic math helpers that we use throughout the code.
The `Transformation` dataclass is already provided and wraps a homogeneous 4x4 matrix.
It exposes the full matrix as `matrix`, the 3x3 rotation part as `rotation`, and the 3D translation as `position`.

- **`rotation_matrix_from_axis_angle`**:
  This method takes a rotation axis (a 3D vector) and the angle in radians that describes the rotation around the given axis.
  From these two values, a 3x3 rotation matrix is created. See Rodrigues' rotation formula from the exercise.

- **`homogeneous_transform`**:
  Takes an optional 3x3 rotation matrix and an optional translation vector.
  If the rotation matrix is not given, assume the identity matrix, i.e. no rotation.
  If the translation vector is not given, assume the zero vector, i.e. no translation.
  The method should then construct the corresponding homogeneous 4x4 matrix expressing the rotation and translation
  and return it wrapped as a `Transformation`.
  Note: the dot product can be used to express matrix multiplications.

Now we are ready to implement the joints:

- **`get_local_motion`**:  
  Returns the `Transformation` that represents only the **joint motion**.
  This is effectively the homogeneous transform that describes the rotation or translation of the joint for a given rotation angle or actuation distance.
  Input: rotation angle (radians) or translation distance (mm).

- **`get_local_child_from_parent`**:  
  Returns the **local** `Transformation` from the current joint parent frame to the child frame.
  This should combine the configurable motion with the fixed local child offset.

- **`get_global_child_from_base`**:  
  Returns the **global** `Transformation` from the robot base frame (e.g., where the robot is mounted on the table) to the child frame after this joint.
  In our case the base of the robot is always assumed to be at the origin (identity matrix).
  Input: List of joint configurations.
  Hint: You can use `get_global_parent_from_base`, which gives you the transformation from the robot base to the current joint parent frame.
  The end-effector position can then be read from `.position` on the returned `Transformation`.

Joint configurations (angles or translations) are provided as a list:

```python
joint_configuration[0]    # First joint
joint_configuration[1:-1] # Intermediate joints
joint_configuration[-1]   # Current joint
```

## Tips

- `get_local_child_from_parent` and `get_global_child_from_base` are joint-type independent.
- Keep track of the difference between the **joint parent frame** and the **child frame**.
- All methods must be implemented for the animation to run.
- Feel free to modify the joint chain in the `_main()` method `kinematics.py` to test and explore your solutions.
- If you get stuck or want some inspiration, check out the example implementation in the solution folder.
