# Exercise 1

In this exercise, we are extending the forward kinematics from exercise 01 and implement the corresponding jacobian inverse kinematics.

In this exercise, your joint implementation of exercise 01 is reused and updated with some helper methods - have a look at `e2/solution/joint.py`. All your methods from e1 will also be available to you, but the added methods should simplify the implementation of the IK solver.


## Tasks
We are going to implement a jacobian IK solver as discussed in the exercise video in `e2/ik_solver.py`.
This file contains multiple classes, but most of them are just boiler-plate code to make the next exercise easier. The provided classes can be split into 3 groups:
- `Constraints`, which implement IK constraints for the solver. Currently, we are only focussing on a positional constraint, for which the skeleton is already prepared in `PositionConstraint`.
- `DeltaThetaUpdate`, which implements different ways of calculating the delta-theta update based on the calculated jacobian and end-effector error. Here, `PlainJacobianUpdate` should implement the method for calculating the delta-theta discussed in the exercise.
- `JacobianIKSolver` implements the actual IK solver for this exercise. Based on the exercise, please implement the `solve` method.

There is also an optional taks to the exercise. You might have seen that there exists a `DLSJacobianUpdate` method. This is supposed to implement a "damped least square" update (or Levenberg-Marquardt update), rather than the plain jacobian update. This requires a more in-depth maths dive to understand and fully implement, but it provides a lot more stable solutions. If you are interested, you can try to implement it. A (somewhat) good starting point is the corresponding [wikipedia article](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) - try to understand the following equation:
![DLS error equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/de95bef27493cc5fddb18a6667d3bfbb1d37f02d). The DLS method is not required to know or understand for the exam.

## Tips
- Once you implemented the IK solver, feel free to play around with the values for alpha, and/or the target position. What happens if you use a large alpha (e.g., .8)? What happens, if you choose (0, 0, 0) as the target position for the end effector? Or something far away, e.g., (1000, 0, 0)?
