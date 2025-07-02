# Exercise 9

In this exercise we are implementing the iterative closest point algorithm to align two pointlcouds.

## Keybindings
- Space triggers the next iteration


## Tasks
- Implement `_find_nearest_neighbors` in `icp.py`. The method returns two list of points containing a matching set for the source-target points (i.e., the source_matched[i] is the point from the source point cloud that is closest to the point in target_matched[i]; a point from the target point cloud)
- Next, implement the calculation of the delta-transform update (`_compute_delta_transformation`). This should return the delta transformation for the current ICP step, based on the nearest_neighbour list.
- Finally, calculate the error between the (now updated) source pointcloud and the target point cloud. Remember, we want to minimize the distance between all points.

## Tips:
- Play around with the "true" random option in `registration.py`. How does that compare to the given transformation of the target? Does ICP work all of the time? If not, when does ICP fail?
