# Exercise 7

For exercise 7, we implement the bowyer-watson algorithm in 3D to triangulate a randomized mesh.

## Keybindings
- Space triggers the next iteration
- Enter runs the remaining iterations
- U force-redraws
- P toggles the visibility of the mesh-points
- H prints a help-text in the CLI. This also includes the bindings/modifications that O3D may be using


## Tasks
- Implement `is_point_in_sphere` for the `Sphere` class
- Implement `_calculate_circumsphere` in the `Tetrahedron` class. The circumsphere is the sphere where all four points of the tetrahedron are on the sphere's surface. Have a look at the exercise slides for a visual example for Triangles and Circles.
- Implement `_iteration` in the `BowyerWatsonVisualiser` class. The method wraps one iteration of the bowyer-watson triangulation iteration. I recommend to read the hints in the method!

## Tips:
- Don't use too many points for the algorithm (espc. during debugging/visualization). It may get hard to see what is going on, and (at least the solution) code is not designed for high-performance.
- If you want, you can try to define your own mesh (e.g., a cube). Replace the random_points in `triangulation.py` with any point-list that you are interested in. The point-list must be a single numpy-array of n 3D points.
