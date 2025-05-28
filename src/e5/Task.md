# Exercise 5

In this exercise, we are simulating xray images.


## Tasks
Implement `create_xray` in the python file. The method should return the maximum, minimum and sum of HU values along the x-ray view axis (each slice in the dicom image sequence).

Next, implement `get_fancy_colormap` in `common/fancy_colormap.py`. This method should return a colormapping from HU-units to a color. Try and find a mapping that highlights interesting tissues or objects in the image.

## Tips
- Have a look at the lecture slides for the HU ranges.
- To see the "solution" colormap, just return the `_get_fancy_colormap_solution` instead.

