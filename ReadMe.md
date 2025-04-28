# RSD Exercises

This repository contains the code for the exercises in *Robotics in Surgery and Diagnostics* from the [SPARC Lab at FAU](https://www.sparc.tf.fau.de).  
It is organized into dedicated exercise modules for students (e1, e2, ...) to support the corresponding lecture content.

Each module also includes a `solution` folder with a fully implemented solution.  
If you get stuck or would like to see the intended approach, feel free to compare your work with the provided solutions.

## Important Notice
The exercises are currently under active development.  
New exercises will be added each week — please pull the latest changes weekly and re-install the package (see Step 5 below) to receive all updates.  
I will avoid modifying existing code, so if you use `git pull --ff-only`, your solutions for previous exercises should remain unaffected.

## Getting Started
1. Ensure you have a (relatively) recent version of Python installed. The exercises have been developed and tested using Python 3.11.9.
2. Download the exercise code.
3. [Create a virtual environment](https://docs.python.org/3/library/venv.html).
4. Activate the virtual environment (both in your terminal and in Visual Studio Code).  
   All commands from this point onward assume that the virtual environment is active!
5. Install the exercise code and dependencies by running:
   ```bash
   pip install -e .[development]
   ```
   This will install all required packages for working on the exercises.
6. You are now ready to start solving the exercises!
