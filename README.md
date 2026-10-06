# Laser Rate Equations solved in Python


A Python implementation of simplified laser rate equations for modelling lumped single-mode behavior in a laser cavity.


## Overview

This project numerically solves laser rate equations using Python 3. The project aims to keep the laser rate equations as simple as possible, so they can be easily understood, but 
also serve as a platform for any additional modifications to the model you wish to make.

## Version History

### V2 (2026-07-05)
- Code refactored for improved readability.
- Updated solver for better performance.
### V1
- Initial full release of code.


## Getting Started

### Requirements
- Python 3.x
- matplotlib 3.10.9
- numpy 2.4.4
- scipy 1.17.1
### Usage
- Change values in LASER_PARAMS to match those of the device you wish to model.
- Values in SimConfig can also be changed. The preset values of 2.5 ns for scan time length and 0.1 ps
  for scan time step are typical values for a semiconductor laser that allow the calculation to converge. 


## Documentation

For a basic description of the laser rate equation approach, refer to `Solving_Rate_Equations.pdf`.


## Future Updates
- Migrate `Solving_Rate_Equations.pdf` documentation to CodeWiki.
