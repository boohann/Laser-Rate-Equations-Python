# Laser Rate Equations solved in Python

A Python implementation of laser rate equations for modeling lumped single-mode behavior in laser cavities.

## Overview

This project solves laser rate equations numerically using Python 3. 

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
- Values in SimConfig can also be changed. The preset values of 2.5 ns for scan time length, and 0.1 ps
  for scan time step are typical values for a semiconductor laser to allow the calculation to converge. 

## Documentation

For a basic description on the laser rate equation approach, refer to `Solving_Rate_Equations.pdf`.

## Future Updates
- Migrate `Solving_Rate_Equations.pdf` documentation to CodeWiki.
