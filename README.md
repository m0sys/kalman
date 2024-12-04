# kalman
Implementation of Kalman filtering for IMU and GPS sensor.

> Probably the most straight-forward and open implementation of KF/EKF filters used for sensor fusion of GPS/IMU data.

# Overview

The goal of this project was to integrate IMU data with GPS data to estimate the
pose of a vehicle following a trajectory. It uses a nonlinear INS equation model
for the vehicle (process model) and linear measurement models for GPS and IMU data.

Given the nonlinear nature of the INS model a EKF was chosen for the high-level
XY pose estimates and a KF for the low-level heading and velocity estimates.

The two filters interact with each other in a cascading fashion where the low-level
heading and velocity estimates of KF filter is given as input to the prediction 
stage of the EKF. In both KF/EKF correction phase the input is GPS data. In the
case of KF it uses the GPS heading and velocity measurements to fuse with the 
forward euler integration found in the prediction stage of the KF. In the EKF
it uses the GPS ENU transformed XY coords to correct the nonlinear predictions
made by EKF in prediction stage.

For more  details please see [final design notebook](nbs/07_final_analysis_ckf_design.ipynb).
