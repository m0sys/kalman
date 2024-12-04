import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allantools as av


def find_log_slope(xdat, ydat, i0=None, i1=None):
    assert len(ydat) == len(xdat)
    if i0 is None:
        i0 = len(ydat) // 2
    if i1 is None:
        i1 = len(ydat) - 1
    y1 = ydat[i1]
    y0 = ydat[i0]
    x1 = xdat[i1]
    x0 = xdat[i0]
    y1log = np.log10(y1)
    y0log = np.log10(y0)
    x1log = np.log10(x1)
    x0log = np.log10(x0)
    return (y1log - y0log) / (x1log - x0log)


def analyse_acc_with_allan(csv_path):
    """Analyse accelerometer white noise using Allan Variance and plot results."""

    # Load data.
    stat_acc_df = pd.read_csv(csv_path)
    stat_acc_time_vec = stat_acc_df.loc[:, "Time (s)"].to_numpy()
    stat_acc_yaxis_vec = stat_acc_df.loc[:, "Acceleration y (m/s^2)"].to_numpy()

    # Find sampling properties.
    T = stat_acc_time_vec[1] - stat_acc_time_vec[0]
    Fs = 1 / T
    print(f"Sampling properties: {T}s, {Fs}Hz")

    # Integrate to find speed.
    vely_vec = np.cumsum(stat_acc_yaxis_vec) * T
    accel_taus, accel_adevs, _, _ = av.oadev(vely_vec, rate=Fs, taus="octave")

    # Find indices where slope is about -0.5.
    slope = -0.5
    accel_log_taus = np.log10(accel_taus)
    accel_log_adevs = np.log10(accel_adevs)
    accel_dlogadev = np.diff(accel_log_adevs) / np.diff(accel_log_taus)
    i = np.argmin(accel_dlogadev - slope)

    # Find y-intercept of line equation: y = mx + b => b = y - mx.
    b = accel_log_adevs[i] - slope * accel_log_taus[i]

    # Get log line points.
    N = np.pow(10, slope * np.log10(1) + b)
    accel_line_y = N / np.sqrt(accel_taus)
    accel_line_y.shape

    # Plot results.
    plt.figure()
    plt.title("Allan Deviation of Accelerometer's Y-Axis on LogLog Scale")
    plt.ylabel("$\\sigma$")
    plt.xlabel("$\\tau$ (s)")
    plt.loglog(accel_taus, accel_adevs, "g", label="$\\sigma$")
    plt.loglog(accel_taus, accel_line_y, "--b", label="-0.5 line")
    plt.loglog(1, N, "or")
    plt.grid(True)
    plt.legend()
    print(f"White-noise Slope: {float(find_log_slope(accel_taus, accel_adevs, 2, 10))}")


def analyse_gyr_with_allan(csv_path):
    """Analyse gyro white noise using Allan Variance and plot results."""

    # Load data.
    stat_gyr_df = pd.read_csv(csv_path)
    stat_gyr_time_vec = stat_gyr_df.loc[:, "Time (s)"].to_numpy()
    stat_gyr_zaxis_vec = stat_gyr_df.loc[:, "Gyroscope z (rad/s)"].to_numpy()

    # Find sampling frequency and period.
    T = stat_gyr_time_vec[1] - stat_gyr_time_vec[0]
    Fs = 1 / T
    print(f"Sampling properties: {T}s, {Fs}Hz")

    # Integrate to find angle.
    theta_z_vec = np.cumsum(stat_gyr_zaxis_vec) * T
    gyro_taus, gyro_adevs, _, _ = av.oadev(theta_z_vec, rate=Fs, taus="octave")

    # Find indices where slope is about -0.5.
    slope = -0.5
    gyro_log_taus = np.log10(gyro_taus)
    gyro_log_adevs = np.log10(gyro_adevs)
    gyro_dlogadev = np.diff(gyro_log_adevs) / np.diff(gyro_log_taus)
    i = np.argmin(gyro_dlogadev - slope)

    # Find y-intercept of line equation: y = mx + b => b = y - mx.
    b = gyro_log_adevs[i] - slope * gyro_log_taus[i]

    # Get log line points.
    N = np.pow(10, slope * np.log10(1) + b)
    gyro_line_y = N / np.sqrt(gyro_taus)

    # Plot results.
    plt.figure()
    plt.title("Allan Deviation of Gyro's Z-Axis on LogLog Scale")
    plt.ylabel("$\\sigma$")
    plt.xlabel("$\\tau$ (s)")
    plt.loglog(gyro_taus, gyro_adevs, "g", label="$\\sigma$")
    plt.loglog(gyro_taus, gyro_line_y, "--b", label="-0.5 line")
    plt.loglog(1, N, "or")
    plt.grid(True)
    plt.legend()
    print(f"White-noise Slope: {float(find_log_slope(gyro_taus, gyro_adevs, 4, 8))}")
