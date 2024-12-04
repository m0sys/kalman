import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial


def plot_lat_lon_gps_traj(gps_csv_path, truth_csv_path):
    traj_gps_df = pd.read_csv(gps_csv_path)
    traj_truth_df = pd.read_csv(truth_csv_path)
    traj_truth_lat_vec = traj_truth_df.loc[:, "Latitude (°)"].to_numpy()
    traj_truth_long_vec = traj_truth_df.loc[:, "Longitude (°)"].to_numpy()
    traj_gps_lat_vec = traj_gps_df.loc[:, "Latitude (°)"].to_numpy()
    traj_gps_long_vec = traj_gps_df.loc[:, "Longitude (°)"].to_numpy()

    plt.figure()
    plt.plot(traj_truth_long_vec, traj_truth_lat_vec, "o")
    plt.plot(traj_gps_long_vec, traj_gps_lat_vec, "--")
    plt.title("GPS and Truth Coords of Traj Long/Lat")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")


def process_gps_into_enu(gps_csv_path):

    # Load GPS data.
    traj_gps_df = pd.read_csv(gps_csv_path)
    traj_gps_lat_vec = traj_gps_df.loc[:, "Latitude (°)"].to_numpy()
    traj_gps_long_vec = traj_gps_df.loc[:, "Longitude (°)"].to_numpy()
    traj_gps_h_vec = traj_gps_df.loc[:, "Height (m)"].to_numpy()

    # Set reference plane to init GPS coords.
    lat_r = traj_gps_lat_vec[0]
    long_r = traj_gps_long_vec[0]
    h_r = traj_gps_h_vec[0]

    # Convert.
    ecef_enu_ref = partial(ecef_enu, lat_r, long_r, h_r)
    traj_gps_geodet_vec = np.stack(
        (traj_gps_lat_vec, traj_gps_long_vec, traj_gps_h_vec), axis=1
    )
    traj_gps_enu_vec = np.apply_along_axis(
        lambda row: ecef_enu_ref(row[0], row[1], row[2]), 1, traj_gps_geodet_vec
    )

    return traj_gps_enu_vec


def find_acc_gyr_sampling_properties(acc_csv_path, gyr_csv_path):

    # Load data.
    acc_df = pd.read_csv(acc_csv_path)
    gyr_df = pd.read_csv(gyr_csv_path)
    acc_time_vec = acc_df.loc[:, "Time (s)"].to_numpy()
    gyr_time_vec = gyr_df.loc[:, "Time (s)"].to_numpy()

    # Find accelerometer sampling frequency and period.
    T_acc = acc_time_vec[1] - acc_time_vec[0]
    Fs_acc = 1 / T_acc
    print(f"Sampling properties (ACCEL): {T_acc}s, {Fs_acc}Hz")

    # Find gyro sampling frequency and period.
    T_gyr = gyr_time_vec[1] - gyr_time_vec[0]
    Fs_gyr = 1 / T_gyr
    print(f"Sampling properties (GYRO): {T_gyr}s, {Fs_gyr}Hz")

    return T_acc, Fs_acc, T_gyr, Fs_gyr


def process_all_data_into_df_stream(
    traj_acc_csv_path, traj_gyr_csv_path, traj_gps_csv_path
):
    traj_acc_df = pd.read_csv(traj_acc_csv_path)
    traj_gyr_df = pd.read_csv(traj_gyr_csv_path)
    traj_gps_df = pd.read_csv(traj_gps_csv_path)
    traj_acc_yaxis_vec = traj_acc_df.loc[:, "Acceleration y (m/s^2)"].to_numpy()
    traj_acc_time_vec = traj_acc_df.loc[:, "Time (s)"].to_numpy()
    traj_gyr_zaxis_vec = traj_gyr_df.loc[:, "Gyroscope z (rad/s)"].to_numpy()
    traj_gyr_time_vec = traj_gyr_df.loc[:, "Time (s)"].to_numpy()
    traj_gps_vel_vec = traj_gps_df.loc[:, "Velocity (m/s)"].to_numpy()
    traj_gps_theta_vec = traj_gps_df.loc[:, "Direction (°)"].to_numpy()
    traj_gps_time_vec = traj_gps_df.loc[:, "Time (s)"].to_numpy()

    # Convert GPS heading to rad.
    traj_gps_theta_vec = np.deg2rad(traj_gps_theta_vec)

    # Make GPS heading consistent with GYRO reference frame.
    traj_gps_theta_vec = -traj_gps_theta_vec

    # Get some ENU GPS coords.
    traj_gps_enu_vec = process_gps_into_enu(traj_gps_csv_path)

    # How much of each?
    num_acc = traj_acc_time_vec.shape[0]
    num_gyr = traj_gyr_time_vec.shape[0]
    num_gps = traj_gps_time_vec.shape[0]
    print(f"ACC: {num_acc}, GYR: {num_gyr}, GPS: {num_gps} data points each")

    # Do some hardcore data wrangling that would make Wes proud.
    ACCEL_TYPE = "ACCEL"
    GYRO_TYPE = "GYRO"
    GPS_VEL_TYPE = "GPS_VEL"
    GPS_THETA_TYPE = "GPS_THETA"
    GPS_X_TYPE = "GPS_X"
    GPS_Y_TYPE = "GPS_Y"
    GPS_Z_TYPE = "GPS_Z"

    acc_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_acc_time_vec,
            "Measurement": traj_acc_yaxis_vec,
            "Type": np.repeat(ACCEL_TYPE, num_acc),
        }
    )

    gyr_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_gyr_time_vec,
            "Measurement": traj_gyr_zaxis_vec,
            "Type": np.repeat(GYRO_TYPE, num_gyr),
        }
    )

    gps_vel_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_gps_time_vec,
            "Measurement": traj_gps_vel_vec,
            "Type": np.repeat(GPS_VEL_TYPE, num_gps),
        }
    )

    gps_theta_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_gps_time_vec,
            "Measurement": traj_gps_theta_vec,
            "Type": np.repeat(GPS_THETA_TYPE, num_gps),
        }
    )

    gps_x_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_gps_time_vec,
            "Measurement": traj_gps_enu_vec[:, 0],
            "Type": np.repeat(GPS_X_TYPE, num_gps),
        }
    )

    gps_y_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_gps_time_vec,
            "Measurement": traj_gps_enu_vec[:, 1],
            "Type": np.repeat(GPS_Y_TYPE, num_gps),
        }
    )

    gps_z_seq_df = pd.DataFrame(
        {
            "Time (s)": traj_gps_time_vec,
            "Measurement": traj_gps_enu_vec[:, 2],
            "Type": np.repeat(GPS_Z_TYPE, num_gps),
        }
    )

    data_seq_df = pd.concat(
        (
            acc_seq_df,
            gyr_seq_df,
            gps_vel_seq_df,
            gps_theta_seq_df,
            gps_x_seq_df,
            gps_y_seq_df,
            gps_z_seq_df,
        ),
        ignore_index=True,
    )

    data_seq_df.sort_values(by=["Time (s)", "Type"], inplace=True)
    data_seq_df.reset_index(drop=True, inplace=True)

    return data_seq_df


def basic_ecef_enu(lat_r, long_r, lat_p, long_p):
    R = _ecef_enu_rotation_mat(lat_r, long_r)
    x_r, y_r, z_r = _basic_spherical_coords(lat_r, long_r)
    x_p, y_p, z_p = _basic_spherical_coords(lat_p, long_p)
    delta = np.array([x_p - x_r, y_p - y_r, z_p - z_r])
    delta.reshape((delta.shape[0], 1))
    enu = R @ delta
    return enu


def ecef_enu(lat_r, long_r, h_r, lat_p, long_p, h_p):
    R = _ecef_enu_rotation_mat(lat_r, long_r)
    x_r, y_r, z_r = _geodet_to_ecef(lat_r, long_r, h_r)
    x_p, y_p, z_p = _geodet_to_ecef(lat_p, long_p, h_p)
    delta = np.array([x_p - x_r, y_p - y_r, z_p - z_r])
    delta.reshape((delta.shape[0], 1))
    enu = R @ delta
    return enu


def _ecef_enu_rotation_mat(lat_r, long_r):
    phi_r = np.deg2rad(lat_r)
    lmbd_r = np.deg2rad(long_r)
    R = np.transpose(
        np.array(
            [
                [
                    -np.sin(lmbd_r),
                    -np.sin(phi_r) * np.cos(lmbd_r),
                    np.cos(phi_r) * np.cos(lmbd_r),
                ],
                [
                    np.cos(lmbd_r),
                    -np.sin(phi_r) * np.sin(lmbd_r),
                    np.cos(phi_r) * np.sin(lmbd_r),
                ],
                [0, np.cos(phi_r), np.sin(phi_r)],
            ]
        )
    )
    return R


def _basic_spherical_coords(lat, long):
    R = 6371000
    phi = np.deg2rad(lat)
    lmbd = np.deg2rad(long)
    x = R * np.cos(phi) * np.cos(lmbd)
    y = R * np.cos(phi) * np.sin(lmbd)
    z = R * np.sin(phi)
    return np.array([x, y, z])


def _prime_vert_radius_curvature(lat, a=6378137.0, b=6356752.314245):
    phi = np.deg2rad(lat)
    return a**2 / np.sqrt(a**2 * np.cos(phi) ** 2 + b**2 * np.sin(phi) ** 2)


def _geodet_to_ecef(lat, long, h):
    a = 6378137.0
    b = 6356752.314245
    phi = np.deg2rad(lat)
    lmbd = np.deg2rad(long)
    N = _prime_vert_radius_curvature(lat)
    x = (N + h) * np.cos(phi) * np.cos(lmbd)
    y = (N + h) * np.cos(phi) * np.sin(lmbd)
    z = (b**2 / a**2 * N + h) * np.sin(phi)
    return x, y, z
