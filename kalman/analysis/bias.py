import pandas as pd
from regression.reg import lin_reg_model_approx, plot_regression_with_data


def find_acc_bias_model(csv_path):
    stat_acc_df = pd.read_csv(csv_path)
    accel_time_vec = stat_acc_df.loc[:, "Time (s)"].to_numpy()
    accel_yaxis_vec = stat_acc_df.loc[:, "Acceleration y (m/s^2)"].to_numpy()
    accel_b, accel_m = lin_reg_model_approx(accel_yaxis_vec, accel_time_vec)
    accel_fx = accel_m * accel_time_vec + accel_b
    plot_regression_with_data(
        accel_yaxis_vec,
        accel_time_vec,
        accel_fx,
        title=f"Accerometer Y-Axis Linear Regression Model: b={round(accel_b,5 )}, m={round(accel_m, 10)}",
        xlabel="Time (s)",
        ylabel="Acceleration Y-Axis (m/s^2)",
    )
    print(f"Initial bias (b): {accel_b}")
    print(f"Time-varying bias (b): {accel_m}")
    return accel_fx


def find_gyr_bias_model(csv_path):
    stat_gyr_df = pd.read_csv(csv_path)
    gyro_time_vec = stat_gyr_df.loc[:, "Time (s)"].to_numpy()
    gyro_zaxis_vec = stat_gyr_df.loc[:, "Gyroscope z (rad/s)"].to_numpy()
    gyro_b, gyro_m = lin_reg_model_approx(gyro_zaxis_vec, gyro_time_vec)
    gyro_fx = gyro_m * gyro_time_vec + gyro_b
    plot_regression_with_data(
        gyro_zaxis_vec,
        gyro_time_vec,
        gyro_fx,
        title=f"Gyro Z-Axis Linear Regression Model: b={round(gyro_b,10 )}, m={round(gyro_m, 12)}",
        xlabel="Time (s)",
        ylabel="Gyro Z-Axis (rad/s)",
    )
    print(f"Initial bias (b): {gyro_b}")
    print(f"Time-varying bias (b): {gyro_m}")
    return gyro_fx
