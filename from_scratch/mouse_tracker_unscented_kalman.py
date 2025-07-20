# pylint: disable=C0103
# ////////////////////////////////////////////////////////////////////////
# mouse_tracker_UKF.py by Joseph Betthauser 2015
#     Description:  Unscented Kalman Filter (pos, vel, accel)
# /////////////////////////////////////////////////////////////////////////
import time

import keyboard
import matplotlib.pyplot as plt
import mouse
import numpy as np
from numba import njit


def ukf_init(dt_: float) -> tuple[np.ndarray]:
    """
    _summary_

    Args:
        dt_ (float): _description_

    Returns:
        tuple[np.ndarray]: init of A, B, H, Q, R, c, x, P

    """
    a_ = np.array(
        [
            [1, 0, dt_, 0, 0.5 * dt_**2, 0],
            [0, 1, 0, dt_, 0, 0.5 * dt_**2],
            [0, 0, 1, 0, dt_, 0],
            [0, 0, 0, 1, 0, dt_],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    b_ = np.eye(6)
    h_ = np.eye(6)
    q_ = 0.001 * np.eye(6)
    r_ = 0.1 * np.eye(6)

    x_ = np.zeros((6, 1))  # initial state
    c_ = np.zeros((6, 1))  # goal
    p_ = np.zeros((6, 6))

    return a_, b_, h_, q_, r_, c_, x_, p_


@njit
def ukf_correction(
    h_: np.ndarray,
    r_: np.ndarray,
    x_: np.ndarray,
    z_: np.ndarray,
    p_: np.ndarray,
) -> tuple[np.ndarray]:
    """
    Perform the correction step of the Unscented Kalman Filter.

    Args:
        h_ (np.ndarray): Observation matrix.
        r_ (np.ndarray): Measurement noise covariance matrix.
        x_ (np.ndarray): Predicted state estimate.
        z_ (np.ndarray): Measurement vector.
        p_ (np.ndarray): Predicted estimate covariance.

    Returns:
        tuple[np.ndarray]: Updated state estimate and covariance.

    """
    s_ = h_ @ p_ @ h_.T + r_
    k_ = p_ @ h_.T @ np.linalg.pinv(s_)
    y_ = z_ - h_ @ x_
    x_ = x_ + k_ @ y_
    p_ = (np.eye(6) - k_ @ h_) @ p_
    return x_, p_


@njit
def ukf_predict(
    a_: np.ndarray,
    b_: np.ndarray,
    q_: np.ndarray,
    c_: np.ndarray,
    x_: np.ndarray,
    p_: np.ndarray,
) -> tuple[np.ndarray]:
    """
    Unscented Kalman Filter prediction step

    Args:
        a_ (np.ndarray): State transition matrix
        b_ (np.ndarray): Control input matrix
        q_ (np.ndarray): Process noise covariance matrix
        c_ (np.ndarray): Control input vector
        x_ (np.ndarray): State estimate
        p_ (np.ndarray): Error covariance matrix

    Returns:
        tuple[np.ndarray]: Updated state estimate and covariance (x_pred, p_pred)

    """
    # Predict state
    x_pred = a_ @ x_ + b_ @ c_

    # Predict covariance
    p_pred = a_ @ p_ @ a_.T + q_

    return x_pred, p_pred

    # plt.ion()  # interactive plots on


def main() -> None:  # noqa: PLR0915
    # init vars
    plt.ion()
    measure_noise = 0.25
    screen_width = 2400  # root.winfo_screenwidth()
    screen_height = 1600  # root.winfo_screenheight()
    dim_fix = screen_height
    dt = 0.15
    num_poses = 20  # vars track last N values for green trace
    ramp = np.linspace(0, 1, num_poses)  # noqa: F841
    last_poses = np.nan * np.zeros((2, num_poses))

    rng = np.random.default_rng()  # Use new random generator

    ## init kalman filter params
    a, b, h, q, r, c, x, p = ukf_init(dt)

    ## Make program respond to mouse movements
    print('Press "ESC" key to quit')
    dist2target = 20

    fig, ax = plt.subplots(figsize=(15, 9))
    # fig.canvas.manager.full_screen_toggle()
    measurement_scatter = ax.scatter(0, 0, c="r", marker="o", s=20, label="measurement")
    trace_plot = ax.plot(
        last_poses[0, :],
        last_poses[1, :],
        c="g",
        lw=1,
        label="unscented kf tracker",
    )[0]
    cursor_scatter = ax.scatter(0, 0, c="k", marker="+", s=50, label="true cursor loc.")
    ax.axis([0, screen_width, 0, screen_height])
    # plt.xlabel("x-pos.")
    # plt.ylabel("y-pos.")
    # plt.title("Unscented Kalman Filter Mouse Tracker")
    # plt.legend()

    while True:
        t0 = time.perf_counter()

        ## Get real current x,y position of cursor
        cursor = np.array(mouse.get_position())
        cursor[1] = dim_fix - cursor[1]

        xtrue, ytrue = cursor

        lastx = x  # save last state values before updates

        ## Kalman Position/Vel/Accel Updates
        ## Get noisy measurement
        nx = [xtrue + measure_noise * dist2target * rng.normal()]
        ny = [ytrue + measure_noise * dist2target * rng.normal()]
        vx = (nx - lastx[0]) / dt  # est. x-velocity from measurement
        vy = (ny - lastx[1]) / dt  # est. y-velocity from measurement
        ax = (vx - lastx[2]) / dt  # est. x-accel from measurement
        ay = (vy - lastx[3]) / dt  # est. y-accel from measurement
        z_list = [nx, ny, vx, vy, ax, ay]
        z_vec = np.reshape(np.array(z_list), (6, 1))
        z_list = [nx, ny, vx, vy, ax, ay]
        z_vec = np.reshape(np.array(z_list), (6, 1))

        ## Measurement Updates
        x, p = ukf_correction(h, r, x, z_vec, p)
        dist2target = np.sqrt((x[0, :] - lastx[0, :]) ** 2 + (x[1, :] - lastx[1, :]) ** 2)
        # dist2target = 1.0

        last_poses = np.roll(last_poses, -1, axis=1)
        last_poses[:, -1] = np.squeeze(x[0:2, :])

        ## Plot tracking outputs
        measurement_scatter.set_offsets(np.c_[nx, ny])

        trace_plot.set_xdata(last_poses[0, :])
        trace_plot.set_ydata(last_poses[1, :])

        cursor_scatter.set_offsets(cursor)

        fig.canvas.draw()
        fig.canvas.flush_events()

        if keyboard.is_pressed("escape") or keyboard.is_pressed("space"):
            plt.close()
            break

        tf = time.perf_counter() - t0

        if tf > dt:
            time.sleep(0.001)
        else:
            time.sleep(dt - tf)


if __name__ == "__main__":
    main()
