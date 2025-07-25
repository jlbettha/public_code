# pylint: disable=C0103
## ////////////////////////////////////////////////////////////////////////
#  mouse_tracker_kalman.m by Joseph Betthauser 2015
#     Description: Kalman Filter (pos, vel, accel)
# /////////////////////////////////////////////////////////////////////////
import time

import keyboard
import matplotlib.pyplot as plt
import mouse
import numpy as np
from numba import njit

# my type
ArrayTuple = tuple[np.ndarray[float]]


def kalman_init(dt_: float) -> ArrayTuple:
    """
    _summary_

    Returns:
        ArrayTuple: _description_

    """
    A_ = np.array(
        [
            [1, 0, dt_, 0, 0.5 * dt_**2, 0],
            [0, 1, 0, dt_, 0, 0.5 * dt_**2],
            [0, 0, 1, 0, dt_, 0],
            [0, 0, 0, 1, 0, dt_],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    B_ = np.eye(6)
    H_ = np.eye(6)
    Q_ = 0.001 * np.eye(6)
    R_ = 0.1 * np.eye(6)

    x_ = np.zeros((6, 1))  # initial state
    c_ = np.zeros((6, 1))  # goal
    P_ = np.zeros((6, 6))

    return A_, B_, H_, Q_, R_, c_, x_, P_


@njit
def kalman_correction(
    H_: np.ndarray[float],
    R_: np.ndarray[float],
    x_: np.ndarray[float],
    z_: np.ndarray[float],
    P_: np.ndarray[float],
) -> ArrayTuple:
    """
    _summary_

    Args:
        H (np.ndarray[float]): _description_
        R (np.ndarray[float]): _description_
        x (np.ndarray[float]): _description_
        z (np.ndarray[float]): _description_
        P (np.ndarray[float]): _description_

    Returns:
        ArrayTuple: _description_

    """
    S_ = H_ @ P_ @ H_.T + R_
    K_ = P_ @ H_.T @ np.linalg.pinv(S_)
    y_ = z_ - H_ @ x_
    x_ = x_ + K_ @ y_
    P_ = (np.eye(6) - K_ @ H_) @ P_
    return x_, P_


@njit
def kalman_predict(
    A_: np.ndarray[float],
    B_: np.ndarray[float],
    Q_: np.ndarray[float],
    c_: np.ndarray[float],
    x_: np.ndarray[float],
    P_: np.ndarray[float],
) -> ArrayTuple:
    """
    _summary_

    Args:
        A (np.ndarray[float]): _description_
        B (np.ndarray[float]): _description_
        Q (np.ndarray[float]): _description_
        c (np.ndarray[float]): _description_
        x (np.ndarray[float]): _description_
        P (np.ndarray[float]): _description_

    Returns:
        ArrayTuple: _description_

    """
    x_ = A_ @ x_ + B_ @ c_
    P_ = A_ @ P_ @ A_.T + Q_
    return x_, P_


def main() -> None:
    # root = tk.Tk()

    screen_width = 2400  # root.winfo_screenwidth()
    screen_height = 1600  # root.winfo_screenheight()

    plt.ion()  # interactive plots on

    ## init vars
    measureNoise = 1
    dt = 0.25
    dim_fix = screen_height
    num_poses = 20  # vars track last N values for green trace
    # ramp = np.linspace(0, 1, num_poses)
    last_poses = np.nan * np.zeros((2, num_poses))

    ## init kalman filter params
    [A, B, H, Q, R, c, x, P] = kalman_init(dt)

    ## Make program respond to mouse movements

    dist2target = 20

    fig, ax = plt.subplots(figsize=(15, 9))
    # fig.canvas.manager.full_screen_toggle()
    measurement_scatter = ax.scatter(0, 0, c="r", marker="o", s=20, label="measurement")
    trace_plot = ax.plot(
        last_poses[0, :],
        last_poses[1, :],
        c="g",
        lw=1,
        label="kalman tracker",
    )[0]
    cursor_scatter = ax.scatter(0, 0, c="k", marker="+", s=50, label="true cursor loc.")
    ax.axis([0, screen_width, 0, screen_height])

    # plt.xlabel("x-pos.")
    # plt.ylabel("y-pos.")
    # plt.title("Kalman Filter Mouse Tracker")
    # plt.legend()

    print('Press "ESC" key to quit')

    while True:
        t0 = time.perf_counter()

        cursor = np.array(mouse.get_position())
        cursor[1] = dim_fix - cursor[1]
        xtrue, ytrue = cursor

        lastx = x  # save last state values before updates

        ## Kalman Position/Vel/Accel Updates
        x, P = kalman_predict(A, B, Q, c, x, P)

        ## Get noisy measurement
        nx = [xtrue + measureNoise * dist2target * np.random.randn()]
        ny = [ytrue + measureNoise * dist2target * np.random.randn()]
        vx = nx - lastx[0]  # est. x-velocity from measurement
        vy = ny - lastx[1]  # est. y-velocity from measurement
        ax = vx - lastx[2]  # est. x-accel from measurement
        ay = vy - lastx[3]  # est. y-accel from measurement
        z_list = [nx, ny, vx, vy, ax, ay]
        z_vec = np.reshape(np.array(z_list), (6, 1))

        ## Measurement Updates
        x, P = kalman_correction(H, R, x, z_vec, P)
        dist2target = np.sqrt((x[0, :] - lastx[0, :]) ** 2 + (x[1, :] - lastx[1, :]) ** 2)
        # dist2target = 1

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
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-tmain:.3f} seconds.")
