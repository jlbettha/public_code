# pylint: disable=C0103
""" Betthauser, 2020: kalman filter + example
    For linear systems of the form:
        x(n+1) = Ax(n) + er_x ~ N(0,Q)
        y(n)   = Cx(n) + er_y ~ N(0,R)

        init:
        x(0) = [0, 0].T
        P_init_prior (P_1|0) = (C^T @ R^-1 @ C)^-1
"""

import time
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import mouse
import keyboard


def kalman_predict(
    A: NDArray[np.float64],
    x_post: NDArray[np.float64],
    P_post: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> tuple[NDArray[np.float64]]:
    """_summary_

    Args:
        A (NDArray[np.float64]): State transition model
        x_post (NDArray[np.float64]): Final state est.
        P_post (NDArray[np.float64]): Posterior covariance est.
        Q (NDArray[np.float64]): Expected process noise

    Returns:
        tuple[NDArray[np.float64]]: (x_next_prior, P_next_prior)
    """
    x_next_prior = A @ x_post + np.random.multivariate_normal(np.zeros(2), Q)
    P_next_prior = A @ P_post @ A.T + Q
    return (x_next_prior, P_next_prior)


def kalman_update(
    C: NDArray[np.float64],
    P_last_prior: NDArray[np.float64],
    R: NDArray[np.float64],
    y_now: NDArray[np.float64],
    x_last_prior: NDArray[np.float64],
) -> tuple[NDArray[np.float64]]:
    """_summary_

    Args:
        C (NDArray[np.float64]): Measurement model
        P_last_prior (NDArray[np.float64]): prior covariance est.
        R (NDArray[np.float64]): Expected measurement noise
        y_now (NDArray[np.float64]): Actual measurement
        x_last_prior (NDArray[np.float64]): prior predicted state

    Returns:
        tuple[NDArray[np.float64]]: (x_post, P_post)
    """

    k_n = P_last_prior @ C.T @ np.linalg.pinv(C @ P_last_prior @ C.T + R)

    P_post = (np.eye(P_last_prior.shape[1]) - k_n @ C) @ P_last_prior

    x_post = x_last_prior + k_n @ (y_now - C @ x_last_prior)

    return (x_post, P_post)


def main() -> None:
    """For linear systems of the form:
    # x(n+1) = Ax(n) + er_x ~ N(0,Q)
    # y(n)   = Cx(n) + er_y ~ N(0,R)

    # x_1|0 = [0, 0].T
    # P_init (P_1|0) = (C^T @ R^-1 @ C)^-1
    """

    plt.ion()  # interactive plots on

    # init vars
    delta_t_sec = 1 / 16
    dim_fix = 767
    er_x = 0.01
    er_y = 0.01
    x_prior = np.zeros(2)
    Q = np.array([[er_x, 0], [0, er_x]])
    R = np.array([[er_y, 0], [0, er_y]])
    A = np.eye(2)
    C = np.eye(2)
    P_prior = np.linalg.pinv(C.T @ np.linalg.pinv(R) @ C)

    y_now = np.array(mouse.get_position())
    y_now[1] = dim_fix - y_now[1]

    print(
        f"({y_now[0]}, {y_now[1]}) --- ({x_prior[0]:.2f}, {x_prior[1]:.2f})", flush=True
    )

    x_post, P_post = kalman_update(C, P_prior, R, y_now, x_prior)

    # begin tracking mouse
    times = []
    while True:
        t_mark = time.time()

        x_pred, P_next_prior = kalman_predict(A, x_post, P_post, Q)

        y_now = np.array(mouse.get_position())
        y_now[1] = dim_fix - y_now[1]

        print(
            f"Measured >> ({y_now[0]}, {y_now[1]}) --- ({x_pred[0]:.2f}, {x_pred[1]:.2f}) << Kalman",
            flush=True,
        )

        x_post, P_post = kalman_update(C, P_next_prior, R, y_now, x_pred)

        t_loop = time.time() - t_mark
        times.append(t_loop)

        # Listen for ESC or 'q' key
        if keyboard.is_pressed("escape") or keyboard.is_pressed("space"):
            break

        if t_loop > delta_t_sec:
            continue

        time.sleep(delta_t_sec - t_loop)

    print(f"Average loop time: {np.mean(times):.3f} seconds.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
