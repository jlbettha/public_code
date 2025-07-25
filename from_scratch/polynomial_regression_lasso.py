import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures


def main() -> None:  # noqa: PLR0915
    """_summary_"""
    noise_level = 5

    coeffs = np.array([3, 0, 2.5, 1, 0, 1, 0, 0.2, 0.5, 0, 0.0])
    poly_degree = len(coeffs) - 1
    xs = np.linspace(-1.7, 1.7, 35)

    f_x = (
        np.array(
            [
                np.ones(xs.shape),
                xs,
                xs**2,
                xs**3,
                xs**4,
                xs**5,
                xs**6,
                xs**7,
                xs**8,
                xs**9,
                xs**10,
            ]
        ).T
        @ coeffs
    )

    ## add noise
    rng = np.random.default_rng()
    err = noise_level * rng.normal(size=len(f_x))
    f_x_er = f_x + err

    xs = xs[..., np.newaxis]
    # f_x = f_x[..., np.newaxis]
    step = 0.02
    x_grid = np.arange(np.min(xs), np.max(xs) + step, step)
    x_grid = x_grid.reshape(len(x_grid), 1)

    poly_reg = PolynomialFeatures(degree=poly_degree + 1)
    x_poly = poly_reg.fit_transform(xs)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, f_x_er)

    lin_reg_l2 = Ridge(alpha=0.2)
    lin_reg_l2.fit(x_poly, f_x_er)

    lin_reg_l1 = Lasso(alpha=0.2, tol=1e-7, max_iter=10000)
    lin_reg_l1.fit(x_poly, f_x_er)

    # plots
    grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0.3)

    plt.figure(figsize=(8, 6))
    plt.subplot(grid[:, 0:2])
    ax = plt.gca()
    ax.scatter(xs, f_x_er, color="green", s=5)
    ax.grid(visible=True)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")
    ax.plot(xs, f_x, label="Truth", c="k")
    ax.plot(
        x_grid,
        lin_reg.predict(poly_reg.fit_transform(x_grid)),
        label="L2 Est.",
        c="c",
    )
    ax.plot(
        x_grid,
        lin_reg_l2.predict(poly_reg.fit_transform(x_grid)),
        label="Ridge Est.",
        c="m",
    )
    ax.plot(
        x_grid,
        lin_reg_l1.predict(poly_reg.fit_transform(x_grid)),
        label="Lasso Est.",
        c="orange",
    )
    plt.ylim([-10, 60])
    plt.xlabel("x")
    plt.ylabel("f(x)")
    ax.legend()

    x_reg = np.hstack((lin_reg.intercept_, lin_reg.coef_[1:]))
    x_l2 = np.hstack((lin_reg_l2.intercept_, lin_reg_l2.coef_[1:]))
    x_l1 = np.hstack((lin_reg_l1.intercept_, lin_reg_l1.coef_[1:]))
    real_coeffs = coeffs[: len(x_reg)]

    plt.rcParams.update({"font.size": 8})
    plt.subplot(grid[0, 2])
    plt.stem(real_coeffs, "k", markerfmt="ko", label="Actual")
    plt.stem(x_reg, "m--", markerfmt="mx", label="L2 estimate")
    plt.title(f"weight-sum = {np.sum(np.abs(x_reg)):.2f}", fontsize=8)
    plt.legend(fontsize=8)

    plt.subplot(grid[1, 2])
    plt.stem(real_coeffs, "k", markerfmt="ko", label="Actual")
    plt.stem(x_l2, "m--", markerfmt="mx", label="Ridge est.")
    plt.title(f"weight-sum = {np.sum(np.abs(x_l2)):.2f}", fontsize=8)
    plt.legend(fontsize=8)

    plt.subplot(grid[2, 2])
    plt.stem(real_coeffs, "k", markerfmt="ko", label="Actual")
    plt.stem(x_l1, "m--", markerfmt="mx", label="Lasso est.")
    plt.title(f"weight-sum = {np.sum(np.abs(x_l1)):.2f}", fontsize=8)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tmain = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter() - tmain:.3f} seconds.")
