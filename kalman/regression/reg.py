import matplotlib.pyplot as plt


def lin_reg_model_approx(y, x):
    """Return a0, a1 model approximation y = a0 + a1*x given dataset (y, x)."""
    assert len(y) == len(x)
    m = len(y)
    x_sum = 0.0
    y_sum = 0.0
    xy_sum = 0.0
    xsqr_sum = 0.0
    for i in range(m):
        x_sum += x[i]
        y_sum += y[i]
        xy_sum += x[i] * y[i]
        xsqr_sum += x[i] ** 2

    a1 = (m * xy_sum - x_sum * y_sum) / (m * xsqr_sum - x_sum**2)
    a0 = 1 / m * y_sum - a1 / m * x_sum

    return a0, a1


def plot_regression_with_data(
    y, x, fx, title="Regression Results", xlabel="x", ylabel="y/fx"
):
    plt.figure()
    plt.title(title)
    plt.plot(x, y, "o")
    plt.plot(x, fx)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
