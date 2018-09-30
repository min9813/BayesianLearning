import numpy as np
import matplotlib.pyplot as plt
import math
import sys


def poly_x_vector(x, n_dim=3):
    # out = (n_dim+1, data_size)
    y = [x**dim for dim in range(n_dim + 1)]
    return np.array(y)


def demo_linear_gaussian(seed=0, data_lam=0.5, n_dim=2):
    np.random.seed(seed)
    n_dim = n_dim
    pre_w_loc = np.zeros(n_dim + 1)
    pre_w_lam = np.eye(n_dim + 1)

    pre_data_lam = data_lam

    data_size = 50
    x = np.random.uniform(-5, 5, data_size)
    x_ground = np.linspace(-5, 5, 100)

    w = np.random.multivariate_normal(
        mean=pre_w_loc, cov=np.linalg.inv(pre_w_lam), size=1)

    y_mean = list(np.dot(w, poly_x_vector(x, n_dim=n_dim)).reshape(-1))
    y = np.array([np.random.normal(_y, 1. / pre_data_lam) for _y in y_mean])
    y_ground = np.dot(w, poly_x_vector(x_ground, n_dim=n_dim))

    plt.scatter(x, y, color="B", alpha=0.5)
    plt.plot(x_ground, y_ground.reshape(-1))


def y_line(x):
    return 3 * np.sin(x)


def fit_linear(f, seed=0, n_dim=2, ax=None, data_size=10, data_lam=3, fit_result=True):
    np.random.seed(seed)
    pre_w_loc = np.zeros(n_dim + 1)
    pre_w_lam = np.eye(n_dim + 1)

    x_data = np.random.uniform(-3, 3, data_size)
    y_data = f(x_data)
    x_ground = np.linspace(-3, 3, 100)
    y_ground = f(x_ground)
    x_test = poly_x_vector(x_ground, n_dim=n_dim)

    x_for_predict = poly_x_vector(x_data, n_dim=n_dim)

    est_w_lam = data_lam * np.dot(x_for_predict, x_for_predict.T) + pre_w_lam
    est_w_lam_inv = np.linalg.inv(est_w_lam)
    _m1 = data_lam * np.sum(y_data * x_for_predict,
                            axis=1) + np.dot(pre_w_lam, pre_w_loc)
    est_w_loc = np.dot(est_w_lam_inv, _m1)

    est_data_loc = np.dot(est_w_loc.T, x_test)
    _m1 = np.dot(x_test.T, est_w_lam_inv)
    est_data_lam = 1. / (1 / data_lam + np.sum(_m1.T * x_test, axis=0))

    y_mean = est_data_loc
    y_one_sigma = (est_data_loc + 1. / est_data_lam,
                   est_data_loc - 1. / est_data_lam)

    if fit_result:
        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
        ax.scatter(x_data, y_data, color="black", alpha=0.5, label="True Data")
        ax.plot(x_ground, y_ground.reshape(-1), color="B", label="True Line")
        ax.plot(x_ground, y_mean, color="orange",
                alpha=0.5, label="Estimate Mean Line")
        ax.plot(x_ground, y_one_sigma[0],
                color="pink", linestyle="dashed", alpha=0.5)
        ax.plot(x_ground, y_one_sigma[1],
                color="pink", linestyle="dashed", alpha=0.5)

        plt.legend()

        plt.ylim(-5, 5)
        plt.title("dimension = {}".format(n_dim))

    _m1 = np.sum(data_lam * y_data**2 -
                 np.log(data_lam) + math.log(2 * math.pi))
    _m2 = np.dot(np.dot(pre_w_loc, pre_w_lam), pre_w_loc) - \
        math.log(np.linalg.det(pre_w_lam))

    _m3 = -np.dot(np.dot(est_w_loc, est_w_lam), est_w_loc) + \
        math.log(np.linalg.det(est_w_lam))

    log_model_evidence = -0.5 * (_m1 + _m2 + _m3)
    print("dim = {} model evidence:".format(n_dim), log_model_evidence)
    return log_model_evidence


def check_input(text):
    text = text.lower()
    if text in ("y", "yes"):
        return True
    elif text in ("n", "no"):
        return False
    else:
        sys.exit("Input text must be 'y','yes','n','no'")


if __name__ == "__main__":
    max_dim = 10
    flag_need_result = check_input(input("Need fit result image?Y or N>"))

    log_model_evidence = [fit_linear(
        y_line, data_size=10, n_dim=i, data_lam=1, fit_result=flag_need_result) for i in range(max_dim + 1)]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(range(max_dim + 1), log_model_evidence, label="log model evidence")
    ax.set_ylabel("logP(Y|X)")
    ax.set_xlabel("n_dim")
    plt.title("Model Evidence")
    plt.legend()

    plt.show()
