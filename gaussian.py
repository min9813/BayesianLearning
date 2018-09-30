import numpy as np
import scipy as sp
from scipy.stats import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import argparse


sns.set()


class multivariate_student_t():

    def __init__(self, mu, lam, nu):
        try:
            self.D = mu.shape[0]
        except AttributeError:
            self.D = 1
        if self.D > 1:
            self.mu = mu
            if len(self.mu.reshape(-1)) != D:
                sys.exit("parametor 'mu' must be vector!")
            if lam.shape[0] != lam.shape[1]:
                sys.exit("expect lam.shape[0] == lam.shape[1], Actual:{}!={}".format(
                    lam.shape[0], lam.shape[1]))
            self.lam = lam
        else:
            if lam <= 0:
                sys.exit("parametor 'lam' must > 0")
            self.mu = mu
            self.lam = lam
        if nu <= 0:
            sys.exit("parametor 'nu' must > 0")
        self.nu = nu

    def pdf(self, x):
        m1 = np.exp(math.lgamma((self.nu + self.D) * 0.5) -
                    math.lgamma(self.nu * 0.5))

        if self.D == 1:
            _x = x.reshape(1, -1)
            m2 = np.sqrt(self.lam / (math.pi * self.nu))
            m3 = (1 + self.lam / self.nu *
                  np.square(_x - self.mu))**(-(self.nu + 1) * 0.5)
        else:
            _x = x.reshape(-1, self.D)
            self.mu = self.mu.reshape(1, self.D)
            m2 = np.sqrt(np.linalg.det(self.lam)) / \
                (math.pi * self.nu)**(0.5 * self.D)
            m3 = np.sum(np.dot(_x - self.mu, self.lam)
                        * (_x - self.mu), axis=1)
            m3 = (1 + m3 / self.nu)**(-(self.nu + self.D) * 0.5)

        _y = m1 * m2 * m3
        return _y


def student(x, mu, lm, nu):
    gamma_1 = sp.special.gamma((nu + 1) * 0.5)
    gamma_2 = sp.special.gamma(nu * 0.5)
    y = gamma_1 / gamma_2 * np.sqrt(lm / (nu * np.pi)) * \
        ((1 + lm / nu * np.square(x - mu))**(-(nu + 1) * 0.5))

    return y


def draw_data_distribution(data_size, parametors,
                           ax=None,
                           color="skyblue",
                           distribution="norm",
                           graph_label="data distribution",
                           title="data distribution",
                           need_data=False,
                           seed=0):
    np.random.seed(seed)
    if distribution == "norm":
        print("generate gaussian distribution, loc={}, scale={}".format(
            parametors[0], parametors[1]))
        x_data = np.random.normal(parametors[0], 1 / parametors[1], data_size)
    if ax is None:
        figure = plt.figure(figsize=(6, 4))
        ax = figure.add_subplot(1, 1, 1)
        ax.hist(x_data, bins=30, density=True, color=color, label=graph_label)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("probability-density")
        ax.legend()

    else:
        ax.hist(x_data, bins=30, density=True,
                ec=color, label=graph_label, fill=False)
        ax.legend()
    if need_data:
        return ax, x_data
    else:
        return ax


def draw_function(parametors,
                  ax=None,
                  color="skyblue",
                  draw_range=None,
                  distribution="gauss",
                  label_prefix="",
                  title="estimate distribution",
                  seed=0):
    disp_param = [round(param, 2) for param in parametors]
    if distribution == "gauss":
        print("drawing {} graph, loc={} scale={}".format(
            distribution, parametors[0], parametors[1]))
        graph_label = label_prefix + \
            "{} μ={}, σ={}".format(distribution, disp_param[0], disp_param[1])
        if draw_range is None:
            draw_range = (parametors[0] - 3 * parametors[1],
                          parametors[0] + 3 * parametors[1])
        x = np.linspace(draw_range[0], draw_range[1], 1000)
        y = norm.pdf(x, loc=parametors[0], scale=1 / parametors[1])
    elif distribution == "gamma":
        print("drawing {} graph, a={} b={}".format(
            distribution, parametors[0], parametors[1]))
        graph_label = label_prefix + \
            "{} a={}, b={}".format(distribution, disp_param[0], disp_param[1])
        if draw_range is None:
            draw_range = (0, 10)
        x = np.linspace(draw_range[0], draw_range[1], 1000)
        y = gamma.pdf(x, a=parametors[0], scale=1 / parametors[1])
    elif distribution == "student":
        print("drawing student's T graph, μ={}, λ={}, ν={}".format(
            parametors[0], parametors[1], parametors[2]))
        graph_label = label_prefix + \
            "{} μ={}, λ={}, ν={}".format(
                distribution, disp_param[0], disp_param[1], disp_param[2])
        if draw_range is None:
            draw_range = (parametors[0] - 5, parametors[0] + 5)
        x = np.linspace(draw_range[0], draw_range[1], 1000)
        y = t.pdf(x, parametors[2], loc=parametors[0],
                  scale=1 / np.sqrt(parametors[1]))

    if ax is None:
        figure = plt.figure(figsize=(6, 4))
        ax = figure.add_subplot(1, 1, 1)
        ax.plot(x, y, color=color, label=graph_label)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel(title)
        ax.legend()

    else:
        ax.plot(x, y, color=color, label=graph_label)
        ax.legend()

    return ax


def draw_3d_distribution(data_size, parametors,
                         distribution="norm",
                         need_data=False,
                         cmap=None,
                         ax=None,
                         label_prefix="",
                         title="data distribution",
                         seed=0):
    np.random.seed(seed)
    if distribution == "norm":
        mean = parametors[0]
        sigma = np.linalg.inv(parametors[1])
        print("-----------------------------------------")
        print("generate multi-gaussian distribution")
        print("loc=", parametors[0])
        print("scale=", parametors[1])
        print("ndim={}".format(parametors[1].shape[0]))
        x1 = np.linspace(mean[0] - 4 * sigma[0, 0],
                         mean[0] + 4 * sigma[0, 0], data_size)
        x2 = np.linspace(mean[1] - 4 * sigma[1, 1],
                         mean[1] + 4 * sigma[1, 1], data_size)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.c_[np.ravel(X1), np.ravel(X2)]
        x_data = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
    elif distribution == "student":
        mu = parametors[0]
        lam = parametors[1]
        nu = parametors[2]
        print("-----------------------------------------")
        print("generate multi-student's t distribution")
        print("μ:")
        print(mu)
        print("Λ:")
        print(lam)
        print("ν:")
        print(nu)
        print("ndim={}".format(lam.shape[0]))
        _t = multivariate_student_t(mu, lam, nu)
        x1 = np.linspace(mu[0] - 5, mu[0] + 5, data_size)
        x2 = np.linspace(mu[1] - 5, mu[1] + 5, data_size)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.c_[np.ravel(X1), np.ravel(X2)]
        x_data = _t.pdf(x=X)

    if ax is None:
        figure = plt.figure(figsize=(12, 8))
        ax = figure.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, x_data.reshape(
            X1.shape), cmap=cmap, linewidth=0)
        ax.set_title(title)

    else:
        ax.plot_wireframe(X1, X2, x_data.reshape(
            X1.shape), color=cmap, linewidth=0.5)
        ax.set_title(title)

    if need_data:
        return ax, x_data
    else:
        return ax


parser = argparse.ArgumentParser(
    usage="""demonstration of Bayesian inference by gaussian. You can decide which parametor, mean or sigma is known, and start inference.
If you choose both parametor to be known, this program work as only sigma to bw known.""",
    description="""This file is\
 used to get 3D image, if you just want to try Bayesian Inference, you should use gaussian.ipynb""")
parser.add_argument("-m", "--mean",
                    help="is mean value known",
                    action="store_true")
parser.add_argument("-s", "--sigma",
                    help="is sigma value known", action="store_true")
parser.add_argument("-d", "--datasize",
                    help="size of real data", type=int, default=100)
args = parser.parse_args()


if __name__ == "__main__":
    real_mu = np.array([3, 5])
    real_sigma_inv = np.array([[10, 2],
                               [2, 5]])
    pre_sigma_nu = 3
    pre_sigma_w = np.array([[1, 0],
                            [0, 1]])

    pre_mu_loc = np.array([0, 0])
    pre_mu_beta = 1
    pre_mu_scale_inv = np.array([[1, 0],
                                 [0, 1]])

    data_size = args.datasize
    D = 2

    ax = draw_3d_distribution(data_size, (real_mu, real_sigma_inv),
                              distribution="norm",
                              cmap="gist_gray_r",
                              seed=3,
                              need_data=False)

    # 事前予測分布

    if args.sigma:
        _m1 = np.linalg.inv(real_sigma_inv)
        _m2 = np.linalg.inv(pre_mu_scale_inv)
        est_data_sigma_inv = np.linalg.inv(_m1 + _m2)
        est_data_mu = pre_mu_loc
        ax1 = draw_3d_distribution(data_size, (est_data_mu, est_data_sigma_inv),
                                   cmap="blue",
                                   ax=ax,
                                   title="pre-estimated distribution")

        ax = draw_3d_distribution(data_size, (real_mu, real_sigma_inv),
                                  distribution="norm",
                                  cmap="gist_gray_r",
                                  seed=3,
                                  need_data=False)
        est_mu_scale_inv = data_size * real_sigma_inv + pre_mu_scale_inv

        x_data = np.random.multivariate_normal(
            mean=real_mu, cov=np.linalg.inv(real_sigma_inv), size=data_size)

        _m1 = np.linalg.inv(est_mu_scale_inv)
        _m2 = np.dot(real_sigma_inv, np.sum(x_data, axis=0).reshape(-1, 1)
                     ) + np.dot(pre_mu_scale_inv, pre_mu_loc.reshape(-1, 1))
        est_mu_loc = np.dot(_m1, _m2)

        _m1 = np.linalg.inv(real_sigma_inv)
        _m2 = np.linalg.inv(est_mu_scale_inv)
        est_data_sigma_inv = np.linalg.inv(_m1 + _m2)
        est_data_mu = est_mu_loc

        ax = draw_3d_distribution(data_size, (est_data_mu.reshape(-1), est_data_sigma_inv),
                                  cmap="orange",
                                  ax=ax,
                                  title="post-estimated distribution")
    else:
        est_data_nu = 1 - D + pre_sigma_nu
        if args.mean:
            est_data_mu = real_mu
            est_data_lam = est_data_nu * pre_sigma_w
        else:
            est_data_mu = pre_mu_loc
            est_data_lam = (est_data_nu) * pre_mu_beta / \
                (1 + pre_mu_beta) * pre_sigma_w

        ax1 = draw_3d_distribution(data_size, (est_data_mu, est_data_lam, est_data_nu),
                                   ax=ax, cmap="blue",
                                   distribution="student",
                                   title="pre-estimated distribution")

        ax = draw_3d_distribution(data_size, (real_mu, real_sigma_inv),
                                  distribution="norm",
                                  cmap="gist_gray_r",
                                  seed=3,
                                  need_data=False)

        x_data = np.random.multivariate_normal(
            mean=real_mu, cov=np.linalg.inv(real_sigma_inv), size=data_size)
        # x_data.shape=(data_size,D)

        if args.mean:
            est_data_mu = real_mu
            print("x_data shape:", x_data.shape)
            est_sigma_w = np.linalg.inv(
                np.dot((x_data - real_mu).T, x_data - real_mu) + np.linalg.inv(pre_sigma_w))
            est_sigma_nu = data_size + pre_sigma_nu
            print("w shape:", est_sigma_w.shape)

            est_data_lam = (1 - D + est_sigma_nu) * est_sigma_w
            est_data_nu = 1 - D + est_sigma_nu
        else:
            # 平均のパラメーター更新
            est_mu_beta = data_size + pre_mu_beta
            est_mu_loc = (np.sum(x_data, axis=0) +
                          pre_mu_beta * pre_mu_loc) / est_mu_beta

            # 精度のパラメーター更新
            est_sigma_nu = data_size + pre_sigma_nu
            _m1 = np.dot(x_data.T, x_data)
            _m2 = pre_mu_beta * np.dot(pre_mu_loc.reshape(D, -1),
                                       pre_mu_loc.reshape(-1, D))
            _m3 = -est_mu_beta * np.dot(est_mu_loc.reshape(D, -1),
                                        est_mu_loc.reshape(-1, D))
            est_sigma_w = np.linalg.inv(
                _m1 + _m2 + _m3 + np.linalg.inv(pre_sigma_w))

            print("---------")
            print("estimated value:")
            print("m=", est_mu_loc)
            print("β=", est_mu_beta)
            print("ν=", est_sigma_nu)
            print("W=", est_sigma_w)

            est_data_mu = est_mu_loc
            est_data_nu = 1 - D + est_sigma_nu
            est_data_lam = (est_data_nu) * est_mu_beta * \
                est_sigma_w / (1 + est_mu_beta)
        ax = draw_3d_distribution(data_size, (est_data_mu, est_data_lam,
                                              est_data_nu), ax=ax, cmap="orange", distribution="student", title="post-estimate distribution")

    plt.show()
