import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore')


class MixPoison(object):

    def __init__(self, ini_lam=1, ini_pi=0.5, ini_a_hat=0.1, ini_b_hat=0.1, ini_alpha=0.1, random_state=None):
        self.initial_values = {
            "lam": ini_lam,
            "pi": ini_pi,
            "a_hat": ini_a_hat,
            "b_hat": ini_b_hat,
            "alpha": ini_alpha
        }
        self.ini_lam = ini_lam
        self.ini_pi = ini_pi
        self.ini_a_hat = ini_a_hat
        self.ini_b_hat = ini_b_hat
        self.ini_alpha = ini_alpha

        self.cluster_proba = None
        self.estimate_cluster = None

        self.estimate_cluster = None
        self.max_iter = None
        self.last_data = None

        self.real_lam = None

        self.lam_history = []
        self.pi_history = []

        self.random_state = random_state

    def generate_data(self, size, cluster=2, plot_dist=True):
        np.random.seed(self.random_state)
        self.real_lam = np.zeros((cluster))
        for k in range(cluster):
            self.real_lam[k] = np.random.randint(k * 10, (k + 1) * 10, size=1)

        data_matrix = np.zeros((cluster, size))
        plt.figure()
        for k in range(cluster):
            print("cluster {}: mean={}".format(k, self.real_lam[k]))
            data_matrix[k, :] = np.random.poisson(
                lam=self.real_lam[k], size=size)
            if plot_dist:
                sns.distplot(data_matrix[k, :], kde=False, rug=False, label="lam={}".format(
                    self.real_lam[k]), bins=10)
                plt.legend()
        data_matrix = data_matrix.reshape(-1)

        if plot_dist:
            plt.title("colored poison")
            plt.figure()
            sns.distplot(data_matrix, bins=20)
            plt.title("all poison")

        self.last_data = data_matrix

        return data_matrix

    def fit(self, data, est_cluster, method="gibs", max_iter=50):
        np.random.seed(self.random_state)

        data = data.reshape(-1, 1)
        self.cluster_num = est_cluster
        self.max_iter = max_iter
        self.last_data = data
        print("estimate cluster number = {}".format(est_cluster))
        print("--------------------------------------")
        print("initial values are follows:")
        for key in self.initial_values.keys():
            print("{}:{}".format(key, self.initial_values[key]))

        a_hat = np.ones((1, est_cluster)) * self.ini_a_hat
        b_hat = np.ones((1, est_cluster)) * self.ini_b_hat
#         alpha_hat = np.ones((1, est_cluster)) * self.ini_alpha
        alpha_hat = np.random.rand(1, est_cluster)

        if method == "gibs":
            _lam = np.ones((1, est_cluster)) * self.ini_lam
            _pie = np.ones((1, est_cluster)) * self.ini_pi
            self.fit_gibs_sampling(
                data, est_cluster, _lam, _pie, a_hat, b_hat, alpha_hat, max_iter=max_iter)
        elif method == "var":
            self.fit_variational_inference(
                data, est_cluster, a_hat, b_hat, alpha_hat, max_iter=max_iter)
        elif method == "col_gibs":
            self.fit_collapse_gibs_sampling(
                data, est_cluster, a_hat[0], b_hat[0], alpha_hat[0], max_iter)

    def fit_variational_inference(self, data, est_cluster, a, b, alpha, max_iter=50):
        self.lam_history = np.zeros((max_iter, est_cluster))
        self.pi_history = np.zeros((max_iter, est_cluster))

        a_hat = a.copy()
        b_hat = b.copy()
        alpha_hat = alpha.copy()

        print("start fitting...")
        for iteration in tqdm(range(max_iter)):
            _loglam = scipy.special.digamma(a_hat) - np.log(b_hat)
            _lam = a_hat / b_hat
            _logpie = scipy.special.digamma(
                alpha_hat) - scipy.special.digamma(np.sum(alpha_hat))
            # data shape = (data_len, 1)
            # param shape = (1, category_len)
            # nu shape = (data_len, category_len)

            nu = np.exp(data * _loglam - _lam + _logpie)
            nu = nu / np.sum(nu, axis=1, keepdims=True)

            a_hat = np.sum(data * nu, axis=0) + a
            b_hat = np.sum(nu, axis=0) + b

            alpha_hat = np.sum(nu, axis=0) + alpha

            self.pi_history[iteration, :] = np.random.dirichlet(
                alpha_hat[0], 1)
            self.lam_history[iteration, :] = stats.gamma.rvs(
                a_hat[0], scale=1 / b_hat[0], size=est_cluster, random_state=0)

        for cls in range(est_cluster):
            print("estimation of cluster {}: lambda = {:.2f}, real={}".format(
                cls, self.lam_history[iteration, cls], self.real_lam[cls]))
        _s = np.zeros((data.shape[0], est_cluster))
        for _n in range(data.shape[0]):
            cat = stats.rv_discrete(name='custm', values=(
                range(est_cluster), nu[_n, :]))
            _class = cat.rvs(size=1)
            _s[_n, _class] = 1
        self.estimate_cluster = _s
        self.cluster_proba = nu

    def fit_collapse_gibs_sampling(self, data, est_cluster, a, b, alpha, max_iter=50):
        self.lam_history = np.zeros((max_iter, est_cluster))
        self.pi_history = np.zeros((max_iter, est_cluster))

        # initialize
        p = np.ones(est_cluster, dtype=np.float) / est_cluster
        cat = stats.rv_discrete(name='custm', values=(range(est_cluster), p))
        _class = cat.rvs(size=data.shape[0])
        _s = np.zeros((data.shape[0], est_cluster))
        _s[:, _class] = 1
        alpha_hat = np.sum(_s, axis=0) + alpha
        a_hat = np.sum(_s * data, axis=0) + a
        b_hat = np.sum(_s, axis=0) + b

        self.cluster_proba = np.zeros((data.shape[0], est_cluster))

        for iteration in tqdm(range(max_iter)):
            for _n in range(data.shape[0]):
                alpha_hat = alpha_hat - _s[_n, :]
                a_hat = a_hat - _s[_n, :] * data[_n]
                b_hat = b_hat - _s[_n, :]

                _nu = alpha_hat / np.sum(alpha_hat)
                _nb = stats.nbinom.pmf(np.broadcast_to(
                    data[_n], est_cluster), a_hat, 1 - 1 / b_hat)

                _nu = _nu * _nb
                _nu = _nu / np.sum(_nu)

                cat = stats.rv_discrete(
                    name='custm', values=(range(est_cluster), _nu))
                _class = cat.rvs(size=1)
                tmp_s = np.identity(est_cluster)[_class][0]

                a_hat += tmp_s * data[_n]
                b_hat += tmp_s
                alpha_hat += tmp_s
                _s[_n, :] = tmp_s
                self.cluster_proba[_n, :] = _nu
            self.pi_history[iteration, :] = np.random.dirichlet(alpha_hat, 1)
            self.lam_history[iteration, :] = stats.gamma.rvs(
                a_hat, scale=1 / b_hat, size=est_cluster, random_state=0)
        for cls in range(est_cluster):
            print("estimation of cluster {}: lambda = {:.2f}, real={}".format(
                cls, self.lam_history[iteration, cls], self.real_lam[cls]))

    def fit_gibs_sampling(self, data, est_cluster, _lam, _pie, a, b, alpha, max_iter=50):

        self.lam_history = np.zeros((max_iter, est_cluster))
        self.pi_history = np.zeros((max_iter, est_cluster))
        a_hat = a.copy()
        b_hat = b.copy()
        alpha_hat = alpha.copy()
        print("start fitting...")
        for iteration in tqdm(range(max_iter)):
            # sample s
            _nu = np.exp(data * np.log(_lam) - (_lam - np.log(_pie)))
            _nu = _nu / np.sum(_nu, axis=1, keepdims=True)
            _s = np.zeros((data.shape[0], est_cluster))
            for _n in range(data.shape[0]):
                cat = stats.rv_discrete(name='custm', values=(
                    range(est_cluster), _nu[_n, :]))
                _class = cat.rvs(size=1)
                _s[_n, _class] = 1

            # sample lam
            a_hat = np.sum(data * _s, axis=0) + a
            b_hat = np.sum(_s, axis=0) + b
            alpha_hat = np.sum(_s, axis=0) + alpha

            _lam = stats.gamma.rvs(
                a_hat[0], scale=1 / b_hat[0], size=est_cluster, random_state=0)
            _pie = np.random.dirichlet(alpha_hat[0], 1)
            self.lam_history[iteration, :] = _lam
            self.pi_history[iteration, :] = _pie
        self.estimate_cluster = _s
        self.cluster_proba = _nu

        for cls in range(est_cluster):
            print("estimation of cluster {}: lambda = {:.2f}, real={}".format(
                cls, _lam[cls], self.real_lam[cls]))

    def plot_result(self, method):
        plt.figure()
        print(self.lam_history.shape)
        print(self.pi_history.shape)
        print(self.cluster_proba.shape)

        for cls in range(self.cluster_num):
            plt.plot(np.arange(self.max_iter),
                     self.lam_history[:, cls], label="lam class={}".format(cls))
            plt.legend()
        plt.title("estimation lam")
        plt.show()
#         plt.figure()
#         for cls in range(self.cluster_num):
#             plt.plot(np.arange(self.max_iter), self.pi_history[:, cls], label="lam class={}".format(cls))
#             plt.legend()
#         plt.title("estimation pi")
#         plt.show()

        plt.figure()
        plt.scatter(data, self.cluster_proba[:, 0], color="purple",
                    label="class=0, lam={:.1f}".format(self.lam_history[-1][0]))
        plt.scatter(data, self.cluster_proba[:, 1], color="skyblue",
                    label="class=1, lam={:.1f}".format(self.lam_history[-1][1]))
        plt.title("estimated probability belonging to each category")
        plt.legend()
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--max_iter",
                    help="maximum iteration for inferance.",
                    default=10,
                    type=int)
parser.add_argument("--method",
                    help="mthod to infer",
                    default="var",
                    choices=["var", "gibs", "col_gibs"])
parser.add_argument("--cluster_num",
                    help="cluster_number.",
                    default=2,
                    type=int)
parser.add_argument("--data_size",
                    help="number of data to generate.",
                    default=100,
                    type=int)
args = parser.parse_args()

if __name__ == "__main__":
    poitest = MixPoison(random_state=1)
    data = poitest.generate_data(args.data_size, cluster=args.cluster_num)
    poitest.fit(data, args.cluster_num,
                max_iter=args.max_iter, method=args.method)
    poitest.plot_result(method=args.method)
