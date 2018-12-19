import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import warnings
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

    def fit(self, data, est_cluster, max_iter=50):
        np.random.seed(self.random_state)

        self.cluster_num = est_cluster
        self.max_iter = max_iter
        self.last_data = data
        print("estimate cluster number = {}".format(est_cluster))
        print("--------------------------------------")
        print("initial values are follows:")
        for key in self.initial_values.keys():
            print("{}:{}".format(key, self.initial_values[key]))
        _lam = np.ones(est_cluster) * self.ini_lam
        _pie = np.ones(est_cluster) * self.ini_pi

        a_hat = np.ones(est_cluster) * self.ini_a_hat
        b_hat = np.ones(est_cluster) * self.ini_b_hat
        alpha_hat = np.ones(est_cluster) * self.ini_alpha

        self.lam_history = np.zeros((max_iter, est_cluster))
        self.pi_history = np.zeros((max_iter, est_cluster))

        print("start fitting...")
        for iteration in tqdm(range(max_iter)):
            # sample s
            _nu = np.exp(data * np.log(_lam).reshape(-1, 1) -
                         (_lam - np.log(_pie)).reshape(-1, 1))
            _nu = _nu / np.sum(_nu, axis=0)
            _s = np.zeros((est_cluster, data.shape[0]))
            for _n in range(data.shape[0]):
                cat = stats.rv_discrete(name='custm', values=(
                    range(est_cluster), _nu[:, _n]))
                _class = cat.rvs(size=1)
                _s[_class, _n] = 1

            # sample lam
            for clust in range(est_cluster):
                a_hat[clust] = np.sum(data * _s[clust]) + a_hat[clust]
                b_hat[clust] = np.sum(_s[clust]) + b_hat[clust]
                alpha_hat[clust] = np.sum(_s[clust]) + alpha_hat[clust]
                _lam[clust] = stats.gamma.rvs(
                    a_hat[clust], scale=1 / b_hat[clust], size=1, random_state=0)
            _pie = np.random.dirichlet(alpha_hat, 1)
            self.lam_history[iteration, :] = _lam
            self.pi_history[iteration, :] = _pie
        self.estimate_cluster = _s
        self.cluster_proba = _nu

        for cls in range(est_cluster):
            print("estimation of cluster {}: lambda = {:.2f}, real={}".format(
                cls, _lam[cls], self.real_lam[cls]))

    def plot_result(self):
        plt.figure()

        for cls in range(self.cluster_num):
            plt.plot(np.arange(self.max_iter),
                     self.lam_history[:, cls], label="lam class={}".format(cls))
            plt.legend()
        plt.title("estimation lam")
        plt.show()
        plt.figure()
        for cls in range(self.cluster_num):
            plt.plot(np.arange(self.max_iter),
                     self.pi_history[:, cls], label="lam class={}".format(cls))
            plt.legend()
        plt.title("estimation pi")
        plt.show()

        plt.figure()
        plt.scatter(self.last_data, self.cluster_proba[0], color="purple", label="class=0, lam={:.1f}".format(
            self.lam_history[-1][0]))
        plt.scatter(self.last_data, self.cluster_proba[1], color="skyblue",
                    label="class=1, lam={:.1f}".format(self.lam_history[-1][1]))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    poitest = MixPoison(random_state=1)
    data = poitest.generate_data(100)
    poitest.fit(data, 2)
    poitest.plot_result()
