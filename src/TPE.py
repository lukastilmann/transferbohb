import statsmodels.api as sm
import math
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
import numpy as np


class TPE:

    def __init__(self, configspace, top_quantile, sample_n, configs=None, results=None):

        self.configspace = configspace
        self.n_min = len(self.configspace.values()) + 1
        self.top_quantile = top_quantile
        self.sample_n = sample_n
        self.data_types = ""

        for p in list(self.configspace.values()):
            if isinstance(p, UniformFloatHyperparameter):
                self.data_types += "u"
            if isinstance(p, CategoricalHyperparameter):
                self.data_types += "c"
            else:
                "Added hyperparameter with currently unsupported data type. As of now, only continuous and categorical hyperparams are supported."

        if configs and results:
            self.configs = configs
            self.results = results

            if len(self.configs) > self.n_min:
                self.update_configs()
                self.update_models()

            else:
                self.l = None
                self.g = None
                self.configs_good = None
                self.conigs_bad = None
        else:
            self.configs = []
            self.results = []
            self.configs_good = None
            self.configs_bad = None
            self.l = None
            self.g = None

    def add_result(self, config, result):
        self.configs.append(config)
        self.results.append(result)

        if len(self.configs) > self.n_min:
            self.update_configs()
            self.update_models()

    def update_configs(self):
        top_q_n = math.floor(len(self.results) * self.top_quantile)
        good_n = max(self.n_min, top_q_n)
        bad_n = max(self.n_min, len(self.results) - top_q_n)
        loss_sorted = np.argsort(self.results)

        ix_good = loss_sorted[:good_n]
        ix_bad = loss_sorted[-bad_n:]
        configs_arr = np.array(list([config.get_array() for config in self.configs]))
        self.configs_good = configs_arr[ix_good]
        self.configs_bad = configs_arr[ix_bad]

    def update_models(self):

        self.l = sm.nonparametric.KDEMultivariate(data=self.configs_good, var_type=self.data_types)
        self.g = sm.nonparametric.KDEMultivariate(data=self.configs_bad, var_type=self.data_types)
        print("models fit")

    def sample(self):
        n = 2
        if not self.configs:
            return self.configspace.sample_configuration()

        else:
            # Randomly select n data points from the original data
            sampled_indices = np.random.choice(len(self.configs_good), size=n, replace=False)
            sampled_data = self.configs_good[sampled_indices]

            # TODO: multiply with bandwidth factor
            cov = np.diag(self.l.bw ** 2)
            mean = np.zeros(len(cov))
            print(self.l.bw)

            noise = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
            sampled_data_with_noise = sampled_data + noise

            densities = list(self.l.pdf(d) for d in sampled_data_with_noise)
            index_max = np.argmax(densities)
            return sampled_data_with_noise[index_max]