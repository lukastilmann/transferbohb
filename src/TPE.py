import statsmodels.api as sm
import math
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace import Configuration
import numpy as np


class TPE:

  def __init__(self, configspace, top_quantile, sample_n, bw_factor, configs=None, results=None):

    self.configspace = configspace
    self.n_min = len(self.configspace.values()) + 1
    self.top_quantile = top_quantile
    self.sample_n = sample_n
    self.bw_factor = bw_factor
    self.data_types = ""

    for p in list(self.configspace.values()):
      if isinstance(p, UniformFloatHyperparameter):
        self.data_types += "c"
      if isinstance(p, CategoricalHyperparameter):
        self.data_types += "u"
      else:
        "Added hyperparameter with currently unsupported data type. As of now, only continuous and categorical hyperparams are supported."

    if configs and results:
      self.configs = configs
      self.results = results

    else:
      self.configs = []
      self.results = []
      self.configs_good = None
      self.configs_bad = None
      self.l = None
      self.g = None

  def add_evals(self, configs, results):
    self.configs = configs
    self.results = results


#TODO: delete?
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

    if not len(self.configs) > self.n_min:
      #TODO: throw exception
      print("Not enough data points")
      return

    top_q_n = math.floor(len(self.results) * self.top_quantile)
    good_n = max(self.n_min, top_q_n)
    bad_n = max(self.n_min, len(self.results) - top_q_n)
    loss_sorted = np.argsort(self.results)

    ix_good = loss_sorted[:good_n]
    ix_bad = loss_sorted[-bad_n:]
    configs_arr = np.array(list([config.get_array() for config in self.configs]))
    self.configs_good = configs_arr[ix_good]
    self.configs_bad = configs_arr[ix_bad]

    self.l = sm.nonparametric.KDEMultivariate(data=self.configs_good, var_type=self.data_types)
    self.g = sm.nonparametric.KDEMultivariate(data=self.configs_bad, var_type=self.data_types)
    print("models fit")


  def reset_configs(self):

    self.configs = []
    self.results = []


  def sample(self):
    if not self.l:
      return self.configspace.sample_configuration()

    else:
      # Randomly select n data points from the original data to sample from l
      sampled_indices = np.random.choice(len(self.configs_good), size=self.sample_n, replace=True)
      sampled_data = self.configs_good[sampled_indices]

      # adding gaussian noise for continous values
      cov = np.diag((self.l.bw ** 2) * self.bw_factor)
      mean = np.zeros(len(cov))
      noise = np.random.multivariate_normal(mean=mean, cov=cov, size=self.sample_n)

      c_vals = [dt == "c" for dt in self.data_types]
      sampled_data[:, c_vals] = sampled_data[:, c_vals] + noise[:, c_vals]

      # categorical values
      for i in range(len(self.data_types)):
        if self.data_types[i] == "u":
          #TODO: get number of variables from configspace
          n_categories = len(np.unique(sampled_data[:, i]))
          dist = (np.ones(n_categories) * self.l.bw[i]) / n_categories - 1
          dist[0] = 1 - self.l.bw[i]
          noise = np.random.multinomial(1, dist, size = n_categories)
          sampled_data[:, i] == np.mod(sampled_data[:, i] + noise, n_categories)


      densities = list(self.l.pdf(d) / self.g.pdf(d) for d in sampled_data)
      index_max = np.argmax(densities)
      opt_config = sampled_data[index_max]

      # rescale if out of bounds of configspace
      c = 0
      for t in self.data_types:
        if t == "c":
          opt_config[c] = 0 if opt_config[c] < 0 else opt_config[c]
          opt_config[c] = 1 if opt_config[c] > 1 else opt_config[c]
        c += 1

      candidate = Configuration(self.configspace, vector=opt_config)
      return(candidate)