import statsmodels.api as sm
import math
from ConfigSpace.hyperparameters import BetaFloatHyperparameter, CategoricalHyperparameter

class TPE:

  def __init__(self, configspace, top_quantile, sample_n, configs=None, results=None):

    self.configspace = configspace
    self.n_min = len(self.configspace.values()) + 1
    self.top_quantile = top_quantile
    self.sample_n = sample_n
    self.data_types = ""

    for _, p in dict(self.configspace):
      if isinstance(p, BetaFloatHyperparameter):
        self.data_types += "u"
      if isinstance(p, CategoricalHyperparameter):
        self.data_types += "u"
      else:
        "Added hyperparameter with currently unsupported data type. As of now, only continuous and categorical hyperparams are supported."

    if configs and results:
      self.configs = configs
      self.results = results

      if len(self.configs) > self.n_min:
        self.kde_good, self.kde_bad = self.update_models()
      else:
        self.kde_good = None
        self.kde_bad = None
    else:
      self.configs = []
      self.results = []
      self.kde_good = None
      self.kde_bad = None

  def add_result(self, config, result):
    self.configs.append(config)
    self.results.append(result)

    if len(self.configs) > self.n_min:
      self.update_models()

  def update_models(self):
    top_q_n = math.floor(len(self.results) * self.top_quantile)
    good_n = max(self.n_min, top_q_n)
    bad_n = max(self.n_min, len(self.results) - top_q_n)
    loss_sorted = np.argsort(self.results)
    print(loss_sorted)
    ix_good = loss_sorted[:good_n]
    ix_bad = loss_sorted[-bad_n:]
    print(ix_good)
    #TODO: has to be np array
    configs_good = self.configs[ix_good]
    configs_bad = self.configs[ix_bad]

    configs_good_arr = [config.get_array() for config in self.configs_good]
    configs_bad_arr = [config.get_array() for config in self.configs_bad]

    self.l = sm.nonparametric.KDEMultivariate(data=configs_good_arr, var_type=self.data_types)
    self.g = sm.nonparametric.KDEMultivariate(data=configs_bad_arr, var_type=self.data_types)
    print("models fit")

  def sample(self):
    #TODO:multiply with bandwidth factor

    return self.configspace.sample_configuration()