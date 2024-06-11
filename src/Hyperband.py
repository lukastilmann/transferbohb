from datasets import load_dataset
import numpy as np
from transferbohb import TPE, HyperbandRound

class Hyperband:

  def __init__(self, config_space, min_budget, max_budget, eta, p, dataset_name):
    self.config_space = config_space
    self.min_budget = min_budget
    self.max_budget = max_budget
    self.eta = eta
    self.p = p
    #self.initial_configs = config_space.sample_configuration(n_configs)
    self.data_train = load_dataset(dataset_name, split='train')
    self.data_val = load_dataset(dataset_name, split='validation[:1000]')

    self.s_max = int(np.emath.logn(self.eta, self.max_budget/min_budget))


  def run(self):

    tpe = TPE(self.config_space, 0.2, 5, 3)

    for s in range(self.s_max, 0, -1):
      n = int((self.s_max + 1 / s + 1) * (self.eta ** s))
      print("nr of configs: " + str(n))
      b_0 = self.max_budget * (self.eta ** -s)
      print("budget: " + str(b_0))

      tpe = HyperbandRound(n, b_0, self.max_budget, self.p, self.eta,
                           self.data_train, self.data_val, self.config_space,
                           tpe).run()
