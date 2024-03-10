from datasets import load_dataset

class Hyperband:

    def __init__(self, config_space, max_budget, eta, n_configs, dataset_name):
        self.config_space = config_space
        self.max_budget = max_budget
        self.eta = eta
        self.initial_configs = config_space.sample_configuration(n_configs)
        self.data_train = load_dataset(dataset_name, split='train')
        self.data_val = load_dataset(dataset_name, split='validation[:10000]')
        print(self.initial_configs)

    def next_round(self):
        # configs = self.config_space.sample_configuration(n)
        configs = self.initial_configs
        round = HyperbandRound(configs, self.max_budget, self.data_train, self.data_val)
        round.run()
