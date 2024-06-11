import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor
from torchvision.transforms.v2 import Normalize, ToTensor, Compose, RandomResizedCrop, Resize, RandomHorizontalFlip
from torch.utils.data import DataLoader
from torchvision import datasets, models
import numpy as np

from transferbohb import HyperparamEvaluation

import numpy as np


class HyperbandRound:

    def __init__(self, n_configs, budget_0, budget_max, p, eta, data_train, data_val, configspace, tpe):
        self.n_configs = n_configs
        self.budget = budget_0
        self.budget_max = budget_max
        self.p = p
        self.eta = eta
        # self.performances = []
        # self.configs = []
        self.data_train = data_train
        self.data_val = data_val
        self.num_classes = len(self.data_train.unique('label'))
        self.configspace = configspace
        self.n_min = len(self.configspace.values()) + 1
        self.tpe = tpe

    @staticmethod
    def apply_transforms(transforms):
        def transform_batch(examples):
            examples['image'] = [transforms(img.convert('RGB')) for img in examples['image']]
            return examples

        return transform_batch

    @staticmethod
    def create_train_dataloader(config, dataset):
        checkpoint = "google/vit-base-patch16-224-in21k"
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        size = (255, 255)

        # TODO: more complicated logic based on configuration
        trans = []
        if "Crop" in config["transforms"]:
            trans.append(RandomResizedCrop(size))
        if "Flip" in config["transforms"]:
            trans.append(RandomHorizontalFlip(p=0.5))
        transformations = Compose(trans + [ToTensor(), normalize])
        dataset.set_transform(HyperbandRound.apply_transforms(transformations))
        dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True)

        return dataloader

    @staticmethod
    def create_val_dataloader(dataset):
        checkpoint = "google/vit-base-patch16-224-in21k"
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        size = (255, 255)

        # TODO: more complicated logic based on configuration
        transformations = Compose([Resize(size), ToTensor(), normalize])
        dataset.set_transform(HyperbandRound.apply_transforms(transformations))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        return dataloader

    @staticmethod
    def create_model(config, num_classes):
        # Load a pre-trained model from torchvision
        if config["model"] == "resnet":
            pretrained_model = models.resnet18(pretrained=True)
        elif config["model"] == "densenet":
            pretrained_model = models.densenet121(pretrained=True)
        else:
            pretrained_model = models.resnet101(pretrained=True)

        # Freeze all layers
        for param in pretrained_model.parameters():
            param.requires_grad = False

        # unfreeze fully connected layer and n layers before if part of hyperparameters
        if "layers_unfreeze" in config.keys():
            num_layers_to_unfreeze = int(config["layers_unfreeze"])
            # Unfreeze the fully connected layer plus n layers before it
            # We'll need to identify the last `num_layers_to_unfreeze` layers
            layers = list(pretrained_model.children())
            # Reverse the list and flatten the nested children
            all_layers = []
            for layer in reversed(layers):
                if isinstance(layer, nn.Sequential):
                    for sublayer in reversed(list(layer.children())):
                        all_layers.append(sublayer)
                else:
                    all_layers.append(layer)

            # Unfreeze the last `num_layers_to_unfreeze` layers
            for layer in all_layers[:num_layers_to_unfreeze]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Modify the number of output classes in the final classification layer and unfreezing fully connected layer
        if config["model"] == "resnet":
            pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
            pretrained_model.fc.requires_grad = True
        elif config["model"] == "densenet":
            pretrained_model.classifier = nn.Linear(pretrained_model.classifier.in_features, num_classes)
            pretrained_model.classifier.requires_grad = True
        else:
            pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
            pretrained_model.fc.requires_grad = True

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(pretrained_model.parameters(), lr=config["lr"])

        return pretrained_model, criterion, optimizer

    def run(self):

        config_evals = []
        configs = []
        # sample initial configurations
        for i in range(self.n_configs):
            if np.random.binomial(1, self.p) == 1:
                config = self.configspace.sample_configuration()
            else:
                config = self.tpe.sample()
            dataloader_val = self.create_val_dataloader(self.data_val)
            model, criterion, optimizer = self.create_model(config, self.num_classes)
            config_eval = HyperparamEvaluation(self.create_train_dataloader(config, self.data_train), dataloader_val,
                                               model, criterion, optimizer)
            config_evals.append(config_eval)
            configs.append(config)

        # successive halving
        performances = []
        while self.budget < self.budget_max:
            for config_eval in config_evals:
                perf = config_eval.evaluate(self.budget)
                performances.append(perf)

            ranked_indices = np.argsort(performances)[:len(config_evals) * (1 / self.eta)]
            new_evals = [config_evals[i] for i in ranked_indices]
            new_configs = [configs[i] for i in ranked_indices]
            self.budget = self.budget * self.eta

            if len(performances > self.n_min):
                self.tpe.add_evals(configs, performances)

            configs = new_configs
            config_evals = new_evals

        self.tpe.update_models()
        return self.tpe

    def get_top_performances(self, eta):
        top_n = len(self.performances) * (1 / eta)
        top_indices = np.argsort(self.performances)[:top_n]
