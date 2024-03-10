import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor
from torchvision.transforms import Normalize, ToTensor, Compose, RandomResizedCrop, Resize
from torch.utils.data import DataLoader
from torchvision import datasets, models
import numpy as np

from transferbohb import HyperparamEvaluation

class HyperbandRound:

    def __init__(self, configs, budget, data_train, data_val):
        self.configs = configs
        self.budget = budget
        self.performances = []  # np.empty(len(self.configs))
        self.data_train = data_train
        self.data_val = data_val
        self.num_classes = len(self.data_train.unique('label'))

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
        trans = [RandomResizedCrop(size)]
        transformations = Compose(trans + [ToTensor(), normalize])
        dataset.set_transform(HyperbandRound.apply_transforms(transformations))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
        pretrained_model = models.resnet18(pretrained=True)

        # Freeze all layers except the final classification layer
        for param in pretrained_model.parameters():
            param.requires_grad = False
        pretrained_model.fc.requires_grad = True

        # Modify the number of output classes in the final classification layer
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.AdamW(pretrained_model.parameters(), lr=0.001)
        # TODO: learning rate is proxy for all configs here
        optimizer = optim.AdamW(pretrained_model.parameters(), lr=config)

        return pretrained_model, criterion, optimizer

    def run(self):
        for config in self.configs:
            lr = config['lr']
            dataloader_train = self.create_train_dataloader(config, self.data_train)
            dataloader_val = self.create_val_dataloader(self.data_val)
            model, criterion, optimizer = self.create_model(lr, self.num_classes)
            # start jobs with budget
            config_eval = HyperparamEvaluation(self.budget, dataloader_train, dataloader_val, model, criterion,
                                               optimizer)
            perf = config_eval.evaluate()
            self.performances.append(perf)
            # print(perf)

    def get_top_performances(self, eta):
        top_n = len(self.performances) * (1 / eta)
        top_indices = np.argsort(self.performances)[:top_n]
