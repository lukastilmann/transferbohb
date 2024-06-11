import torch

class HyperparamEvaluation:

  def __init__(self, dataloader_train, dataloader_val, model, criterion, optimizer):
    self.budget_spent = 0
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.epochs = 0

  def evaluate(self, budget):

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    last_time = time.time()
    self.model.train()

    while (self.budget_spent < budget):
      running_loss = 0.0
      for batch in self.dataloader_train:
        inputs, labels = batch['image'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()

        new_time = time.time()
        self.budget_spent = self.budget_spent + (new_time - last_time)
        last_time = new_time

        if self.budget_spent > budget:
          break

      self.epochs += 1

    print(f"Budget ({budget} seconds) surpassed after {self.epochs} epochs - Loss: {running_loss / len(self.dataloader_train):.4f}")

    #calculating validation loss
    total_loss = 0.0
    total_samples = 0
    self.model.eval()

    with torch.no_grad():
      for data in self.dataloader_val:
        inputs, labels = batch['image'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        total_loss += loss.item()

    val_loss = total_loss / len(self.dataloader_val)
    print(f"Validation Loss: {val_loss:.4f}")

    return val_loss