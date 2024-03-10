class HyperparamEvaluation:

  def __init__(self, budget, dataloader_train, dataloader_val, model, criterion, optimizer):
    self.budget = budget
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer

  def evaluate(self):

    # Training loop
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)
    start_time = time.time()
    self.model.train()

    for epoch in range(num_epochs):
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

        elapsed_time = time.time() - start_time
        if elapsed_time > self.budget:
          print(f"Budget ({self.budget} seconds) surpassed after {epoch+1} epochs.")
          break

      print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(self.dataloader_train):.4f}")
      elapsed_time = time.time() - start_time
      if elapsed_time > self.budget:
        break

    # calculating validation loss
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