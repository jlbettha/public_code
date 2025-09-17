import torch
from torch.nn.modules import Module

"""Build Early-stopping class for training PyTorch models.

    -Low patience (1-3 epochs): Training might stop too early, before the model reaches optimal performance
    -High patience (15-20 epochs): You might waste computation time, but have a better chance of finding the best model
    -Moderate patience (5-10 epochs): Usually a good compromise for most projects

    Can also be used as a PyTorch Lightning callback. Trainer(..., callbacks=[early_stopping])
"""


class EarlyStopping:
    def __init__(
        self, patience: int = 5, min_delta: float = 0.0, verbose: bool = False, filename: str = "es_checkpoint.pt"
    ):
        self.filename = filename
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score: float = None
        self.early_stop: bool = False
        self.counter: int = 0
        self.best_model_state: dict = None
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.filename)
        self.val_loss_min = val_loss

    def load_best_model(self, model: Module):
        if self.verbose:
            print("Loading best model state.")
        model.load_state_dict(self.best_model_state)


def main():
    # Example usage of EarlyStopping
    # early_stopping = EarlyStopping(patience=5, verbose=True)
    # num_epochs = 20

    # # Training loop
    # for epoch in range(num_epochs):
    #   model.train()
    #   for x_batch, y_batch in train_loader:
    #       # Training steps...
    #       pass

    #   # Validation step
    #   model.eval()
    #   val_loss = 0
    #   with torch.no_grad():
    #       for x_batch, y_batch in val_loader:
    #           y_pred = model(x_batch)
    #           loss = criterion(y_pred, y_batch)
    #           val_loss += loss.item()

    #   val_loss /= len(val_loader)
    #   print(f'Epoch {epoch}: val_loss = {val_loss:.6f}')

    #   # Check early stopping
    #   early_stopping(val_loss, model)
    #   if early_stopping.early_stop:
    #       print("Early stopping triggered")
    #       break

    # Call early stopping
    # early_stopping(val_loss, model=None)  # Replace `model=None` with your actual model

    # if early_stopping.early_stop:
    #     print(f"Early stopping triggered at epoch {epoch + 1}")
    #     break

    # Load the best model
    # model.load_state_dict(torch.load(<model's file name>))
    pass


if __name__ == "__main__":
    main()
