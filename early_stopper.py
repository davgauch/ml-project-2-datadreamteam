class EarlyStopper:
    """
    A class to implement early stopping during model training.

    Early stopping halts training when the improvement in validation loss falls below a specified threshold 

    Attributes:
        patience (int): Number of epochs with insufficient improvement before stopping.
        min_decrease (float): Minimum percentage decrease in validation loss considered as improvement.
        counter (int): Counts epochs without sufficient improvement.
        min_validation_loss (float): Tracks the lowest validation loss observed.
    """

    def __init__(self, patience=1, min_decrease=0):
        self.patience = patience
        self.min_decrease = min_decrease
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            if (self.min_validation_loss - validation_loss) / self.min_validation_loss < self.min_decrease:
                self.counter += 1
            else:
                self.counter = 0
            self.min_validation_loss = validation_loss
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False