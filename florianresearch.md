How to get the uncertainty:

- Use Bayesian Inference on the LSTM:

  - https://github.com/PawaritL/BayesianLSTM/blob/master/Energy_Consumption_Predictions_with_Bayesian_LSTMs_in_PyTorch.ipynb
  - https://arxiv.org/pdf/1709.01907
  - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9924783
- Use Monte Carlo Dropout

  - Apply dropout during training and inference
  - Predict multiple times for the same input to obtain a distributin of predictions
