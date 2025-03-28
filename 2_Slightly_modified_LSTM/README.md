## Experiment 2: Slightly Modified LSTM

Objective:
    To predict daily stock price percentage gains using a slightly modified Long Short-Term Memory (LSTM) model with added Droupout mechanism and Layer normalization.

    # Layer Normalization
    https://www.ricercaintelligente.com/wp-content/uploads/2023/10/layer-normalization.pdf
    https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1
    # Dropout
    https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

    Model Architecture

    LSTM Layer:
        Input size: Number of features per stock (Pct_Gain, RSI, OBV_Ratio, 40-Day Moving Average Ratio, 5-Day Moving Average Ratio, ROC, Close).
        Hidden size: 128 neurons.
        Number of layers: 2 stacked LSTM layers.
        Dropout applied between LSTM layers to prevent overfitting.
    Layer Normalization:
        Applied to the LSTM outputs to stabilize training and normalize activations.
    Fully Connected Layers:
        FC1: Maps LSTM outputs to a reduced feature space.
        ReLU: Activation function for non-linearity.
        Dropout: Regularization to prevent overfitting.
        FC2: Maps to a single scalar output (predicted percentage gain).

Feature Engineering

The model uses a set of engineered features derived from historical stock data:

    Percentage Gain (Pct_Gain): Daily percentage change in the closing price.
    On-Balance Volume Ratio (OBV_Ratio): Ratio of today's OBV to yesterday's OBV.
    40-Day and 5-Day Moving Average Ratios: Ratios of moving averages to the current price.
    Relative Strength Index (RSI): A momentum oscillator measuring speed and change of price movements.
    Rate of Change (ROC): Percentage change between the current price and the price n periods ago.

Training Details

    Data:
        Training Period: February 2020 to December 2023.
        Testing Period: January 2024 onwards.
        Sequence Length: 30 days of historical data used for each prediction.
        Batch Size: 64.
    Hyperparameters:
        Learning Rate: 0.001.
        Hidden Size: 128.
        Dropout Probability: 0.2.
    Optimizer: Adam.
    Loss Function: Mean Squared Error (MSE).
    Epochs: 10.

Results
Epoch	Train Loss (MSE)	Test Loss (MSE)
    1	0.001335	        0.000595
    2	0.000922	        0.000595
    3	0.000921	        0.000595
    4	0.000921	        0.000595
    5	0.000920	        0.000595
    6	0.000920	        0.000595
    7	0.000919	        0.000595
    8	0.000919	        0.000595
    9	0.000919	        0.000596
    10	0.000918	        0.000595
Analysis

    Stability: The modified model achieved consistent test losses across all epochs, demonstrating robust generalization.
    Comparison to Baseline:
        Test loss is significantly lower than the variance of the target variable (0.400485), indicating the model captures meaningful trends.
    Comparison to LSTM:
        The slightly modified LSTM model achieves comparable test losses to the default LSTM, with both models stabilizing at approximately 0.000595 from the early epochs. However, the modified model demonstrates slightly faster convergence in training loss compared to the default LSTM, suggesting that the addition of dropout and layer normalization aids in stabilizing training without significantly altering final performance.    
    Improvements Over Basic LSTM:
        The dropout layers and layer normalization helped stabilize training and reduce overfitting.
        The modified architecture achieves slightly faster convergence compared to the basic LSTM.

Conclusion

The basic LSTM and modified LSTM both demonstrate sufficient capacity to model the relationships within the dataset. The additional dropout and layer normalization did not lead to significant performance improvements. This could be attributed to the dataset already being well-transformed, consisting of relative data that is less noisy and easier to model.