## Experiment 1: Basic LSTM
Objective: 
    To predict daily stock price percentage gains using a basic Long Short-Term Memory (LSTM) model.
    
    Model Architecture

    LSTM Layer:
        Input size: Number of features per stock (Pct_Gain, RSI, OBV_Ratio, 40-Day Moving Average Ratio, 5-Day Moving Average Ratio, ROC, Close).
        Hidden size: 50 neurons.
        Number of layers: 2 stacked LSTM layers.
    Fully Connected Layer:
        Maps the final hidden state from the LSTM to a single scalar prediction (percentage gain).

Features

    The dataset includes the following engineered features:
        Percentage Gain (Pct_Gain)
        Relative Strength Index (RSI)
        On-Balance Volume Ratio (OBV_Ratio)
        40-Day and 5-Day Moving Average Ratios
        Rate of Change (ROC)

Training Details

    Training Period: February 2020 to December 2023.
    Testing Period: January 2024 onwards.
    Sequence Length: 30 days of historical data used for each prediction.
    Batch Size: 64.
    Optimizer: Adam with a learning rate of 0.00001.
    Loss Function: Mean Squared Error (MSE).
    Epochs: 10.

Results
    Epoch	Train Loss (MSE)	Test Loss (MSE)
        1	0.000932	        0.000595
        2	0.000922	        0.000596
        3	0.000921	        0.000597
        4	0.000920	        0.000600
        5	0.000920	        0.000596
        6	0.000920	        0.000596
        7	0.000919	        0.000596
        8	0.000919	        0.000597
        9	0.000919	        0.000598
        10	0.000919	        0.000596
Analysis

    Training Stability: The training loss stabilized after the first few epochs.
    Test Performance: The test loss remained consistent, indicating no significant overfitting or underfitting.
    Baseline Comparison:
        Test loss is lower than the average variance in stock price movements (0.400485), suggesting the model performs better than a naive average predictor.

Conclusion

    The basic LSTM model effectively captures historical price trends to predict percentage gains.
    However, the consistent test loss suggests limited improvement over time. This indicates the potential for improvement through more complex architectures or additional features.