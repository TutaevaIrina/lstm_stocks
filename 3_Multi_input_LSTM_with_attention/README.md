## Experiment 3: Multi-Input LSTM with attention mechanism

Objective:
To predict stock price percentage gains for the next day by leveraging a multi-input LSTM architecture. This model processes multiple feature groups independently, combines their outputs through an attention mechanism, and generates predictions.

    # Layer Normalization
    https://www.ricercaintelligente.com/wp-content/uploads/2023/10/layer-normalization.pdf
    https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1
    # Dropout
    https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer
    # Multi-input LSTM using Attention-base
    https://proceedings.mlr.press/v95/li18c/li18c.pdf

Model Architecture

    Multi-Input LSTMs:
        Four independent LSTM networks process different input groups:
            Mainstream Features: Pct_Gain
            Positive Trend Indicators: 40D_MA_to_Price_Ratio, 5D_MA_to_Price_Ratio
            Negative Trend Indicator: OBV_Ratio
            Index-Based Indicators: RSI, ROC, Close
        Each LSTM has:
            Hidden Size: 128 neurons.
            Number of Layers: 2 stacked layers.
            Dropout: Applied between layers to prevent overfitting.

    Attention Mechanism:
        The outputs from the four LSTMs are concatenated.
        Attention weights are calculated for each group using a fully connected layer.
        Outputs are dynamically combined based on attention scores.

    Layer Normalization:
        Normalizes the combined LSTM outputs to stabilize training and enhance learning efficiency.

    Fully Connected Layers:
        FC1: Maps the combined features to a reduced feature space.
        ReLU: Applies non-linearity.
        Dropout: Prevents overfitting.
        FC2: Produces the final prediction (percentage gain).

Feature Engineering

The following features are derived from historical stock data and grouped into input streams:

    Mainstream Features:
        Pct_Gain: Daily percentage change in the closing price.
    Positive Trend Indicators:
        40D_MA_to_Price_Ratio: Ratio of the 40-day moving average to the current price.
        5D_MA_to_Price_Ratio: Ratio of the 5-day moving average to the current price.
    Negative Trend Indicator:
        OBV_Ratio: Ratio of today's On-Balance Volume (OBV) to yesterday's OBV.
    Index-Based Indicators:
        RSI: Relative Strength Index (momentum oscillator).
        ROC: Rate of Change (percentage change over 14 days).
        Close: Closing price.

Training Details

    Data:
        Training Period: February 2020 to December 2023.
        Testing Period: January 2024 onwards.
        Sequence Length: 30 days of historical data used per prediction.
        Batch Size: 64.

    Hyperparameters:
        Hidden Size: 128.
        Dropout Probability: 0.2.
        Learning Rate: 0.001.

    Loss Function: Mean Squared Error (MSE).

    Optimizer: Adam.

Results

The model was trained for 10 epochs, and both training and test losses were monitored. The final results are summarized below:
Epoch	Train Loss (MSE)	Test Loss (MSE)
    1	0.509937	        0.000721
    2	0.509915	        0.000586
    3	0.509912	        0.000587
    4	0.509912	        0.000587
    5	0.509913	        0.000593
    6	0.509913	        0.000586
    7	0.509915	        0.000586
    8	0.509913	        0.000589
    9	0.509916        	0.000586
    10	0.509914	        0.000587

Attention Score Distribution

After training, the attention scores across all epochs indicate that the model prioritizes the "Index-Based Indicators" group (RSI, ROC, Close) almost exclusively. The other groups receive consistently low attention weights and the attention scores are stable across all training epochs.
Feature Group	            Average Attention Score
Mainstream Features	                ~0.00083
Positive Trend Indicators	        ~0.00012
Negative Trend Indicators	        ~0.0049
Index-Based Indicators	            ~0.99


Analysis

    Performance:
        The model achieved stable and consistent test losses across epochs, demonstrating its ability to generalize well to unseen data.
    Attention Mechanism:
        By weighting the contribution of different feature groups dynamically, the attention mechanism allows the model to focus on relevant indicators for each prediction.
    Comparison to Baseline:
        The model's test loss (MSE = 0.000595) is significantly lower than the variance of the target variable (0.400485), indicating that it effectively captures meaningful patterns in the data.
    Comparison to LSTM:
        The Multi-Input LSTM with attention mechanism achieves a comparable test loss (~0.000586) to the default LSTM (~0.000596) but provides a more nuanced approach to feature utilization. By processing feature groups separately and dynamically weighting their contributions through attention, the model adapts to different patterns in the data more effectively.  

Conclusion

The Multi-Input LSTM achieved slightly lower test loss (~0.000586 vs. ~0.000596). However, the higher train loss compared to test loss raises concerns about the model's ability to effectively learn from the training data. The lack of diversity in the attention weights undermines the expected adaptability of the mechanism and suggests that the model might be overly reliant on the "Index-Based Indicators" group, limiting its ability to leverage the potential predictive power of other feature groups.