## Experiment 4: LSTM with dot-product attention mechanism
Objective:

To predict daily stock price percentage gains using an enhanced Long Short-Term Memory (LSTM) model with an integrated attention mechanism to improve feature weighting and sequence focus.

    # Layer Normalization
    https://www.ricercaintelligente.com/wp-content/uploads/2023/10/layer-normalization.pdf
    https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1
    # Dropout
    https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer
    # Dot-product attention mechanism
    https://link.springer.com/article/10.1007/s10015-020-00638-y 

Model Architecture:

    LSTM Layer:
        Input size: Number of features per stock (Pct_Gain, RSI, OBV_Ratio, 40-Day Moving Average Ratio, 5-Day Moving Average Ratio, ROC, Close).
        Hidden size: 128 neurons.
        Number of layers: 2 stacked LSTM layers.
        Dropout applied to prevent overfitting.

    Attention Mechanism:
        Utilizes the final hidden state as the query vector.
        Attention scores calculated as the dot product of the query vector and LSTM outputs.
        Scores normalized via softmax to obtain attention weights.
        Context vector derived as the weighted sum of LSTM outputs.

    Fully Connected Layers:
        FC1: Maps the context vector to a reduced feature space.
        ReLU: Activation function for introducing non-linearity.
        Dropout: Regularization to prevent overfitting.
        FC2: Maps to a single scalar output (predicted percentage gain).

Feature Engineering:

The model utilizes engineered features derived from historical stock data:

    Percentage Gain (Pct_Gain): Daily percentage change in the closing price.
    On-Balance Volume Ratio (OBV_Ratio): Ratio of today's OBV to yesterday's OBV.
    40-Day and 5-Day Moving Average Ratios: Ratios of moving averages to the current price.
    Relative Strength Index (RSI): Measures the momentum and speed of price movements.
    Rate of Change (ROC): Percentage change between the current price and the price n periods ago.

Training Details:

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

Results:
Epoch	Train Loss (MSE)	Test Loss (MSE)
    1	    0.000931	        0.000597
    2	    0.000925	        0.000598
    3	    0.000923	        0.000596
    4	    0.000923	        0.000595
    5	    0.000924	        0.000597
    6	    0.000923	        0.000595
    7	    0.000925	        0.000598
    8	    0.000924	        0.000597
    9	    0.000924	        0.000602
    10	    0.000924	        0.000596
Analysis:

    The model reaches convergence early (by epoch 4), with minimal change in loss values thereafter, indicating that further training epochs provide diminishing returns.
    The fluctuations in test loss are negligible and well within an acceptable range, suggesting the model is robust against data variability.   

Comparison to Baseline:
    Test loss is significantly lower than the variance of the target variable (0.400485), indicating the model captures meaningful trends.

Comparison to Basic LSTM:
    The LSTM with attention mechanism achieves comparable performance to the basic LSTM model, as reflected in the train and test losses. Both models converge to similar test loss values (~0.000595 to ~0.000602) by the final epochs, suggesting strong generalization capabilities in both architectures.
        
Performance:
    The addition of the attention mechanism enhances the model's ability to weigh sequence components, but does not drastically alter performance compared to basic LSTM models with sufficient capacity.

Conclusion:

The LSTM with attention mechanism shows robust performance and strong generalization ability. While the attention mechanism provides improved feature weighting, the minimal difference in performance compared to simpler LSTM models suggests the dataset's inherent structure and preprocessing make it less noisy and easier to model. 