# General description and goal of project

The primary goal is to develop a predictive model that analyzes historical stock data and forecasts stock price movements. The project includes:

    Data Collection: Gathering historical stock price and volume data from reliable sources.
    Feature Engineering: Calculating and preparing key technical indicators to capture market trends and momentum.
    Model Training: Training LSTM models to predict the percentage change in stock prices for the next trading day.
    Evaluation: Assessing the model's performance using mean squared error (MSE) and comparing it to baseline models.

The project incorporates the following technical indicators to enhance prediction accuracy:

    Percentage Gain (Pct_Gain): Daily percentage change in the stock's closing price, representing the stock's day-to-day performance.
    On-Balance Volume Ratio (OBV_Ratio): Measures the ratio of today's On-Balance Volume (OBV) to yesterday's OBV, combining price and volume movements to assess buying/selling pressure.
    40-Day and 5-Day Moving Average Ratios: Compares the stock's current price to its moving averages over short-term (5-day) and long-term (40-day) periods, capturing both immediate and sustained trends.
    Relative Strength Index (RSI): A momentum oscillator that measures the speed and change of price movements, helping identify overbought or oversold conditions.
    Rate of Change (ROC): Calculates the percentage change in price over a specified time frame, quantifying the velocity of price movements.
  

# Summary and Results Across All Experiments

    Experiment 1: Basic LSTM

    Objective: Predict stock price percentage gains using a standard LSTM architecture.
    Model Highlights:
        Hidden Size: 50 neurons.        
        2 Stacked LSTM Layers.
        Fully Connected Layers for Prediction.
    Results:
        Train Loss (MSE): ~0.000919 (final epoch).
        Test Loss (MSE): ~0.000596.
    Conclusion: The basic LSTM captures meaningful patterns in the dataset with consistent losses across epochs, demonstrating sufficient modeling capacity for this task.

Experiment 2: Slightly Modified LSTM

    Objective: Introduce layer normalization and dropout to stabilize training and reduce overfitting.
    Model Highlights:
        Layer Normalization applied to LSTM outputs.
        Dropout probability of 0.2 between LSTM layers and in fully connected layers.
    Results:
        Train Loss (MSE): ~0.000918 (final epoch).
        Test Loss (MSE): ~0.000595.
    Comparison to Basic LSTM: The addition of layer normalization and dropout slightly stabilizes training but does not significantly improve test loss, indicating the dataset is already well-prepared for the task.
    Conclusion: While the modifications improve training stability, the performance remains similar to the basic LSTM.

Experiment 3: Multi-Input LSTM with Attention Mechanism

    Objective: Process multiple feature groups independently through separate LSTMs and dynamically combine their outputs using attention.
    Model Highlights:
        Separate LSTMs for feature groups.
        Attention mechanism to dynamically weight feature group contributions.
    Results:
        Train Loss (MSE): ~0.509914 (final epoch).
        Test Loss (MSE): ~0.000586.

        Attention scores: 
        
        Mainstream Features	                ~0.00083
        Positive Trend Indicators	        ~0.00012
        Negative Trend Indicators	        ~0.0049
        Index-Based Indicators	            ~0.99

    Comparison to Basic LSTM: The attention mechanism improves the flexibility and interpretability of feature weighting, achieving slightly lower test loss (~0.000586 vs. ~0.000596). However, the higher train loss compared to test loss raises concerns about the model's ability to effectively learn from the training data. Also, the attention weights undermine the expected adaptability of the mechanism.

Experiment 4: LSTM with Dot-Product Attention Mechanism

    Objective: Enhance feature focus and sequence weighting by incorporating a dot-product attention mechanism.
    Model Highlights:
        Dot-product attention to compute scores and weight sequence outputs.
        Context vector derived from LSTM outputs for prediction.
    Results:
        Train Loss (MSE): ~0.000924 (final epoch).
        Test Loss (MSE): ~0.000595.
    Comparison to Basic LSTM: The attention mechanism achieves similar performance to the basic LSTM.
    Conclusion: The model provides benefits without a substantial improvement in performance over simpler architectures.



Overall Insights and Comparisons
Metric	         Basic LSTM	   Modified LSTM  	Multi-Input LSTM with Attention	  LSTM with Dot-Product Attention
Train Loss (MSE)  	~0.000919	      ~0.000918	          ~0.509914	                       ~0.000924
Test Loss (MSE)	    ~0.000596	      ~0.000595	          ~0.000586	                       ~0.000595

Key Findings

    Baseline Performance: The basic LSTM achieves competitive results with minimal architectural complexity, demonstrating strong generalization capabilities.

    Layer Normalization and Dropout: These enhancements improve stability but do not significantly impact performance due to the well-structured dataset.

    Attention Mechanisms: Both dot-product and multi-input attention mechanisms improve flexibility in feature weighting. The multi-input attention model performs slightly better, suggesting its architecture is better suited for datasets with diverse feature groups.

    General Observations: The test loss is relatively stable across all models, reflecting the dataset’s suitability for modeling. Improvements in architecture lead to interpretability gains but minimal performance differences.

Conclusion

All experiments demonstrate robust performance with test losses consistently below 0.0006, indicating the dataset is well-structured and easy to model. While advanced architectures (e.g., attention mechanisms) provide interpretability and flexibility benefits, they do not yield substantial performance improvements over the basic LSTM. 