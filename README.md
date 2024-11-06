# Anomaly-Detector
Anomaly detector using LSTM-Autoencoder for detecting anomalies in conveyor belt vibration sensors
## Anomaly Detection Approach

In practice, anomaly detection can be implemented using two different techniques. The first technique involves generating real-time alerts based on the monitored system. The second technique records events for subsequent analysis. The first approach is somewhat risky, as anomalies do not necessarily indicate a problem within the system. This could lead to the generation of "noise" and increase the operational costs. The second technique, while "risk-averse," does not allow for timely intervention in case of system issues.

For these reasons, this study adopts a hybrid approach, with the following logic: an autoencoder is a type of neural network that learns to compress input data into a lower-dimensional representation and then reconstructs the original input from the compressed data. If the input data are "normal," the autoencoder will be able to reconstruct them with a low reconstruction error. Conversely, if the data are "anomalous," the reconstruction error will be significantly higher. This logic was used to detect anomalies. After training, the reconstruction loss (MAE) is calculated on both the training and test data, and using the probability distribution of the loss function on the training data, a threshold was established. If this threshold is exceeded, it indicates the presence of an anomaly.
## Dataset

The original datasets consisted of three features representing the three axial directions of the VTV-122 vibration sensor. The dataset containing non-anomalous values had three columns and 9,632,614 rows, with each value recorded at a sampling frequency of 400 points per second. The dataset containing anomalous data also had three columns but consisted of 531,370 rows, with the same sampling frequency of 400 points per second.

For this study, we chose to keep only the feature representing the z-axis, as it was the axis most affected by vibrations.

The non-anomalous data contain peaks, which are due to sensor setup errors. For this reason, these observations were excluded from the dataset, reducing the number of rows to 3,100,000. 

After this, the construction of the training and test sets proceeded as follows: The training set was formed from the original observations in the non-anomalous dataset, with the last one thousand observations replaced by the first one thousand from the anomalous dataset. The test set was composed of observations from the anomalous dataset, with the first one thousand replaced by the last one thousand observations from the non-anomalous series.
## Kalman's filter
As mentioned earlier, the data comprising the dataset were collected from vibration sensors, which are subject to a certain degree of measurement error (or noise). This is where the Kalman Filter comes into play, as it helps to provide cleaner and more accurate training and test data for training and evaluating the network.
## Seasonality Extraction

A time series can be decomposed into its constituent components using a statistical technique called Seasonal Decomposition. These components are:

Trend: This refers to long-term changes in the series observed over time, representing the general direction in which the series is moving.
Seasonality: This refers to the cyclical patterns observed within a predetermined time period.
Residuals: This represents the random variation that remains after accounting for the trend and seasonality components. It captures the noise or error in the data that cannot be explained by trend or seasonality.
In general, seasonal decomposition is used to gain insights into the hidden dynamics within time series data.

In our study, after filtering the data with the Kalman Filter, we applied seasonal decomposition using the seasonal_decompose function from the statsmodels.tsa.seasonal module in the Python library statsmodels, which allowed us to decompose the series into the components mentioned above.

Of the three main components, we decided to include only the seasonality in the dataset, as the trends in both the anomalous and normal time series were similar. To identify anomalies, our focus was on distinguishing characteristics between the two series, which also allowed us to reduce the dataset’s size and achieve faster computation times.
## Data Normalization

Finally, before being fed into the network, the data are normalized using the `"sklearn.preprocessing"` module from the `scikit-learn` library in Python, with the `MinMaxScaler` class. This class scales each feature/independent variable in the data so that they fall within a specified range between 0 and 1.

After completing this step, the training and test datasets are finally ready to be used for training and evaluating the network.






