import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt %matplotlib inline
from numpy.random import seed
import tensorflow as tf
import joblib
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose

#EDA
tf.random.set_seed(10)
#dataset with Anomalies
df=pd.read_csv("")
col_labels = ['x', 'y', 'z']
df.columns = col_labels
x=df["x"]
y=df["y"]
z=df["z"]
#dataset with normal data
df1 = pd.read_csv("")
col_labels = ['x1', 'y1', 'z1']
df1.columns = col_labels
x1=df1["x1"]
y1=df1["y1"]
z1=df1["z1"]

#plots
x_axisx = np.arange(len(x))
x_axisy = np.arange(len(y))
x_axisz = np.arange(len(z))


plt.figure(figsize=(12, 4))
# x's plot
plt.subplot(1, 3, 1)
plt.plot(x_axisx, x, marker='o', linestyle='-', color='b')
plt.title('Grafico di x')
plt.xlabel('Indice')
plt.ylabel('Valore di x')
# y's plot
plt.subplot(1, 3, 2)
plt.plot(x_axisy, y, marker='o', linestyle='-', color='r')
plt.title('Grafico di y')
plt.xlabel('Indice')
plt.ylabel('Valore di y')
# z's plot
plt.subplot(1, 3, 3)
plt.plot(x_axisz, z, marker='o', linestyle='-', color='g')
plt.title('Grafico di z')
plt.xlabel('Indice')
plt.ylabel('Valore di z')
#
plt.tight_layout()
plt.savefig('anomali.png')
plt.show()

x_axis1x = np.arange(len(x1))
x_axis1y = np.arange(len(y1))
x_axis1z = np.arange(len(z1))
# 
plt.figure(figsize=(12, 4))
# x1's plot
plt.subplot(1, 3, 1)
plt.plot(x_axis1x , x1, marker='o', linestyle='-', color='b')
plt.title('Grafico di x1')
plt.xlabel('Indice')
plt.ylabel('Valore di x1')
# y1's plot
plt.subplot(1, 3, 2)
plt.plot(x_axis1y , y1, marker='o', linestyle='-', color='r')
plt.title('Grafico di y1')
plt.xlabel('Indice')
plt.ylabel('Valore di y1')
# z1's plot
plt.subplot(1, 3, 3)
plt.plot(x_axis1z , z1, marker='o', linestyle='-', color='g')
plt.title('Grafico di z1')
plt.xlabel('Indice')
plt.ylabel('Valore di z1')
#
plt.tight_layout()
plt.savefig('sani.png')
plt.show()
# Test and Train set
start_index = int(0.59* 10**7)
end_index=int(0.90*10**7)
train = df1.iloc[start_index:end_index, df1.columns.get_loc('z1')].copy().to_frame()
train=train.reset_index(drop=True)
start_index1 = int(0.98 * 10**7)
test = df1.iloc[start_index1:, df1.columns.get_loc('z1')].copy().to_frame()
end_index1= int(531369)
test1 = z.iloc[:end_index1].copy().to_frame()
test1.rename(columns={'z': 'z1'}, inplace=True)
test = pd.concat([test, test1])
test=test.reset_index(drop=True)
print(train.shape, test.shape)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(np.arange(len(test)), test, marker='o', linestyle='-', color='b')
plt.title("test's plot")
plt.xlabel('Index')
plt.ylabel("test's values")
plt.savefig('test.png')
plt.show()
#
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(np.arange(len(train)), train, marker='o', linestyle='-', color='r')
plt.title("train's plot")
plt.xlabel('Index')
plt.ylabel('train values')
plt.savefig('training.png')
plt.show()
# Kalman
n_samples = len(train)  # Assume that train is the time series of measurement data
measurements = train.values  # Convert measurement data into an array 

# Kalman filter parameters
A = 1.0  # State transition matrix
H = 1.0  # Observation matrix
Q = 0.01  # Process noise covariance
R = 0.1  # Measurement noise covariance

# Initialization
x_hat = measurements[0]  # Initial state estimate
P = 1.0  # Initial error covariance

# Kalman filtering
filtered_state = []
for j in measurements:
    # Prediction
    x_hat_minus = A * x_hat
    P_minus = A * P * A + Q

    # Update
    K = P_minus * H / (H * P_minus * H + R)
    x_hat = x_hat_minus + K * (j - H * x_hat_minus)
    P = (1 - K * H) * P_minus

    filtered_state.append(x_hat)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Original data', marker='o')
plt.plot(train.index, filtered_state, label='Filtered estimate', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Kalman Filter - Training Data')
plt.legend()
plt.grid(True)
plt.show()
####

n_samples = len(test)  
measurements = test.values  

# Kalman filter parameters
A = 1.0  # State transition matrix
H = 1.0  # Observation matrix
Q = 0.01  # Process noise covariance
R = 0.1  # Measurement noise covariance

# Initialization
x_hat = measurements[0]  # Initial state estimate
P = 1.0  # Initial error covariance

# Kalman filtering
filtered_state1 = []
for i in measurements:
    # Prediction
    x_hat_minus = A * x_hat
    P_minus = A * P * A + Q

    # Update
    K = P_minus * H / (H * P_minus * H + R)
    x_hat = x_hat_minus + K * (i - H * x_hat_minus)
    P = (1 - K * H) * P_minus

    filtered_state1.append(x_hat)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Original data', marker='o')
plt.plot(test.index, filtered_state1, label='Filtered estimate', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Kalman Filter - Test Data')
plt.legend()
plt.grid(True)
plt.show()

# Seasonality extraction

Normal=pd.read_csv("filtered_state.csv")
Anomal=pd.read_csv("filtered_state1.csv")
Normal = Normal[:531369]

Normal.index = pd.to_datetime(Normal.index)

# 
rolling_mean = Normal.rolling(window='1H').mean()

# 
rolling_mean.dropna(inplace=True)

# 
decomposition = seasonal_decompose(rolling_mean, model='additive', period=60*60)  

# 
plt.figure(figsize=(12, 8))



# 
plt.subplot(412)
plt.plot(rolling_mean)
plt.title('Media Mobile')

# 
plt.subplot(413)
plt.plot(decomposition.trend)
plt.title('Tendenza')

# 
plt.subplot(414)
plt.plot(decomposition.seasonal)
plt.title('Stagionalità')

plt.tight_layout()
plt.show()


Anomal.set_index("Unnamed: 0", inplace=True)  # Set the "Unnamed: 0" column as the index
Anomal.index = pd.to_datetime(Anomal.index)  # Convert the index to datetime format

# Rolling mean with a 1-hour window
rolling_mean1 = Anomal.rolling(window='1H').mean()

# Drop NaN values
rolling_mean1.dropna(inplace=True)

# Seasonal decomposition using additive model with a period of 3600 seconds (1 hour)
decomposition1 = seasonal_decompose(rolling_mean1, model='additive', period=60*60)  # 60*60 seconds corresponds to one hour

# Create a plot with a specific figure size
plt.figure(figsize=(12, 8))

# Plot rolling mean
plt.subplot(412)
plt.plot(rolling_mean)
plt.title('Rolling Mean')

# Plot trend component
plt.subplot(413)
plt.plot(decomposition.trend)
plt.title('Trend')

# Plot seasonal component
plt.subplot(414)
plt.plot(decomposition1.seasonal)
plt.title('Seasonality')

plt.tight_layout()  # Adjust layout to avoid overlap

plt.show()  # Display the plot

seasonal_component = decomposition.seasonal  # Extract the seasonal component from the decomposition

# Add the seasonal component as a new column to the original DataFrame
Normal['Seasonality'] = seasonal_component  # 'Normal' is the dataset with normal data

seasonal_component1 = decomposition1.seasonal  # Extract the seasonal component from the second decomposition
Anomal["Seasonality"] = seasonal_component1  # 'Anomalies' is the dataset with anomalous data
Normal = Normal.reset_index(drop=True)  # Reset the index of the 'Normal' DataFrame, dropping the old index
col_labels = ['z', "Seasonality"]  # Define the new column labels
Normal.columns = col_labels  # Assign the new column labels to the 'Normal' DataFrame
Anomal= Anomal.reset_index(drop=True)  # Reset the index of the 'Anomalies' DataFrame, dropping the old index
col_labels = ['z', "Seasonality"]  # Define the new column labels
Anomal.columns = col_labels  # Assign the new column labels to the 'Anomalies' DataFrame

GG = Normal.tail(4000)  # Select the last 4000 observations from 'Normal'

# Remove the last 1000 observations from 'Normal'
Normal = Normal.iloc[:-1000]

# Add the last 1000 observations to the beginning of 'Anomalies'
Anomal = pd.concat([GG, Anomal]).reset_index(drop=True)

train= pd.DataFrame(Normal)
test=pd.DataFrame(Anomal)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_Stagionali"
joblib.dump(scaler, scaler_filename)

#LSTM-Autoencoder
def create_time_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        windows.append(window)
    return windows


window_size = 100


X_train = create_time_windows(X_train, window_size)
X_test= create_time_windows(X_test, window_size)

X_train = np.array(X_train)
X_test = np.array(X_test)

# reshape inputs for LSTM [samples, timesteps, features] 
X_train = X_train.reshape(X_train.shape[0], 100, X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 100, X_test.shape[2])

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(8, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(8, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(16, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# create the autoencoder model/creiamo il modello autoencoder
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()

# fit the model to the data/ fittimao il modello ai dati
nb_epochs = 20
batch_size = 50
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.25).history

# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.savefig('Model_loss.png')
plt.show()

# plot the loss distribution of the training set/plot della distribuzione della loss di training
X_pred = model.predict(X_train)
X_pred = X_pred[:len(train)]
X_pred = X_pred.reshape(len(X_pred), -1)
X_pred = pd.DataFrame(X_pred, columns=[f'Feature_{i}' for i in range(X_train.shape[1] * X_train.shape[2])])
X_pred.index = train.index[:len(X_pred)]

scored = pd.DataFrame(index=train.index[:len(X_pred)])
Xtrain = X_train.reshape(X_train.shape[0], -1)
scored['Loss_mae'] = np.mean(np.abs(X_pred.values - Xtrain[:len(X_pred)]), axis=1)

plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
plt.xlim([0.0, 0.5])

max_loss_mae = scored['Loss_mae'].max()
# calculate the loss on the test set
X_pred = model.predict(X_test)

X_pred = X_pred.reshape(X_test.shape[0], -1)
X_pred = pd.DataFrame(X_pred, columns=[f'Feature_{i}' for i in range(X_test.shape[1] * X_test.shape[2])])
X_pred.index = test.index[:len(X_pred)]

scored = pd.DataFrame(index=test.index[:len(X_pred)])
Xtest = X_test[:len(X_pred)].reshape(len(X_pred), -1)
scored['Loss_mae'] = np.mean(np.abs(X_pred.values - Xtest), axis=1)
scored['Threshold'] = max_loss_mae
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']


threshold = max_loss_mae
plt.rcParams['agg.path.chunksize'] = 10000
plt.figure(figsize=(16, 9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
scored['Loss_mae'].plot(logy=True, ylim=[1e-2, 1e2], color='blue')
plt.plot(scored[scored['Anomaly']].index, scored[scored['Anomaly']]['Loss_mae'], 'ro')
plt.xlabel('Timestamp')
plt.ylabel('Loss (MAE)')
plt.show()

X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train[:len(train)]
X_pred_train = X_pred_train.reshape(len(X_pred_train), -1)
X_pred_train = pd.DataFrame(X_pred_train, columns=[f'Feature_{i}' for i in range(X_train.shape[1] * X_train.shape[2])])
X_pred_train.index = train.index[:len(X_pred_train)]

scored_train = pd.DataFrame(index=train.index[:len(X_pred_train)])
Xtrain = X_train.reshape(X_train.shape[0], -1)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train.values - Xtrain[:len(X_pred_train)]), axis=1)


scored_train['Threshold'] = max_loss_mae
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
threshold = max_loss_mae
plt.rcParams['agg.path.chunksize'] = 10000
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
plt.plot(scored_train.index, scored_train['Loss_mae'], color='blue',label='Train_Loss')
plt.axhline(y=threshold, color='purple', linestyle='--', label='Threshold')
plt.xlabel('Timestamp')
plt.ylabel('Loss (MAE)')
plt.show()

## Detecting anomalies

scored.index = range(len(scored_train.index), len(scored_train.index) + len(scored))
threshold = max_loss_mae
plt.figure(figsize=(16, 9))
plt.plot(scored_train.index, scored_train['Loss_mae'], color='blue',label='Train_Loss')
plt.plot(scored.index, scored['Loss_mae'], color='red', label='Test Loss')
plt.scatter(scored[scored['Anomaly'] == True].index,
scored[scored['Anomaly'] == True]['Loss_mae'],
color='green', label='Anomaly (Positive)')
plt.axhline(y=threshold, color='purple', linestyle='--', label='Threshold')
plt.yscale('log')
plt.ylim([1e-2, 1e2])
plt.xlabel('Index')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.title('Train vs Test Loss with Anomalies')
plt.savefig('Train_vs_Test_loss.png')
plt.show()
