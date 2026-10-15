"""LSTM autoencoder architecture based on the thesis implementation."""


def build_autoencoder(window_size: int, feature_count: int):
    """Import TensorFlow lazily so preprocessing/tests work without GPU tooling."""
    from tensorflow import keras

    inputs = keras.Input(shape=(window_size, feature_count))
    encoded = keras.layers.LSTM(16, activation="tanh", return_sequences=True)(inputs)
    encoded = keras.layers.LSTM(8, activation="tanh")(encoded)
    decoded = keras.layers.RepeatVector(window_size)(encoded)
    decoded = keras.layers.LSTM(8, activation="tanh", return_sequences=True)(decoded)
    decoded = keras.layers.LSTM(16, activation="tanh", return_sequences=True)(decoded)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(feature_count))(decoded)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mae")
    return model
