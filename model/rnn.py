from keras.layers import Dense, LayerNormalization, ReLU, SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

def compiled_rnn(look_back, input_features=5, optimizer_lr=0.001, dropout_rate=0.1, regularization_val=0.01):
    """
    Build and return a RNN model for time-series prediction.
 
    Parameters:
    - look_back: Number of previous time steps to use as input variables.
    - input_features (default=3): Number of features in the input data.
    - optimizer_lr (default=0.005): Learning rate for the optimizer.
    - dropout_rate (default=0.5): Fraction of the input units to drop.
    - regularization_val (default=0.01): Regularization strength for L2 regularization.
 
    Returns:
    - model: Compiled GRU model.
    """
 
    # Initialize a sequential model.
    model = Sequential()
 
    # Add the GRU layer with 64 units. The input shape is based on the look-back period and number of features.
    model.add(SimpleRNN(64, input_shape=(look_back, input_features), return_sequences=True, dropout=dropout_rate,
                        recurrent_dropout=0.2))
    # Add a second GRU layer with 64 units.
    model.add(SimpleRNN(64, return_sequences=False, dropout=dropout_rate, recurrent_dropout=0.2))
 
    # Batch normalization layer normalizes activations of the previous layer.
    model.add(LayerNormalization())
 
    # Activation function layer using ReLU activation.
    model.add(ReLU())
 
    # Dense layer for further processing.
    model.add(Dense(32, activation='relu'))
 
    # Output layer, which outputs a prediction. It uses linear activation (default for regression tasks) and L2 regularization.
    model.add(Dense(2, activation='linear', kernel_regularizer=l2(regularization_val)))
 
    # Compile the model with an Adam optimizer (with a custom learning rate) and mean squared error loss function.
    optimizer = Adam(learning_rate=optimizer_lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', run_eagerly=True)
 
    # Return the compiled model.
    return model