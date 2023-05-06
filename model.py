import tensorflow as tf 


# Create the neural network architecture 

def create_model(config): 

model = tf.keras.models.Sequential([ 

tf.keras.layers.Input(shape=(None, 14)), 

tf.keras.layers.LSTM(units=config['lstm_units'], return_sequences=True), 

tf.keras.layers.Dense(units=14) 

]) 

return model 

