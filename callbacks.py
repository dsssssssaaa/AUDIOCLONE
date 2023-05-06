import tensorflow as tf 


# Create the callbacks 

def create_callbacks(config): 

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( 

filepath=config['checkpoint_dir'] + 'checkpoint_{epoch:02d}.h5', 

save_freq=config['checkpoint_rate'] * len(train_data), 

save_weights_only=True, 

verbose=1 

) 

callbacks = [checkpoint_callback] 

return callbacks 

