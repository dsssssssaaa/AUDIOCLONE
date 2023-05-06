
import argparse 

import tensorflow as tf 

import json 

from data_loader import load_audio_files 

from model import create_model 

from callbacks import create_callbacks 


# Define the arguments 

parser = argparse.ArgumentParser() 

parser.add_argument('--config', type=str, default='config.json', help='Path to the config file') 

args = parser.parse_args() 


# Load the config file 
with open(args.config, 'r') as f: 
    config = json.load(f) 

# Load training data 
train_data = load_audio_files(config['zip_file_path']) 

# Create the model 
model = create_model(config) 

# Compile the model 
model.compile(optimizer=config['optimizer'], loss=config['loss']) 

# Create the callbacks 
callbacks = create_callbacks(config) 

# Train the model 
model.fit(train_data, train_data, batch_size=config['batch_size'], epochs=config['epochs'], callbacks=callbacks) 

# Save the trained model 
model.save(config['model_path'])


