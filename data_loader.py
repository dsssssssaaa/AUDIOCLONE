import numpy as np 

import librosa 

import crepe 

from pitch import pitch_tuner 

import zipfile 


# Load audio files and extract features 

def load_audio_files(zip_file_path): 

with zipfile.ZipFile(zip_file_path) as zip_file: 

train_files = zip_file.namelist() 

train_data = [] 

for file_path in train_files: 

with zip_file.open(file_path) as audio_file: 

audio, _ = librosa.load(audio_file, sr=None, mono=True) 

train_data.append(audio) 

train_data = extract_features(train_data) 

return train_data 


# Extract features from audio files 

def extract_features(raw_audio): 

pitches = [] 

for audio in raw_audio: 

pitch, _ = crepe.predict(audio, sr=None, model_capacity='full') 

pitches.append(pitch) 

pitches = pitch_tuner(pitches) 

mfcc_features = [] 

for audio, pitch in zip(raw_audio, pitches): 

mfcc = librosa.feature.mfcc(audio, sr=None, n_mfcc=13) 

mfcc_features.append(np.concatenate([mfcc, np.expand_dims(pitch, axis=0)], axis=0)) 

return np.array(mfcc_features) 

