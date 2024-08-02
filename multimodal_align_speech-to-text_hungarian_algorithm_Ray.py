#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dr. Ray Islam
# Audio and Text alignment - multilmodal
# Optimal alignment using Hungarian algorithm

import numpy as np
import librosa
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Upload audio file and text transcription 
audio_file = "path/.../.wav"
transcript = "Text transcription"

# Convert audio to MFCC features
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  # Load audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr)    # Extract MFCC features
    return mfcc.T

#Calculate DTW distance
def calculate_dtw_distance(features1, features2):
    distance = cdist(features1, features2, metric='euclidean')
    accumulated_cost = np.zeros((len(features1), len(features2)))
    accumulated_cost[0, 0] = distance[0, 0]
    for i in range(1, len(features1)):
        accumulated_cost[i, 0] = distance[i, 0] + accumulated_cost[i-1, 0]
    for j in range(1, len(features2)):
        accumulated_cost[0, j] = distance[0, j] + accumulated_cost[0, j-1]
    for i in range(1, len(features1)):
        for j in range(1, len(features2)):
            accumulated_cost[i, j] = distance[i, j] + min(accumulated_cost[i-1, j], accumulated_cost[i, j-1], accumulated_cost[i-1, j-1])
    return accumulated_cost[-1, -1]

# Extract features from speech audio (MFCC features)
audio_features = extract_features(audio_file)

# Convert transcript text to features (e.g., word embeddings or other features)
# Assumptions: Each word in transcript has corresponding features

# Example: Convert transcript to features 
# Simulate features for each word in the transcript
transcript_words = transcript.split()
transcript_features = np.random.rand(len(transcript_words), 20)  # Replace with actual features

# Calculate DTW distances between audio features and transcript features
dtw_distances = np.zeros((len(audio_features), len(transcript_features)))
for i in range(len(audio_features)):
    for j in range(len(transcript_features)):
        dtw_distances[i, j] = calculate_dtw_distance(audio_features[i], transcript_features[j])

# Perform optimal alignment using Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(dtw_distances)

# Print aligned pairs (index in audio_features, index in transcript_features)
for r, c in zip(row_ind, col_ind):
    print(f"Audio feature index {r} <-> Transcript feature index {c}")

