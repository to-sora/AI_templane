import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
# Specify the directory containing .wav files
wav_directory = "data/train/"  # Replace with your directory path
wav_directory = "data_2/"  # Replace with your directory path
wav_directory = "/home/waito/program_self/year4/AIST2010GROUP/output" 

temp = input("Enter the directory: ")
if temp != "":
    wav_directory = temp
    


 # Replace with your directory path
# Initialize lists to store values
sampling_rates = []
lengths = []
max_dbs = []

# Iterate through all files in the 
list_aud = os.listdir(wav_directory)
list_audio = [x for x in list_aud if x.endswith(".wav")]
for filename in tqdm(list_audio):
    if filename.endswith(".wav"):
        filepath = os.path.join(wav_directory, filename)
        
        # Load the audio file
        audio, sr = librosa.load(filepath, sr=None)
        
        # Compute values
        sampling_rates.append(sr)
        lengths.append(librosa.get_duration(y=audio, sr=sr))
        max_dbs.append(20 * np.log10(np.max(np.abs(audio)) + 1e-6))

# Plotting Sampling Rate Distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(sampling_rates, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Sampling Rate (Hz)')
plt.ylabel('Count')
plt.title('Sampling Rate Distribution')

# Plotting Length Distribution
plt.subplot(1, 3, 2)
plt.hist(lengths, bins=10, color='lightgreen', edgecolor='black')
plt.xlabel('Length (Seconds)')
plt.ylabel('Count')
plt.title('Length Distribution')

# Plotting Max dB Distribution
plt.subplot(1, 3, 3)
plt.hist(max_dbs, bins=10, color='salmon', edgecolor='black')
plt.xlabel('Max dB')
plt.ylabel('Count')
plt.title('Max dB Distribution')

plt.tight_layout()
plt.savefig(f"{wav_directory.replace('/',"_")}_udio_stats.png")
plt.show()
