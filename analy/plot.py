import os
import random
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_random_wav(directory):
    wav_files = [file for file in os.listdir(directory) if file.endswith('.wav')]
    if not wav_files:
        raise FileNotFoundError("No .wav files found in the directory.")
    selected_file = random.choice(wav_files)
    waveform, sample_rate = torchaudio.load(os.path.join(directory, selected_file))
    return waveform, sample_rate, selected_file

def apply_transform(transform, waveform, sample_rate):
    return transform(waveform)

def apply_cqt(waveform, sample_rate, hop_length=512, n_bins=84, bins_per_octave=12):
    """
    Apply Constant-Q Transform (CQT) to the waveform using librosa.

    Parameters:
    - waveform (torch.Tensor): Audio waveform tensor.
    - sample_rate (int): Sampling rate of the audio.
    - hop_length (int): Number of samples between successive CQT columns.
    - n_bins (int): Number of frequency bins.
    - bins_per_octave (int): Number of bins per octave.

    Returns:
    - np.ndarray: CQT magnitude spectrogram.
    """
    # Ensure waveform is a 1D NumPy array
    waveform_np = waveform.squeeze().numpy()
    
    # If stereo, take the first channel
    if waveform_np.ndim > 1:
        waveform_np = waveform_np[0]
    
    # Compute CQT using librosa
    cqt = librosa.cqt(waveform_np, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    # Convert to magnitude
    cqt_mag = np.abs(cqt)
    print(cqt_mag.shape)
    
    return cqt_mag

def plot_and_save_transform(transform_type, spectrogram, output_dir, file_name, sample_rate=None):
    plt.figure(figsize=(10, 4))
    if transform_type == 'CQT':
        # Transpose to ensure correct orientation for librosa display
        librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=512, x_axis='time', y_axis='cqt_hz', cmap='viridis')
    else:
        # Convert torch tensor to numpy for plotting
        spectrogram_np = spectrogram.log2()[0].numpy()
        librosa.display.specshow(spectrogram_np, sr=sample_rate, hop_length=512, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{transform_type} of {file_name}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    output_path = os.path.join(output_dir, f"{transform_type}_{file_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {transform_type} plot at: {output_path}")

def process_and_plot(directory, output_dir):
    waveform, sample_rate, file_name = load_random_wav(directory)
    transforms_config = {
        "Spectrogram": T.Spectrogram(n_fft=1024, win_length=1024, hop_length=512, power=2.0),
        "MelSpectrogram": T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=2048, win_length=2048, hop_length=512, n_mels=128, power=2.0),
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process torchaudio transforms
    for transform_type, transform in transforms_config.items():
        spectrogram = apply_transform(transform, waveform, sample_rate)
        plot_and_save_transform(transform_type, spectrogram, output_dir, file_name, sample_rate=sample_rate)

    # Process CQT using librosa
    cqt_spectrogram = apply_cqt(waveform, sample_rate)
    plot_and_save_transform("CQT", cqt_spectrogram, output_dir, file_name, sample_rate=sample_rate)

# Example usage
if __name__ == "__main__":
    directory = '../project/data4'  # Directory containing .wav files
    output_dir = './output_plots'   # Directory to save spectrogram plots
    process_and_plot(directory, output_dir)
