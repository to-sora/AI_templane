import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable, Optional, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sqlite3
import io
import numpy as np
class AudioDataset(Dataset):
    def __init__(self, audiofile, cache_dir, split='train', transforms=None, freq_limit=4200, config=None):
        self.cache_dir = cache_dir
        self.freq_limit = freq_limit
        os.makedirs(self.cache_dir, exist_ok=True)

        # Build transforms and cache_prefix from config
        if transforms is None:
            transforms_config = config.get('transforms', {})
            transform_type = transforms_config.get('type', 'Spectrogram')
            transform_params = transforms_config.get('params', {})

            # Create transform based on type
            if transform_type == 'Spectrogram':
                self.transforms = torchaudio.transforms.Spectrogram(**transform_params)
            elif transform_type == 'MelSpectrogram':
                self.transforms = torchaudio.transforms.MelSpectrogram(**transform_params)
            print(f"------------->Using {transform_type} transform with params: {transform_params}")

            # Build cache_prefix
            self.cache_prefix = f"{transform_type}_"
            for key, value in transform_params.items():
                self.cache_prefix += f"{key}_{value}_"
        else:
            self.transforms = transforms
            self.cache_prefix = "CustomTransform_"

        # Database setup
        self.db_path = os.path.join(self.cache_dir, 'spectrogram_cache.db')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        # Create table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spectrogram_cache (
                cache_prefix TEXT,
                audio_filename TEXT,
                spectrogram BLOB,
                PRIMARY KEY (cache_prefix, audio_filename)
            )
        ''')
        self.conn.commit()

        # Get list of audio files
        self.audio_files = audiofile
        # Apply debug mode
        if config['debug']:
            self.audio_files = self.audio_files[:100]
            print("Debug mode, using only 100 samples")

        self.data_dir = config['data_dir']
        # Initialize lists to hold data in RAM
        self.spectrograms = []
        self.targets = []

        # Load all data into RAM
        for audio_filename in tqdm(self.audio_files, desc=f"Loading {split} data"):
            audio_path = os.path.join(self.data_dir, audio_filename)
            spectrogram = self.load_or_get_spectrogram(audio_filename, audio_path)

            # Load labels
            csv_filename = audio_filename.replace('.wav', '.csv')
            csv_path = os.path.join(self.data_dir, csv_filename)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Label file {csv_filename} not found for audio {audio_filename}")
            df = pd.read_csv(csv_path)

            # Process labels
            target = self.process_labels(df)

            # Store in RAM
            self.spectrograms.append(spectrogram)
            self.targets.append(target)

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        target = self.targets[idx]
        return spectrogram, target

    def load_or_get_spectrogram(self, audio_filename, audio_path):
        # Check if spectrogram is in cache
        self.cursor.execute('SELECT spectrogram FROM spectrogram_cache WHERE cache_prefix=? AND audio_filename=?', (self.cache_prefix, audio_filename))
        result = self.cursor.fetchone()
        if result is not None:
            spectrogram_blob = result[0]
            spectrogram = torch.load(io.BytesIO(spectrogram_blob))
        else:
            # Load waveform
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 44100:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
            spectrogram = self.transforms(waveform)

            # Adjust frequency bins based on freq_limit
            if isinstance(self.transforms, (torchaudio.transforms.Spectrogram, torchaudio.transforms.MelSpectrogram)):
                freq_bins = spectrogram.size(1)
                max_bin = int(self.freq_limit / (44100 / self.transforms.n_fft))
                max_bin = min(max_bin, freq_bins)
                spectrogram = spectrogram[:, :max_bin, :]
            elif isinstance(self.transforms, CQTTransform):
                frequencies = self.transforms.frequencies
                indices = np.where(frequencies <= self.freq_limit)[0]
                if len(indices) > 0:
                    max_bin = indices[-1] + 1
                    spectrogram = spectrogram[:, :max_bin, :]
                else:
                    # All frequencies are above freq_limit
                    spectrogram = spectrogram[:, :0, :]

            # Average over channels if needed
            spectrogram = spectrogram.mean(dim=0).unsqueeze(0)

            # Save spectrogram to cache
            buffer = io.BytesIO()
            torch.save(spectrogram, buffer)
            spectrogram_blob = buffer.getvalue()
            self.cursor.execute('INSERT INTO spectrogram_cache (cache_prefix, audio_filename, spectrogram) VALUES (?, ?, ?)', (self.cache_prefix, audio_filename, spectrogram_blob))
            self.conn.commit()
        return spectrogram

    def process_labels(self, df):
        # Convert labels to tensor format suitable for the model
        labels = {}

        # Map Note_Type to indices
        note_type_mapping = {'Single': 1, 'Chord': 2}
        labels['note_type'] = torch.tensor(df['Note_Type'].map(note_type_mapping).values, dtype=torch.long)

        # Map Instrument to indices (you may need to expand this mapping)
        labels['instrument'] = torch.tensor(df['Instrument'].apply(self.instrument_to_index).values, dtype=torch.long)

        # Adjust Pitch (MIDI note numbers)
        labels['pitch'] = torch.tensor(df['Pitch'].values, dtype=torch.long)  # Adjusted for piano keys starting from A0 (MIDI 21)

        # Regression targets
        labels['start_time'] = torch.tensor(df['Start_Time'].values, dtype=torch.float32)
        labels['duration'] = torch.tensor(df['Duration'].values, dtype=torch.float32)
        labels['velocity'] = torch.tensor(df['Velocity'].values, dtype=torch.float32)

        return labels

    def instrument_to_index(self, instrument_name):
        # Map instrument names to indices (assuming mapping is predefined)
        # TODO: Expand this mapping based on input dir  or config
        instruments = [
            'Acoustic Grand Piano',
            'Violin',
            'Flute',
            'Electric Guitar (jazz)'
            # Add more instruments as needed
        ]
        instrument_dict = {instrument: idx + 1 for idx, instrument in enumerate(instruments)}
        # You may need to expand this mapping based on your dataset
        return instrument_dict.get(instrument_name, 0)  # Default to 0 if not found

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, Any]:
        """
        Custom collate function to handle batches with variable-length spectrograms and labels.

        Args:
            batch (list): List of tuples (spectrogram, target).

        Returns:
            dict: Batched spectrograms and labels.
        """
        spectrograms, targets = zip(*batch)  # Unzip the batch

        # Find the maximum time dimension
        max_time = max([s.size(2) for s in spectrograms])

        # Initialize batch tensor
        batch_size = len(spectrograms)
        channels = spectrograms[0].size(0)
        freq_bins = spectrograms[0].size(1)

        # Create a tensor of zeros to hold the batch
        padded_spectrograms = torch.zeros(batch_size, channels, freq_bins, max_time)

        # Copy spectrograms into the batch tensor
        for i, s in enumerate(spectrograms):
            time = s.size(2)
            padded_spectrograms[i, :, :, :time] = s

        # Handle labels
        # Since labels are dictionaries with tensors, we'll need to collate each field separately
        collated_labels = {}
        label_keys = targets[0].keys()
        for key in label_keys:
            # Collect all tensors for this key
            label_list = [t[key] for t in targets]
            if label_list[0].dim() == 1:
                # Variable-length sequences, pad them
                collated_labels[key] = pad_sequence(label_list, batch_first=True, padding_value=0)
            else:
                # Fixed-size tensors, stack them
                collated_labels[key] = torch.stack(label_list)
        return [padded_spectrograms, collated_labels]

    def __del__(self):
        self.conn.close()
