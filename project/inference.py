#!/usr/bin/env python3
# inference.py
import csv
import json5 as json
import os
import argparse
# import json
import torch
import torchaudio
import pandas as pd
import pretty_midi
from midi2audio import FluidSynth
from models.detr_audio import DETRAudio  # Ensure this import path is correct
from utils.criterion import HungarianMatcher
from utils.dataset import CQTTransform
from tqdm import tqdm
import io
import numpy as np
# Add these mappings at the beginning of your script or within the `process_outputs` function

# Note_Type mapping
note_type_dict = {
    0: 'No_Note',       # Assuming 0 is 'No_Note' or 'None'
    1: 'Note',
    2: 'Chord',
    # Add other note types as necessary
}

# Instrument mapping
instrument_dict = {
    0: 'None',  # Assuming 0 is 'None' or 'No Instrument'
    1: 'Acoustic Grand Piano',
    2: 'Violin',
    3: 'Flute',
    4: 'Electric Guitar (jazz)',
    # Add more instruments as per your model's training classes
}

def load_config(config_path):
    """
    Loads the JSON configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def initialize_model(config, model_path, device):
    """
    Initializes the DETRAudio model and loads the trained weights.

    Args:
        config (dict): Configuration dictionary.
        model_path (str): Path to the model checkpoint.
        device (str): Device to load the model on.

    Returns:
        nn.Module: Loaded DETRAudio model.
    """
    model = DETRAudio(config)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    return model

def preprocess_audio(audio_path, config, chunk_size=None, overlap=0):
    """
    Loads and preprocesses the audio file into spectrogram chunks.

    Args:
        audio_path (str): Path to the input audio file.
        config (dict): Configuration dictionary.
        chunk_size (float, optional): Duration of each chunk in seconds. If None, process the entire audio.
        overlap (float, optional): Overlap duration between consecutive chunks in seconds.

    Returns:
        list of torch.Tensor: List of preprocessed spectrogram tensors for each chunk.
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    target_sample_rate = config.get('sample_rate', 44100)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Calculate total duration
    total_duration = waveform.size(1) / sample_rate  # in seconds

    # Define chunk parameters
    if chunk_size is not None:
        hop_size = chunk_size - overlap
        if hop_size <= 0:
            raise ValueError("Overlap must be smaller than chunk_size.")
    else:
        hop_size = total_duration  # Process entire audio as one chunk

    # Split waveform into chunks
    spectrogram_chunks = []
    start_sample = 0
    chunk_samples = int(chunk_size * sample_rate) if chunk_size else waveform.size(1)
    hop_samples = int(hop_size * sample_rate) if chunk_size else waveform.size(1)

    while start_sample < waveform.size(1):
        end_sample = start_sample + chunk_samples
        chunk_waveform = waveform[:, start_sample:end_sample]
        
        # Handle last chunk
        if chunk_waveform.size(1) < chunk_samples:
            if chunk_size:
                # Optionally, pad the last chunk with zeros
                padding = chunk_samples - chunk_waveform.size(1)
                chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, padding))
            else:
                pass  # No padding needed for full audio

        # Apply transforms
        transforms_config = config.get('transforms', {})
        transform_type = transforms_config.get('type', 'Spectrogram')
        transform_params = transforms_config.get('params', {})

        if transform_type == 'Spectrogram':
            transform = torchaudio.transforms.Spectrogram(**transform_params)
        elif transform_type == 'MelSpectrogram':
            transform = torchaudio.transforms.MelSpectrogram(**transform_params)
        elif transform_type == 'CQT':
            transform = CQTTransform(sample_rate=sample_rate, **transform_params)
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

        spectrogram = transform(chunk_waveform)  # Shape: [channels, freq_bins, time_steps]

        # Adjust frequency bins based on freq_limit
        freq_limit = config.get('freq_limit', 4200)
        if isinstance(transform, (torchaudio.transforms.Spectrogram, torchaudio.transforms.MelSpectrogram)):
            n_fft = transform_params.get('n_fft', 1024)
            freq_bins = spectrogram.size(1)
            max_bin = int(freq_limit / (sample_rate / n_fft))
            max_bin = min(max_bin, freq_bins)
            spectrogram = spectrogram[:, :max_bin, :]
        elif isinstance(transform, CQTTransform):
            frequencies = transform.frequencies
            indices = [i for i, f in enumerate(frequencies) if f <= freq_limit]
            if indices:
                max_bin = indices[-1] + 1
                spectrogram = spectrogram[:, :max_bin, :]
            else:
                spectrogram = spectrogram[:, :0, :]  # All frequencies above freq_limit

        # Average over channels if needed
        spectrogram = spectrogram.mean(dim=0).unsqueeze(0)  # Shape: [1, freq_bins, time_steps]

        # Add batch dimension and permute to [batch_size, channels, freq_bins, time_steps] if needed
        spectrogram = spectrogram.unsqueeze(0)  # Shape: [1, 1, freq_bins, time_steps]

        spectrogram_chunks.append(spectrogram)

        # Update start_sample
        start_sample += hop_samples

    print(f"Total duration: {total_duration:.2f}s, Number of chunks: {len(spectrogram_chunks)}")
    return spectrogram_chunks

def run_inference(model, spectrogram_chunks, device):
    """
    Runs the model on the preprocessed spectrogram chunks to get predictions.

    Args:
        model (nn.Module): Loaded DETRAudio model.
        spectrogram_chunks (list of torch.Tensor): List of spectrogram tensors for each chunk.
        device (str): Device to run the model on.

    Returns:
        list of dict: List of model outputs for each chunk.
    """
    all_outputs = []
    for i, spectrogram in enumerate(tqdm(spectrogram_chunks, desc="Running inference on chunks")):
        with torch.no_grad():
            spectrogram = spectrogram.to(device)
            outputs = model(spectrogram)
            all_outputs.append(outputs)
    return all_outputs

def process_outputs(all_outputs, config, device, chunk_start_times, chunk_size):
    """
    Processes model outputs from all chunks to generate aggregated prediction DataFrame.

    Args:
        all_outputs (list of dict): List of model outputs for each chunk.
        config (dict): Configuration dictionary.
        device (str): Device used for computation.
        chunk_start_times (list of float): List of start times for each chunk in seconds.
        chunk_size (float): Duration of each chunk in seconds.

    Returns:
        pd.DataFrame: DataFrame containing aggregated predictions.
    """
    # matcher = HungarianMatcher(
    #     cost_note_type=config.get('cost_note_type', 1.0),
    #     cost_instrument=config.get('cost_instrument', 1.0),
    #     cost_pitch=config.get('cost_pitch', 1.0),
    #     cost_regression=config.get('cost_regression', 1.0),
    #     use_softmax=config.get('loss', {}).get('use_softmax', False)
    # )
    # print(all_outputs)

    all_predictions = []

    for chunk_idx, outputs in enumerate(all_outputs):
        # Extract predictions
        pred_note_type = outputs['pred_note_type']  # [batch_size, num_queries, num_note_types]
        pred_instrument = outputs['pred_instrument']  # [batch_size, num_queries, num_instruments]
        pred_pitch = outputs['pred_pitch']  # [batch_size, num_queries, num_pitches]
        pred_regression = outputs['pred_regression']  # [batch_size, num_queries, 3]

        # Apply softmax to classification outputs if needed
        if config.get('loss', {}).get('use_softmax', False):
            pred_note_type = torch.softmax(pred_note_type, dim=-1)
            pred_instrument = torch.softmax(pred_instrument, dim=-1)
            pred_pitch = torch.softmax(pred_pitch, dim=-1)
        prob_no_obj_note_type = pred_note_type[..., 0]  # [batch_size, num_queries]
        prob_no_obj_instrument = pred_instrument[..., 0]  # [batch_size, num_queries]
        prob_no_obj_pitch = pred_pitch[..., -1]  # [batch_size, num_queries]
        # print(pred_pitch.shape)
        no_obj_mask = (prob_no_obj_note_type > 0.8) | (prob_no_obj_instrument > 0.8) | (prob_no_obj_pitch > 0.8)

        # Exclude 'no object' class when taking argmax
        pred_note_type_no_noobj = pred_note_type.clone()
        pred_note_type_no_noobj[..., 0] = -float('inf')
        _, pred_note_type_ids = pred_note_type_no_noobj.max(-1)

        pred_instrument_no_noobj = pred_instrument.clone()
        pred_instrument_no_noobj[..., 0] = -float('inf')
        _, pred_instrument_ids = pred_instrument_no_noobj.max(-1)

        pred_pitch_no_noobj = pred_pitch.clone()
        pred_pitch_no_noobj[..., -1] = -float('inf')
        _, pred_pitch_ids = pred_pitch_no_noobj.max(-1)

        # # Get the most probable classes
        # _, pred_note_type_ids = pred_note_type.max(-1)  # [batch_size, num_queries]
        # _, pred_instrument_ids = pred_instrument.max(-1)  # [batch_size, num_queries]
        # _, pred_pitch_ids = pred_pitch.max(-1)  # [batch_size, num_queries]


        # For regression, extract start_time, duration, velocity
        pred_start_time = pred_regression[:, :, 0]  # [batch_size, num_queries]
        pred_duration = pred_regression[:, :, 1]    # [batch_size, num_queries]
        pred_velocity = pred_regression[:, :, 2]    # [batch_size, num_queries]
        note_type_names = [note_type_dict.get(int(id_), 'Unknown') for id_ in pred_note_type_ids.flatten()]
        # print(pred_note_type_ids)
        instrument_names = [instrument_dict.get(int(id_), 'Unknown') for id_ in pred_instrument_ids.flatten()]
        # print(pred_instrument_ids)

        # Reshape to original dimensions
        note_type_names = np.array(note_type_names).reshape(pred_note_type_ids.shape)
        instrument_names = np.array(instrument_names).reshape(pred_instrument_ids.shape)

        # Convert tensors to CPU and numpy
        pred_note_type_ids = pred_note_type_ids.cpu().numpy()
        pred_instrument_ids = pred_instrument_ids.cpu().numpy()
        pred_pitch_ids = pred_pitch_ids.cpu().numpy()
        pred_start_time = pred_start_time.cpu().numpy()
        pred_duration = pred_duration.cpu().numpy()
        pred_velocity = pred_velocity.cpu().numpy()

        # Assuming batch_size=1 for inference
        batch_size = pred_note_type_ids.shape[0]
        for b in range(batch_size):
            for q in range(pred_note_type_ids.shape[1]):
                # Get names instead of IDs
                if no_obj_mask[b, q]:
                    continue  # Skip this prediction
                    print("No object")
                # print("Object")
                note_type = note_type_names[b, q]
                instrument = instrument_names[b, q]
                pitch = pred_pitch_ids[b, q]
                start_time = pred_start_time[b, q]
                duration = pred_duration[b, q]
                velocity = pred_velocity[b, q]
                # print(note_type, instrument, pitch, start_time, duration, velocity)

                # Filter out 'No_Note' or 'None' predictions
                if note_type in ['No_Note', 'Unknown'] and instrument in ['None', 'Unknown']:
                    continue  # Skip 'no object' predictions

                # Adjust start_time based on chunk index and overlap
                adjusted_start_time = chunk_start_times[chunk_idx] + start_time

                all_predictions.append({
                    'Note_Type': note_type,
                    'Instrument': instrument,
                    'Pitch': int(pitch),
                    'Start_Time': adjusted_start_time,
                    'Duration': duration,
                    'Velocity': int(velocity)
                })
                # print(all_predictions[-1])


    df = pd.DataFrame(all_predictions)
    df = df.sort_values(by='Start_Time', ascending=True).reset_index(drop=True)
    print(df)
    return df

def save_csv(df, csv_path):
    """
    Saves the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing predictions.
        csv_path (str): Path to save the CSV file.
    """
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to CSV at {csv_path}")

def csv_to_midi(csv_path, midi_path, config):
    """
    Converts the CSV predictions to a MIDI file.

    Args:
        csv_path (str): Path to the CSV file with predictions.
        midi_path (str): Path to save the MIDI file.
        config (dict): Configuration dictionary.
    """
    df = pd.read_csv(csv_path)
    midi = pretty_midi.PrettyMIDI()

# Add these mappings at the beginning of your script or within the `process_outputs` function

# Note_Type mapping
    # note_type_dict = {
    #     0: 'No_Note',       # Assuming 0 is 'No_Note' or 'None'
    #     1: 'Note',
    #     2: 'Chord',
    #     # Add other note types as necessary
    # }

    # # Instrument mapping
    # instrument_dict = {
    #     0: 'None',  # Assuming 0 is 'None' or 'No Instrument'
    #     1: 'Acoustic Grand Piano',
    #     2: 'Violin',
    #     3: 'Flute',
    #     4: 'Electric Guitar (jazz)',
    #     # Add more instruments as per your model's training classes
    # }

    # TODO: load the instruments from the config file

    # Create mapping from instrument ID to instrument name
    # instrument_dict = {idx + 1: instrument for idx, instrument in enumerate(instruments)}
    # instrument_dict[0] = 'None'  # Assuming 0 is 'Unknown' or 'No instrument'

    # # Create a dictionary to hold instruments
    instruments = {}

    for _, row in df.iterrows():
        note_type = row['Note_Type']
        instrument_id = row['Instrument']
        pitch = row['Pitch']
        start_time = row['Start_Time']
        duration = row['Duration']
        velocity = int(row['Velocity'])
        if not (0 <= pitch <= 127):
            continue  # Skip invalid pitches

        # Get instrument name
        # get global instrument_dict
        # global instrument_dict
        instrument_name = instrument_dict.get(instrument_id, 'Acoustic Grand Piano')

        try:
            # Convert the instrument name to a MIDI program number
            program = pretty_midi.instrument_name_to_program(instrument_name)
        except ValueError:
            # Fallback to default instrument if the name is invalid
            program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            instrument_name = 'Acoustic Grand Piano'

        # Check if the instrument already exists in the `instruments` dictionary
        if instrument_name not in instruments:
            # Create and add a new instrument if not already present
            instrument = pretty_midi.Instrument(program=program, name=instrument_name)
            instruments[instrument_name] = instrument
        else:
            # Retrieve the existing instrument
            instrument = instruments[instrument_name]


        # Create note
        end_time = start_time + duration
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(pitch),
            start=float(start_time),
            end=float(end_time)
        )
        instrument.notes.append(note)

    # Add instruments to MIDI
    for inst in instruments.values():
        midi.instruments.append(inst)

    # Write MIDI file
    midi.write(midi_path)
    print(f"Converted CSV to MIDI at {midi_path}")

def midi_to_wav(midi_path, wav_path, sound_font):
    """
    Converts a MIDI file to WAV using FluidSynth.

    Args:
        midi_path (str): Path to the MIDI file.
        wav_path (str): Path to save the WAV file.
        sound_font (str): Path to the SoundFont (.sf2) file.
    """
    fs = FluidSynth(sound_font=sound_font)
    fs.midi_to_audio(midi_path, wav_path)
    print(f"Converted MIDI to WAV at {wav_path}")

def initialize_metadata(metadata_path):
    """
    Initializes the metadata CSV file with headers.

    Parameters:
        metadata_path (str): Path to the metadata CSV file.
    """
    headers = [
        'File_ID',
        'Number_of_Notes',
        'WAV_Length_Seconds',
        'Enable_Scale',
        'Scale_Name',
        'Enable_Chord_Progression',
        'Enable_Multiple_Instruments',
        'Enable_Phrases',
        'Phrase_Length_Range',
        'Enable_Augmentation',
        'Transpose_Range',
        'Enable_Overlap',
        'Note_Range',
        'Duration_Range',
        'Velocity_Range',
        'Num_Notes_Range',
        'Chord_Probability',
        'Max_Notes_Per_Chord'
    ]
    with open(metadata_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

def append_metadata(metadata_path, metadata_entry):
    """
    Appends a single metadata entry to the metadata CSV file.

    Parameters:
        metadata_path (str): Path to the metadata CSV file.
        metadata_entry (dict): Dictionary containing metadata fields.
    """
    with open(metadata_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            metadata_entry['File_ID'],
            metadata_entry['Number_of_Notes'],
            metadata_entry['WAV_Length_Seconds'],
            metadata_entry['Enable_Scale'],
            metadata_entry['Scale_Name'],
            metadata_entry['Enable_Chord_Progression'],
            metadata_entry['Enable_Multiple_Instruments'],
            metadata_entry['Enable_Phrases'],
            metadata_entry['Phrase_Length_Range'],
            metadata_entry['Enable_Augmentation'],
            metadata_entry['Transpose_Range'],
            metadata_entry['Enable_Overlap'],
            metadata_entry['Note_Range'],
            metadata_entry['Duration_Range'],
            metadata_entry['Velocity_Range'],
            metadata_entry['Num_Notes_Range'],
            metadata_entry['Chord_Probability'],
            metadata_entry['Max_Notes_Per_Chord']
        ])

def main():
    parser = argparse.ArgumentParser(description="Inference Script for DETRAudio Model with Chunking Support")
    parser.add_argument('--input', type=str, required=True, help='Path to input audio file (e.g., WAV)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output CSV, MIDI, WAV files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON configuration file')
    parser.add_argument('--sound_font', type=str, default=None, help='Path to SoundFont (.sf2) file for MIDI to WAV conversion')
    
    # New arguments for chunking
    parser.add_argument('--chunk_size', type=float, default=None, help='Duration of each audio chunk in seconds (e.g., 5.0)')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap duration between consecutive chunks in seconds (default: 0.0)')
    
    args = parser.parse_args()

    input_audio = args.input
    output_dir = args.output_dir
    model_path = args.model_path
    config_path = args.config
    sound_font = args.sound_font
    chunk_size = args.chunk_size
    overlap = args.overlap

    # Load configuration
    config = load_config(config_path)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model
    model = initialize_model(config, model_path, device)

    # Preprocess audio into chunks
    spectrogram_chunks = preprocess_audio(input_audio, config, chunk_size=chunk_size, overlap=overlap)
    print(f"Preprocessed audio {input_audio} into {len(spectrogram_chunks)} spectrogram chunks.")

    # Calculate chunk start times for time adjustment
    chunk_start_times = []
    if chunk_size is not None:
        hop_size = chunk_size - overlap
        start_time = 0.0
        for _ in spectrogram_chunks:
            chunk_start_times.append(start_time)
            start_time += hop_size
    else:
        chunk_start_times.append(0.0)

    # Run inference on all chunks
    all_outputs = run_inference(model, spectrogram_chunks, device)
    print("Model inference completed on all chunks.")

    # Process outputs and aggregate predictions
    df = process_outputs(all_outputs, config, device, chunk_start_times, chunk_size if chunk_size else 0.0)
    # print(df)
    print("Processed model outputs into aggregated predictions.")

    # Prepare output file paths
    base_filename = os.path.splitext(os.path.basename(input_audio))[0]
    csv_output_path = os.path.join(output_dir, f"{base_filename}_predictions.csv")
    midi_output_path = os.path.join(output_dir, f"{base_filename}.mid")
    wav_output_path = os.path.join(output_dir, f"{base_filename}.wav")
    metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    save_csv(df, csv_output_path)

    # Convert CSV to MIDI
    csv_to_midi(csv_output_path, midi_output_path, config)

    # Convert MIDI to WAV
    if sound_font is None:
        # Use default sound font from config or raise error
        sound_font = config.get('sound_font_path', './Fluid_related/FluidR3_GM.sf2')
        if not os.path.exists(sound_font):
            raise FileNotFoundError(f"SoundFont file not found at {sound_font}. Please provide a valid path using --sound_font.")
    midi_to_wav(midi_output_path, wav_output_path, sound_font)

    # Optional: Initialize and append metadata
    # Initialize metadata CSV if not exists
    if not os.path.exists(metadata_path):
        initialize_metadata(metadata_path)
    
    # Gather metadata
    num_notes = len(df)
    wav_length = torchaudio.info(wav_output_path).num_frames / torchaudio.info(wav_output_path).sample_rate if os.path.exists(wav_output_path) else ''
    
    # Prepare metadata entry
    metadata_entry = {
        'File_ID': base_filename,
        'Number_of_Notes': num_notes,
        'WAV_Length_Seconds': wav_length,
        'Enable_Scale': config.get('Enable_Scale', False),
        'Scale_Name': config.get('Scale_Name', ''),
        'Enable_Chord_Progression': config.get('Enable_Chord_Progression', False),
        'Enable_Multiple_Instruments': config.get('Enable_Multiple_Instruments', False),
        'Enable_Phrases': config.get('Enable_Phrases', False),
        'Phrase_Length_Range': config.get('Phrase_Length_Range', ''),
        'Enable_Augmentation': config.get('Enable_Augmentation', False),
        'Transpose_Range': config.get('Transpose_Range', ''),
        'Enable_Overlap': overlap > 0.0,
        'Note_Range': config.get('Note_Range', ''),
        'Duration_Range': config.get('Duration_Range', ''),
        'Velocity_Range': config.get('Velocity_Range', ''),
        'Num_Notes_Range': config.get('Num_Notes_Range', ''),
        'Chord_Probability': config.get('Chord_Probability', ''),
        'Max_Notes_Per_Chord': config.get('Max_Notes_Per_Chord', '')
    }

    # Append metadata to metadata.csv
    append_metadata(metadata_path, metadata_entry)
    print(f"Appended metadata to {metadata_path}")

    print("Inference pipeline with chunking completed successfully.")

if __name__ == "__main__":
    main()
