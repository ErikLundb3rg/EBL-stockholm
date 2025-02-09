import librosa
import numpy as np
import json
import sys
print(sys.path)  # This will show us where Python is looking for modules
from moviepy import *
from moviepy.editor import VideoFileClip
from numpy.polynomial import Polynomial
import os
from text_data import client

def extract_audio(mp4_path, wav_path):
    """
    Extract audio from an MP4 file and save it as WAV.
    
    Parameters:
    mp4_path (str): Path to the input MP4 file
    wav_path (str): Path where the WAV file should be saved
    """
    try:
        # Load the video file
        video = VideoFileClip(mp4_path)
        
        # Extract the audio
        audio = video.audio
        
        # Write the audio to a WAV file
        audio.write_audiofile(wav_path)
        
        # Close the files to free up system resources
        audio.close()
        video.close()
        
        print(f"Successfully extracted audio to {wav_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def pich(video_file_path):
    """
    Extract pitch and intensity features from a video file's audio.
    
    This function extracts audio from a video file and analyzes it to determine
    the fundamental frequency (pitch) and intensity (RMS energy) over time.
    It uses the PYIN algorithm for pitch estimation, which is more accurate
    than basic pitch detection methods.
    
    Parameters:
    -----------
    video_file_path : str
        Path to the input video file from which to extract audio features
        
    Returns:
    --------
    tuple
        A tuple containing three elements:
        - time_marks : ndarray
            Array of time points corresponding to the extracted features
        - f0 : ndarray
            Array of fundamental frequency values (pitch) in Hz
        - rms : ndarray
            Array of RMS energy values representing intensity
            
    Notes:
    ------
    - The pitch detection range is set from C1 to C8 on the musical scale
    - The audio is temporarily saved as 'output_audio.wav'
    - Uses librosa's PYIN implementation for pitch detection
    - Sample rate is preserved from the original audio
    """
    # Load the audio file
    audio_file = 'output_audio.wav'  # replace with your file path
    extract_audio(video_file_path, audio_file)
    y, sr = librosa.load(audio_file, sr=None)  # y is the audio signal, sr is the sampling rate

    # 1. Extract intensity (RMS) - Root Mean Square Energy
    rms = librosa.feature.rms(y=y)

    # 2. Extract pitch (fundamental frequency) using pyin (a better pitch estimator)
    print("pitch")
    if int(os.getenv("TIME_SAVE", 0)) == 0:
        f0, _voiced_flag, _voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), sr=sr)
    else:
        f0 = np.full_like(rms[0], 100)
    print("pitch")

    return librosa.times_like(rms, sr=sr), f0, rms[0]


def analyze_pitch_ranges(pitch_values):
    """
    Analyze pitch values and categorize them into 5 intensity levels.
    
    Parameters:
    pitch_values (array-like): Array of pitch values in Hertz
    
    Returns:
    tuple: (range_counts, range_labels)
        - range_counts: Count of values in each intensity level
        - range_labels: String labels for each level
    """
    pitch_values = np.array(pitch_values)
    
    # Remove any negative values or zeros
    pitch_values = pitch_values[pitch_values > 0]
    
    # Define 5 intensity ranges with boundaries
    boundaries = [85, 130, 200, 300, 400, 500]  # Hz
    labels = [
        "Very Low (85-130Hz)",
        "Low (130-200Hz)",
        "Medium (200-300Hz)",
        "High (300-400Hz)",
        "Very High (400-500Hz)"
    ]
    
    # Initialize counts array
    range_counts = np.zeros(5, dtype=int)
    
    # Count values in each range
    for i in range(5):
        range_counts[i] = np.sum((pitch_values >= boundaries[i]) & 
                                (pitch_values < boundaries[i + 1]))
    
    return range_counts, labels

def add_pich_to_transcript(video_file_path, transcript, interpolation_degree=3):
    """
    Analyze pitch features from a video and add them to a transcript dictionary,
    optionally performing polynomial interpolation over sentence segments.
    
    This function extracts pitch and intensity features from a video's audio
    and adds them to an existing transcript dictionary. It can also perform
    polynomial interpolation over the time spans of transcribed sentences.
    
    Parameters:
    -----------
    video_file_path : str
        Path to the input video file
    transcript : dict
        Dictionary containing transcript data. Expected to have a 'transcription'
        key with a 'sentences' list containing objects with 'start' and 'end' times
    interpolation_degree : int, optional
        Degree of polynomial to use for interpolation. If 0 or None,
        no interpolation is performed. Default is 3
        
    Returns:
    --------
    dict
        Updated transcript dictionary with added 'pitch' key containing:
        - timestamps : ndarray
            Time points for the extracted features
        - f0 : ndarray
            Fundamental frequency values (pitch)
        - rms : ndarray
            RMS energy values (intensity)
        - interpolations : ndarray, optional
            Interpolated pitch values if interpolation_degree > 0
            
    Notes:
    ------
    - If 'pitch' key already exists in transcript, its values are updated
    - Interpolation is performed over each sentence's time span if specified
    - Uses polynomial interpolation to fill gaps in pitch data
    """
    time_marks, f0, rms = pich(video_file_path=video_file_path)
    if "pitch" in transcript:
        transcript["pitch"]["timestamps"] = time_marks.tolist()
        transcript["pitch"]["f0"] = f0.tolist()
        transcript["pitch"]["rms"] = time_marks.tolist()
    else:
        transcript["pitch"] = {
            "timestamps": time_marks.tolist(), 
            "f0": f0.tolist(),
            "rms": rms.tolist(),
        }
    if interpolation_degree:
        time_pairs = [(obj["start"], obj["end"]) for obj in transcript["transcription"]["sentences"]]
        transcript["pitch"]["interpolations"] = interpolate_regions(
            time_marks,
            f0,
            time_pairs
        ).tolist()

    range_counts, labels = analyze_pitch_ranges(f0)
    transcript["pitch"]["range_counts"] = range_counts.tolist()
    transcript["pitch"]["range_labels"] = labels

    gemini_evaluation_of_pitch = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
            From these labels and ranges, Please classify the speakers pitch. Be clear and concise, refer to the speaker as 'you'.
            {labels}
            {range_counts}
        """,
    ).text

    transcript["pitch"]["evaluation"] = gemini_evaluation_of_pitch
    
    return transcript




def interpolate_regions(times: np.ndarray, values: np.ndarray, time_pairs:list[tuple[int]], degree=3):
    """
    Perform polynomial interpolation on specified time regions in a time series.
    
    Parameters:
    times: np.array
        Array of time points
    values: np.array
        Array of values corresponding to the time points
    time_pairs: list of tuples
        List of (start_time, end_time) pairs defining regions to interpolate
    degree: int
        Degree of the polynomial to fit (default: 3)
        
    Returns:
    np.array
        Array of same shape as input values, with interpolated values in specified regions
    """
    result = np.full_like(values, np.nan, dtype=np.double)
    
    for start_time, end_time in time_pairs:
        # Get indices for the time region
        mask = (times >= start_time) & (times <= end_time)
        region_times = times[mask]
        region_values = values[mask]
        
        # Skip if no valid points in region
        if len(region_times) == 0:
            continue
            
        # Get only non-NaN values for fitting
        valid_mask = ~np.isnan(region_values)
        if not np.any(valid_mask):
            continue
            
        valid_times = region_times[valid_mask]
        valid_values = region_values[valid_mask]
        
        # Normalize times to prevent numerical issues
        time_offset = valid_times[0]
        time_scale = valid_times[-1] - valid_times[0]
        if time_scale == 0:
            continue
            
        normalized_times = (valid_times - time_offset) / time_scale
        
        try:
            # Fit polynomial
            poly = Polynomial.fit(normalized_times, valid_values, deg=degree)
            
            # Generate interpolated values for all points in the region
            normalized_region_times = (region_times - time_offset) / time_scale
            interpolated_values = poly(normalized_region_times)
            
            # Replace values in the result array
            result[mask] = interpolated_values
            
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Warning: Could not interpolate region {start_time}-{end_time}: {str(e)}")
            continue
    
    return result
