import numpy as np
import soundfile as sf
import os
import time
from tritonclient.utils import *
from PIL import Image
import tritonclient.http as httpclient
from datasets import load_dataset
import librosa 
import soundfile
import jiwer 
import pandas as pd
import sys
import json

def transform_path_to_filename(path):
    """
    Transform a path by:
    1. Extracting the portion after 'extracted/'
    2. Converting '/' to '-'
    3. Creating a single filename
    
    Args:
        path (str): The input path
        
    Returns:
        str: Transformed filename
    """
    # Split the path by 'extracted/' and take the second part
    extracted_part = path.split('extracted/')[-1]
    
    # Replace all '/' with '-'
    transformed = extracted_part.replace('/', '-')

    transformed = transformed.replace('.mp3', '.wav')
    
    return transformed

if __name__ == "__main__":    

    if len(sys.argv) < 1:
        print("Usage: python preprocessing.py <samples>")
        sys.exit(1)

    samples = int(sys.argv[1])
    dataset_name = 'mozilla-foundation/common_voice_11_0'
    split = 'test'
    dataset = load_dataset(dataset_name, "en", split=split, trust_remote_code=True)
    print(dataset)
    performance = {
        'file name': [],
        'audio length (s)': [],
        'time taken (s)': [],
        'transcribed text': [],
        'actual text': [],
        'accuracy': []
    }
    initial_time = time.time()
    dir = "handlers/triton/audio"
    if not os.path.exists(dir):
        os.makedirs(dir)

    op = {}
    
    for idx, audio_file in enumerate(dataset):
        if samples == 0:
            break
        audio = audio_file['audio']['array']
        print(f'{samples}\n\n\n')
        print("before", audio)
        start = time.time()
        audio = librosa.resample(audio, orig_sr=audio_file['audio']['sampling_rate'], target_sr=16000)
        print(os.path.split(audio_file["path"]))
        op[transform_path_to_filename(audio_file["path"])] = audio_file["sentence"]
        output_path = f'{dir}/{transform_path_to_filename(audio_file["path"])}'
        sf.write(output_path, audio, 16000)
        print("after: ", audio)
        samples = samples - 1
    
    with open('map.json', 'w') as f:
        json.dump(op, f)