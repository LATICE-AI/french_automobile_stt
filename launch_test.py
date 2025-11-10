"""
Speech-to-Text (STT) Services Testing and Comparison Script
================================================================================
"""

import logging
import asyncio
from dotenv import load_dotenv
import time
import csv
import os
from livekit.plugins import openai, deepgram, groq, gladia, silero, fal, speechmatics, elevenlabs, google
from livekit.plugins.speechmatics.types import TranscriptionConfig
from livekit.agents.utils.codecs import AudioStreamDecoder
import aiohttp
from livekit import rtc
from livekit.agents import stt
import requests
import uuid
from collections import OrderedDict

# Logging configuration
logger = logging.getLogger("test_stt")
logger.setLevel(logging.INFO)

# Load environment variables from .env file
load_dotenv(override=True)

    
    
def latice_stt(audio_bytes):
    api_url = "http://api.latice.ai/transcribe"
    headers = {
        "api-key": "********-****-****-****-************",
        "Accept": "application/json",
        "model-id": "********-****-****-****-************",
    }
    files = {
        "audio_file": (f"audio_{uuid.uuid4()}.mp3", audio_bytes, "audio/mp3")
    }
    response = requests.post(f"{api_url}", files=files, headers=headers, timeout=500)
    return response.json()['transcription']

# STT services configuration to test
# ======================================

stt_to_tests = [
    # Custom STT service 
    {"name": "latice", "stt": None, "script": latice_stt},
    
    # Cloud STT services 
    {"name": "deepgram-nova-3", "stt": deepgram.STT(language="fr", model="nova-3")},
    {"name": "gladia", "stt": gladia.STT(languages=["fr"])},
    {"name": "google-latest_long", "stt": google.STT(model="latest_long", spoken_punctuation=False, languages="fr-FR", location="eu")},
    {"name": "groq-whisper-large-v3", "stt": openai.STT.with_groq(model="whisper-large-v3",language="fr")},
    {"name": "groq-whisper-large-v3-turbo", "stt": groq.STT(model="whisper-large-v3-turbo",language="fr")},
    {"name": "elevenlabs", "stt": elevenlabs.STT(language_code="fr")},
    {"name": "fal", "stt": fal.WizperSTT(language="fr")}, 
    {"name": "speechmatics", "stt": speechmatics.STT(language="fr")},
    {"name": "openai-gpt-4o-transcribe", "stt": openai.STT(model="gpt-4o-transcribe",language="fr")},
]

async def load_audio_samples():
    samples_dir = 'Audio'
    samples = []
    
    
    for filename in os.listdir(samples_dir):
        if filename.endswith(('.wav', '.mp3', '.m4a')):
            decoder = AudioStreamDecoder(sample_rate=16000, num_channels=1) 
            with open(os.path.join(samples_dir, filename), 'rb') as f:
                audio_bytes = f.read()
            with open(os.path.join(samples_dir, filename), 'rb') as f:
                while chunk := f.read(4096):
                    decoder.push(chunk)
                decoder.end_input() 
                frames = []
                async for frame in decoder:
                    frames.append(frame)
                
                samples.append({
                    'path': os.path.join(samples_dir, filename),
                    'decoder': decoder,
                    'frames': frames,
                    'audio_bytes': audio_bytes
            })
    return samples

vad = silero.VAD.load()

async def stt_node(audio: list[rtc.AudioFrame], stt:stt.STT):
    """
    Process an audio sample with an STT service
    """
    try:
        # Method 1: Direct transcription (faster)
        print(f"Number of audio frames: {len(audio)}")
        merged_audio = rtc.combine_audio_frames(audio)
        result = await stt.recognize(merged_audio)
        if result and result.alternatives and len(result.alternatives) > 0:
            return result.alternatives[0].text
        else:
            return None
    except Exception as e:
        return None

async def test_stt():
    """
    Main function for testing STT services
    """
    csv_path = 'result.csv'  # Output results file
    fieldnames = ['sample', 'model', 'response_time', 'transcription', 'accuracy']

    # Structures to manage existing and new results
    existing_rows_by_sample: OrderedDict[str, list[dict]] = OrderedDict()
    existing_by_key: set[tuple[str, str]] = set()

    # Load existing results from CSV file
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as csvfile:
            try:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Ignore empty lines
                    if row is None or not any((value or '').strip() for value in row.values()):
                        continue
                    
                    sample_name = (row.get('sample') or '').strip()
                    model_name = (row.get('model') or '').strip()
                    if not sample_name or not model_name:
                        continue
                    
                    # Organiser les résultats par échantillon
                    if sample_name not in existing_rows_by_sample:
                        existing_rows_by_sample[sample_name] = []
                    
                    kept_row = {
                        'sample': sample_name,
                        'model': model_name,
                        'response_time': row.get('response_time', ''),
                        'transcription': row.get('transcription', ''),
                        'accuracy': row.get('accuracy', ''),
                    }
                    existing_rows_by_sample[sample_name].append(kept_row)
                    existing_by_key.add((sample_name, model_name))
            except Exception:
                # In case of read error, start from scratch
                existing_rows_by_sample = OrderedDict()
                existing_by_key = set()

    # Build canonical model order
    # 1. First existing models (in encountered order)
    # 2. Then new models from this execution
    existing_model_order: list[str] = []
    for sample_name, rows in existing_rows_by_sample.items():
        for r in rows:
            m = r.get('model', '')
            if m and m not in existing_model_order:
                existing_model_order.append(m)
    
    run_model_order: list[str] = [t['name'] for t in stt_to_tests]
    canonical_model_order: list[str] = list(existing_model_order)
    for m in run_model_order:
        if m not in canonical_model_order:
            canonical_model_order.append(m)

    run_model_names: set[str] = set(run_model_order)

    # Shared HTTP session for all services
    async with aiohttp.ClientSession() as session:
        # Configure session for all STT services
        for test in stt_to_tests:
            if test.get("stt"):
                test["stt"]._session = session

        # Initialize performance metrics
        response_times = {test['name']: [] for test in stt_to_tests}
        
        # Load all audio samples
        samples = await load_audio_samples()
        samples_by_basename = {os.path.basename(s['path']): s for s in samples}

        # Preserve existing sample order, add new ones at the end
        sample_order: list[str] = list(existing_rows_by_sample.keys())
        for s in samples:
            base = os.path.basename(s['path'])
            if base not in sample_order:
                sample_order.append(base)

        # Traiter chaque échantillon
        final_rows_by_sample: OrderedDict[str, list[dict]] = OrderedDict()
        for sample_basename in sample_order:
            final_rows: list[dict] = []
            existing_rows_for_sample = {row['model']: row for row in existing_rows_by_sample.get(sample_basename, [])}

            # Test each model on this sample
            for model_name in canonical_model_order:
                # If the result already exists, keep it
                if model_name in existing_rows_for_sample:
                    final_rows.append(existing_rows_for_sample[model_name])
                    continue

                # Only calculate for models requested in this execution
                if model_name not in run_model_names:
                    continue

                # Check that the sample exists
                sample_data = samples_by_basename.get(sample_basename)
                if not sample_data:
                    continue

                # Find the configuration of the model to test
                to_test = next((t for t in stt_to_tests if t['name'] == model_name), None)
                if to_test is None:
                    continue

                print(f"Processing sample {sample_basename} - Testing model {model_name} ({len(sample_data['frames'])} frames)")
                
                # Measure response time
                start_time = time.time()
                transcription = None
                
                try:
                    # Use a custom script or standard STT service
                    if 'script' in to_test and to_test['script'] is not None:
                        # Custom script (e.g. external API)
                        transcription = to_test['script'](sample_data['audio_bytes'])
                    else:
                        # Standard STT service
                        stt_instance = to_test['stt']
                        transcription = await stt_node(sample_data['frames'], stt_instance)
                except Exception as e:
                    logger.error(f"Error testing {model_name} on {sample_basename}: {str(e)}")
                    transcription = None
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.setdefault(model_name, []).append(response_time)

                # Enregistrer le résultat
                final_rows.append({
                    'sample': sample_basename,
                    'model': model_name,
                    'response_time': f"{response_time:.2f}",
                    'transcription': transcription if transcription is not None else '',
                    'accuracy': 0,  # Will be calculated separately
                })

            final_rows_by_sample[sample_basename] = final_rows

        # Save all results to CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, sample_basename in enumerate(sample_order):
                for row in final_rows_by_sample.get(sample_basename, []):
                    writer.writerow(row)
                # Add empty line between samples for readability
                if idx < len(sample_order) - 1:
                    writer.writerow({})

        # Display performance statistics
        for model_name, times in response_times.items():
            average_time = sum(times) / len(times) if times else 0
            print(f"Average response time for {model_name}: {average_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(test_stt())