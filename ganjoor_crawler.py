import requests
import csv
import os
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from tqdm import tqdm
from mutagen.mp3 import MP3
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://ganjoor.net"
COLUMNS = ['poet', 'poemId', 'poemFullTitle', 'poemFullUrl', 'recitationId', 
           'audioArtist', 'audioTitle', 'audioDuration', 'audioSampleRate', 
           'audioNumChannels', 'audioBitrate' 'poem', 'poemTextPath', 'audioPath']

def get_all_poets():
    response = requests.get(BASE_URL)

    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract poem links - exclude other pages
    poet_links = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/') and href[1:].isalpha() and href not in [
            '/', '/random', '/map', '/mundex', '/hashieha', '/simi',
            '/about', '/vazn', '/sources', '/photos', '/faq',
            '/donate', '/contact', '/privacy']:
            poet_links.add(href)
    return list(sorted(poet_links))

def write_txt_to_path(text, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def open_txt_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def open_txt_file_as_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def get_audio_api(id, save_path):
    url = f'https://api.ganjoor.net/api/audio/file/{id}.mp3'
    save_path = os.path.join(save_path, f"{id}.mp3")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return f"Downloaded {id}.mp3"
        else:
            return f"Error: Unable to fetch audio for ID {id}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error downloading ID {id}: {e}"

def get_xml_api(id, save_path):
    url = f'https://api.ganjoor.net/api/audio/file/{id}.xml'
    save_path = os.path.join(save_path, f"{id}.xml")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return f"Downloaded {id}.xml"
        else:
            return f"Error: Unable to fetch audio for ID {id}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error downloading ID {id}: {e}"

def get_text_api(id):
    url = f'https://api.ganjoor.net/api/ganjoor/poem/{id}'
    response = requests.get(url)
    # Check if the response status is OK (200)
    if response.status_code == 200:
        return response.json()
    else:
        # If the request failed, return a failure message
        return f"Error: Unable to fetch poem. Status code: {response.status_code}"

def get_recitation_api(id):
    url = f'https://api.ganjoor.net/api/ganjoor/poem/{id}/recitations'
    response = requests.get(url)
    # Check if the response status is OK (200)
    if response.status_code == 200:
        return response.json()
    else:
        # If the request failed, return a failure message
        return f"Error: Unable to fetch poem. Status code: {response.status_code}"

def process_text_from_api(response, save_path):
    id = response['id']
    save_path = os.path.join(save_path, str(id)+'.txt')
    text = response['plainText']
    write_txt_to_path(text, save_path)

def has_recitations(id):
    """
    Check if the poem with the given ID has recitations.
    """
    url = f'https://api.ganjoor.net/api/ganjoor/poem/{id}/recitations'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:  # Check if recitations are present
            return data
    return None

def extract_poems_with_recitations(ids, output_file, max_workers=50):
    # Define the CSV header
    headers = ['poemId', 'poemFullTitle', 'poemFullUrl', 'audioTitle', 'audioArtist', 'recitationId']
    
    # Open the CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header row
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(has_recitations, id): id for id in ids}

            # Use tqdm to show progress
            for future in tqdm(as_completed(future_to_id), total=len(ids), desc="Processing Poem IDs", unit="poem"):
                id = future_to_id[future]
                try:
                    data = future.result()
                    if data:
                        for recitation in data:
                            # Extract recitation details
                            poemId = recitation['poemId']
                            poemFullTitle = recitation['poemFullTitle']
                            poemFullUrl = recitation['poemFullUrl']
                            audioTitle = recitation['audioTitle']
                            audioArtist = recitation['audioArtist']
                            recitationId = recitation['id']
                            
                            # Write recitation data as a row in the CSV
                            writer.writerow([poemId, poemFullTitle, poemFullUrl, audioTitle, audioArtist, recitationId])
                except Exception as e:
                    print(f"Error processing ID {id}: {e}")

def extract_text_from_ids(ids, output_file, max_workers=50):
    headers = ['poemId', 'poem']

    # Open the CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header row
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(get_text_api, id): id for id in ids}

            # Use tqdm to show progress
            for future in tqdm(as_completed(future_to_id), total=len(ids), desc="Processing Poem IDs", unit="poem"):
                id = future_to_id[future]
                try:
                    data = future.result()
                    if isinstance(data, dict) and 'plainText' in data:
                        poem = data['plainText']
                        # Write recitation data as a row in the CSV
                        writer.writerow([id, poem])
                except Exception as e:
                    print(f"Error processing ID {id}: {e}")

def extract_audios_from_ids(ids, save_directory, max_workers=50):
    """
    Downloads audio files for a list of IDs and saves them to a specified directory.
    
    Args:
        ids (List[str]): List of IDs for the audio files to download.
        save_directory (str): Directory to save the audio files.
        max_workers (int): Maximum number of threads for concurrent downloads.
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(get_audio_api, id, save_directory): id for id in ids}

        # Display progress with tqdm
        for future in tqdm(as_completed(future_to_id), total=len(ids), desc="Downloading Audio Files", unit="file"):
            id = future_to_id[future]
            try:
                result = future.result()
                print(result)  # Log the result of each download
            except Exception as e:
                print(f"Error processing ID {id}: {e}")
def extract_audio_metadata(file_path: str) -> dict:
    try:
        audio = MP3(file_path)
        info = audio.info
        bitrate = getattr(info, 'bitrate', None)
        return {
            'audioDuration': getattr(info, 'length', None),
            'audioSampleRate': getattr(info, 'sample_rate', None),  # Use getattr for robustness
            'audioNumChannels': getattr(info, 'channels', None),
            'audioBitrate': bitrate // 1000 if bitrate else None # convert to kbps
        }
    except Exception as e:
        print(f"Error extracting metadata for {file_path}: {e}")
        return {
            'audioDuration': None,
            'audioSampleRate': None,
            'audioNumChannels': None,
            'audioBitrate': None
        }

def generate_audio_metadata_csv(audio_files: list, output_csv: str):
    headers = ['recitationId', 'audioDuration', 'audioSampleRate', 'audioNumChannels', 'audioBitrate']

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header row

        # Extract metadata for each file
        for audio_file in tqdm(audio_files, desc="Processing Audio Files", unit="file"):
            recitation_id = os.path.splitext(os.path.basename(audio_file))[0]
            metadata = extract_audio_metadata(audio_file)
            writer.writerow([
                recitation_id,
                metadata['audioDuration'],
                metadata['audioSampleRate'],
                metadata['audioNumChannels'],
                metadata['audioBitrate']
            ])

def get_audio_files(directory: str) -> list:
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.mp3')
    ]