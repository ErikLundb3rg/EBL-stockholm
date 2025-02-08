import os
from time import sleep
import json
import requests

def make_request(url, headers, method="GET", data=None, files=None):
    if method == "POST":
        response = requests.post(url, headers=headers, json=data, files=files)
    else:
        response = requests.get(url, headers=headers)
    return response.json()

def get_files(file_path):
    if not os.path.exists(file_path):  # This is here to check if the file exists
        return {"error": "video_file_path doesn't exist"}

    _file_name, file_extension = os.path.splitext(
        file_path
    )  # Get your audio file name + extension

    with open(file_path, "rb") as f:  # Open the file
        file_content = f.read()  # Read the content of the file

    files = [("audio", (file_path, file_content, "audio/" + file_extension[1:]))]

    return files

def transcribe(video_file_path, timeout=1000):
    files = get_files(file_path=video_file_path)
    
    API_KEY = os.getenv("GLADIA_API_KEY", "")
    if not API_KEY:
        return {"error": "Gladia API key missing"}
    
    headers = {
        "x-gladia-key": API_KEY,  # Replace with your Gladia Token
        "accept": "application/json",
    }

    # Uploading files to Gladia
    upload_response = make_request(
        "https://api.gladia.io/v2/upload/", headers, "POST", files=files
    )

    # Upload response with File ID: upload_response
    audio_url = upload_response.get("audio_url")

    data = {
        "audio_url": audio_url,
        "diarization": True,
        "sentences": True
    }

    headers["Content-Type"] = "application/json"

    # Sending request to Gladia API
    post_response = make_request(
        "https://api.gladia.io/v2/transcription/", headers, "POST", data=data
    )

    # Post response with Transcription ID: post_response
    result_url = post_response.get("result_url")

    if result_url:
        for _ in range(timeout):
            print("Polling for results...")
            poll_response = make_request(result_url, headers)

            if poll_response.get("status") == "done":
                # Transcription done
                return poll_response.get("result")
            elif poll_response.get("status") == "error":
                # Transcription failed
                return {
                    "error": f"Transcription failed: {poll_response}",
                    "error_code": 400
                }
            else:
                # Transcription status
                _current_status = poll_response.get("status")
            sleep(1)
    return {"error": "Transcription timed out"}
