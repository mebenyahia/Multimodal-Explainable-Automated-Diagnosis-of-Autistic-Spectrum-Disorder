pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pandas

from __future__ import print_function
import os
import io
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Authenticate and create the Google Drive API client
def authenticate_gdrive():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service

# Get the list of files in a Google Drive folder
def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    return items

# Download a file from Google Drive
def download_file(service, file_id, file_name, destination_folder):
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(destination_folder, file_name)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

# Main function
def main():
    # Authenticate and create the API client
    service = authenticate_gdrive()

    # ID of the folder containing the files on Google Drive
    folder_id = 'your-folder-id-here'

    # Local directory to save the downloaded files
    destination_folder = os.path.join(os.getcwd(), 'data', 'fmri')
    os.makedirs(destination_folder, exist_ok=True)

    # Read the CSV file with SUB_IDs
    file_path = '/mnt/data/Phenotypic_V1_0b_preprocessed1.csv'
    data = pd.read_csv(file_path)
    sub_ids = data['SUB_ID'].astype(str).tolist()

    # List all files in the folder
    files = list_files_in_folder(service, folder_id)

    # Download files that match the SUB_IDs
    for file in files:
        for sub_id in sub_ids:
            if sub_id in file['name']:
                print(f"Downloading {file['name']}...")
                download_file(service, file['id'], file['name'], destination_folder)

if __name__ == '__main__':
    main()
