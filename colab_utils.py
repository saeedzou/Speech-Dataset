from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os

# Authenticate and initialize the pydrive2 client
def initialize_drive():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return GoogleDrive(gauth)

# Function to create a folder on Google Drive
def create_folder(drive, parent_folder_id, subfolder_name):
    new_folder = drive.CreateFile({
        'title': subfolder_name,
        'parents': [{'id': parent_folder_id}],
        'mimeType': 'application/vnd.google-apps.folder'
    })
    new_folder.Upload()
    return new_folder

# Function to get or create a folder
def get_or_create_folder(drive, parent_folder_id, folder_name):
    query = f"title = '{folder_name}' and '{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
    folder_list = drive.ListFile({'q': query}).GetList()

    if len(folder_list) == 0:
        print(f"Creating {folder_name} folder...")
        new_folder = create_folder(drive, parent_folder_id, folder_name)
        return new_folder['id']
    else:
        print(f"{folder_name} folder already exists.")
        return folder_list[0]['id']

# Function to upload a file to Google Drive
def upload_file_to_drive(drive, parent_folder_id, file_path, file_name):
    gfile = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': parent_folder_id}]
    })
    gfile.SetContentFile(file_path)
    gfile.Upload()
    print(f"Uploaded {file_name} to folder ID {parent_folder_id}.")
    return gfile['id']