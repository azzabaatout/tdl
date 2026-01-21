import urllib.request
import zipfile
import os
import tempfile
import sys
import pathlib

#
# Run this script. This will download the required dataset 
#


url = "https://syncandshare.lrz.de/dl/fiDGmQHJBu8282XT6E48jv/in.zip"
extract_to = pathlib.Path().resolve()


with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp_file.name)
    
    if extract_to is None:
        extract_to = os.getcwd()
    
    print(f"Extracting to {extract_to}")
    with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.unlink(tmp_file.name)
    print("Done")

