# Copyright 2017 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for downloading files, downloading files from google drive, uncompressing targz"""

from __future__ import print_function
import os
import cgi
import tarfile
import gzip
import shutil
import zipfile
try:
    from urllib import request
    from http import cookies, cookiejar
except ImportError:
    # Fall back to Python 2
    import urllib2 as request
    import Cookie as cookies
    import cookielib as cookiejar


def download(directory=".", url=None, google_drive_fileid=None, extract_targz=False, extract_gz=False, extract_zip=False, file_name=None):
    """Downloads a file from provided URL or file id at google drive"""

    if url is None and google_drive_fileid is not None:
        url = "https://drive.google.com/uc?export=download&id=" + google_drive_fileid
        cj = cookiejar.CookieJar()
        opener = request.build_opener(request.HTTPCookieProcessor(cj))
        u = opener.open(url)
        cookie = cookies.SimpleCookie()
        cookie.load(u.info().get("set-cookie"))
        token = ""
        for key, value in cookie.items():
            if key.startswith('download_warning'):
                token = value.value
        url += "&confirm=" + token
        u = opener.open(url)
    else:
        u = request.urlopen(url)

    meta = u.info()

    if file_name is None:
        cd = meta.get("content-disposition")
        if cd is not None:  
            value, params = cgi.parse_header(cd)
            cd_file = params['filename']
            if cd_file is not None:
                file_name = cd_file

    if file_name is None:
        file_name = url.split('/')[-1]

    file_path = os.path.join(directory, file_name)

    file_size = 0
    length_header = meta.get("Content-Length")
    if length_header is not None:
        file_size = int(length_header)
        print("Downloading: %s Bytes: %d" % (file_name, file_size))
    else:
        print("Downloading: %s" % file_name)

    if os.path.exists(file_path) and (os.path.getsize(file_path) == file_size or file_size == 0):
        print("File %s already exists, skipping" % file_path)
        return

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as file:
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            file.write(buffer)
            if file_size > 0:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            else:
                status = r"%10d" % file_size_dl
            status += chr(8)*(len(status)+1)
            print(status, end='')

        print()

    if extract_targz:
        print("Extracting...")
        tarfile.open(name=file_path, mode="r:gz").extractall(directory)
        
    if extract_gz:
        file_out_path = file_path.replace('.gz','')
            
        print("Extracting...")
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if extract_zip:
        print("Extracting...")
        zipfile.ZipFile(file_path, 'r').extractall(directory)
    print("Done")
