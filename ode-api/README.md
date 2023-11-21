# mars-api
Copy + fixes of https://github.com/samiriff/mars-ode-data-access
Allows API access to directly query and download satellite images of the marsian surface

Activate the venv inside ```codebase-v1/```:

```console
source ~/codebase-v1/venv/bin/activate
```

Configure the query parameters in `main.py` (documented within the file) and run it:

```console
python3 main.py
```

The `image_download.py` aims to automatically create a database of images by continiously sending requests for small chunks of the marsian surface. Downloaded images will (should) be skipped automatically. All images are split into smaller chunks and downloaded within a directory called `database` (please make sure that it doesnt gets added to the repo because of file sizes). 
