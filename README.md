# app-clipsearch

This repository provides a wrapper for using CLIP embeddings to search for video content with text prompts

## Requirements

Generally, an CLAMS app requires 

- Python3 with the `clams-python` module installed; to run the app locally. 
- `docker`; to run the app in a Docker container (as a HTTP server).
- A HTTP client utility (such as `curl`); to invoke and execute analysis.

## Building and running the Docker image

From the project directory, run the following in your terminal to build the Docker image from the included Dockerfile:

```bash
docker build . -f Dockerfile -t <app_name>
```

Alternatively, the app maybe already be available on docker hub. 

``` bash 
docker pull <app_name>
```

Then to create a Docker container using that image, run:

```bash
docker run -v /path/to/data/directory:/data -p <port>:5000 <app_name>
```

where /path/to/data/directory is the location of your media files or MMIF objects and <port> is the *host* port number you want your container to be listening to. The HTTP inside the container will be listening to 5000 by default. 

## Invoking the app
Once the app is running as a HTTP server, to invoke the app and get automatic annotations, simply send a POST request to the app with a MMIF input as request body.

MMIF input files can be obtained from outputs of other CLAMS apps, or you can create an empty MMIF only with source media locations using `clams source` command. See the help message for a more detailed instructions. 

```bash
clams source --help
```

(Make sure you installed the same `clams-python` package version specified in the [`requirements.txt`](requirements.txt).)

