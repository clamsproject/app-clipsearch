# Use the same base image version as the clams-python python library version
FROM ghcr.io/clamsproject/clams-python-opencv4-torch2:1.0.3
# See https://github.com/orgs/clamsproject/packages?tab=packages&q=clams-python for more base images
# IF you want to automatically publish this image to the clamsproject organization, 
# 1. you should have generated this template without --no-github-actions flag
# 1. to add arm64 support, change relevant line in .github/workflows/container.yml 
#     * NOTE that a lots of software doesn't install/compile or run on arm64 architecture out of the box 
#     * make sure you locally test the compatibility of all software dependencies before using arm64 support 
# 1. use a git tag to trigger the github action. You need to use git tag to properly set app version anyway

################################################################################
# DO NOT EDIT THIS SECTION
ARG CLAMS_APP_VERSION
ENV CLAMS_APP_VERSION ${CLAMS_APP_VERSION}
################################################################################

################################################################################
# clams-python base images are based on debian distro
# install more system packages as needed using the apt manager
################################################################################

################################################################################
# main app installation
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ADD https://github.com/openai/CLIP/archive/a9b1bf5920416aaeaec965c25dd9e8f98c864f16.tar.gz /clip.tar.gz
RUN tar -x -z -f /clip.tar.gz -C /
RUN pip install /CLIP-a9b1bf5920416aaeaec965c25dd9e8f98c864f16

# default command to run the CLAMS app in a production server 
CMD ["python3", "app.py", "--production"]
################################################################################
