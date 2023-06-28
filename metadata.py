"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Clip Search",
        description="",  # briefly describe what the purpose and features of the app
        app_license="MIT",  # short name for a software license like MIT, Apache2, GPL, etc.
        identifier="clipsearch",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format. 
        url="https://fakegithub.com/some/repository",  # a website where the source code and full documentation of the app is hosted, if you are on the CLAMS team, see ``.github/README.md`` file in this directory.
    )
    # and then add I/O specifications: an app must have at least one input and ont output
    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame, frameType='string')
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(name='a_param', description='example parameter description',
                           type='boolean', default='false')
    # metadta.add_parameter(more...)
    
    # CHANGE this line and make sure return the compiled `metadata` instance
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(appmetadata().jsonify(pretty=True))
