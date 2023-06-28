import argparse
from typing import Union, List

from config import config

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

import cv2
from PIL import Image
import clip
import torch
import math
import numpy as np
# import plotly.express as px
# import datetime
# from IPython.core.display import HTML
# from IPython.core.display_functions import display


class Clipsearch(ClamsApp):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.fps: float = 0.0
        self.debug = True

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def extract_frames(self, video_filename, **kwargs) -> List[Image.Image]:
        """
        Extracts every N-th frame of the video
        :return: List of PIL images for CLIP
        """
        # The frame images will be stored in video_frames
        video_frames = []

        # Open the video file
        capture = cv2.VideoCapture(video_filename)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        if self.debug:
            print(f"fps: {self.fps}")

        current_frame = 0
        while capture.isOpened():
            # Read the current frame
            ret, frame = capture.read()

            # Convert it to a PIL image (required for CLIP) and store it
            if ret:
                video_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break

            # Skip N frames
            current_frame += kwargs.get("sampleRatio")
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            if self.debug and len(video_frames) > 900:
                break

        # Print some statistics
        print(f"Frames extracted: {len(video_frames)}")
        return video_frames

    def encode_frames(self, video_filename, **kwargs):
        # You can try tuning the batch size for very large videos, but it should usually be OK
        batch_size = 256
        video_frames = self.extract_frames(video_filename, **kwargs)
        batches = math.ceil(len(video_frames) / batch_size)

        # The encoded features will bs stored in video_features
        video_features = torch.empty([0, 512], dtype=torch.float16).to(self.device)

        # Process each batch
        for i in range(batches):
            print(f"Processing batch {i + 1}/{batches}")

            # Get the relevant frames
            batch_frames = video_frames[i * batch_size: (i + 1) * batch_size]

            # Preprocess the images for the batch
            batch_preprocessed = torch.stack([self.preprocess(frame) for frame in batch_frames]).to(self.device)

            # Encode with CLIP and normalize
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)

            # Append the batch to the list containing all features
            video_features = torch.cat((video_features, batch_features))

        # Print some stats
        print(f"Features: {video_features.shape}")
        return video_features, video_frames

    def frame_to_time(self, frame_number):
        return frame_number / self.fps

    def search_video(self, video_filename, **kwargs):
        # Check if "query" is in kwargs, and it's a string
        if "query" not in kwargs or not isinstance(kwargs["query"], str):
            raise ValueError('Invalid query')

        queries = kwargs.get("query").split('*')
        queries = [query.replace('+', ' ') for query in queries]

        threshold = .30 if "threshold" not in kwargs else float(kwargs["threshold"])
        video_features, video_frames = self.encode_frames(video_filename, **kwargs)

        all_timeframes = []

        # Encode and normalize each search query using CLIP then search
        for query in queries:
            with torch.no_grad():
                text_features = self.model.encode_text(clip.tokenize(query).to(self.device))
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute the similarity between the search query and each frame using the Cosine similarity
            similarities = (video_features @ text_features.T).squeeze().cpu().numpy()

            if self.debug:
                print(similarities)

            # Find the frames that meet the threshold
            above_threshold_indices = np.where(similarities > threshold)[0]

            timeframes = []

            if len(above_threshold_indices) > 0:
                # Find the contiguous regions of frames above the threshold
                contiguous_regions = np.split(above_threshold_indices,
                                              np.where(np.diff(above_threshold_indices) != 1)[0] + 1)

                # For each contiguous region find the start and end times
                for region in contiguous_regions:
                    start_frame, end_frame = region[0], region[-1]
                    start_time = self.frame_to_time(start_frame)
                    end_time = self.frame_to_time(end_frame)

                    timeframe = {'query': query, 'start': start_time, 'end': end_time, 'start_frame': start_frame,
                                 'end_frame': end_frame}
                    timeframes.append(timeframe)

            all_timeframes.extend(timeframes)
        return all_timeframes

    def _annotate(self, mmif: Union[str, dict, Mmif], **kwargs) -> Mmif:
        # load file location from mmif
        video_filename = mmif.get_document_location(DocumentTypes.VideoDocument)[7:]
        if self.debug:
            print(f"debugging with {video_filename}")
        # config = self.get_configuration(**kwargs)
        unit = kwargs.get("timeUnit")
        new_view: View = mmif.new_view()
        self.sign_view(new_view, config)
        new_view.new_contain(
            AnnotationTypes.TimeFrame,
            timeUnit=unit,
            document=mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0].id,
        )
        timeframes = self.search_video(video_filename, **kwargs)

        if unit == "milliseconds":
            for timeframe in timeframes:
                timeframe_annotation: Annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
                timeframe_annotation.add_property("start", timeframe["start"])
                timeframe_annotation.add_property("end", timeframe["end"])
                timeframe_annotation.add_property("unit", unit)
                timeframe_annotation.add_property("query", timeframe["query"])
        else:
            for timeframe in timeframes:
                timeframe_annotation: Annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
                timeframe_annotation.add_property("start", int(timeframe["start_frame"]))
                timeframe_annotation.add_property("end", int(timeframe["end_frame"]))
                timeframe_annotation.add_property("unit", "frames")
                timeframe_annotation.add_property("query", timeframe["query"])
        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = Clipsearch()

    http_app = Restifier(app, port=int(parsed_args.port)
                         )
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
