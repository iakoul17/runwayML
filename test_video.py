"""Test pre-trained RGB model on a single video.

Date: 01/15/18
Authors: Bolei Zhou and Alex Andonian

This script accepts an mp4 video as the command line argument --video_file
and averages ResNet50 (trained on Moments) predictions on num_segment equally
spaced frames (extracted using ffmpeg).

Alternatively, one may instead provide the path to a directory containing
video frames saved as jpgs, which are sorted and forwarded through the model.

ResNet50 trained on Moments is used to predict the action for each frame,
and these class probabilities are average to produce a video-level predction.

Optionally, one can generate a new video --rendered_output from the frames
used to make the prediction with the predicted category in the top-left corner.

"""

import os
import runway
from runway.data_types import number, text, image
from example_model import ExampleModel
#import moviepy.editor as mpy

import torch.optim
import torch.nn.parallel
from torch.nn import functional as F

from pytube import YouTube #for youtube videos
import cv2     # for capturing videos
import math   # for mathematical operations
import numpy as np
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from PIL import Image, ImageStat


import models
from utils import extract_frames, load_frames, render_frames


# options
videoPath = './tmp'
imgPath = './tmp/frames'
os.makedirs(videoPath, exist_ok=True)
os.makedirs(imgPath, exist_ok=True)
arch = 'resnet3d50'
start = 10
end = 30
video_hash = 'Y6m6DYJ7RW8'

# Load model
@runway.setup
def setup():
  return models.load_model(arch)

# Get dataset categories
categories = models.load_categories()

# Load the video frame transform
transform = models.load_transform()


@runway.command('classify', inputs={ 'video': text() }, outputs={ 'label': text() })
def classify(model, input):
    yt = YouTube('https://youtube.com/embed/%s?start=%d&end=%d' % (input['video'], start, end))
    video = yt.streams.all()[0]
    video_file = video.download(videoPath)
    num_segments = 16


    print('Extracting frames using ffmpeg...')
    frames = extract_frames(video_file, num_segments)


    # Prepare input tensor

   
    input = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)

    # Make video prediction
    with torch.no_grad():
       logits = model(input)
       h_x = F.softmax(logits, 1).mean(dim=0)
       probs, idx = h_x.sort(0, True)

    # Output the prediction.
    return categories[idx[0]]

    """ print('RESULT ON ' + video_file)

    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], categories[idx[i]])) """

# Render output frames with prediction text.

""" prediction = categories[idx[0]]
rendered_frames = render_frames(frames, prediction)
clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
clip.write_videofile(args.rendered_output) """
