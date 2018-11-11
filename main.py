import tensorflow as tf
import numpy as np

from src.LoadVideos import DataHandler
from model.FlowNetwork import FlowNetwork

def main():
    batch_size = 32
    frame = 240
    width = 1280
    height = 720
    channel = 3
    video_directory = "/Users/jeasungpark/PycharmProjects/Super SloMo/data/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos/original_high_fps_videos"

    data = DataHandler(video_directory=video_directory)
    model = FlowNetwork(frame=frame, width=width, height=height, channel=channel, scope="flow")

    coeff_1 = np.zeros(shape=(frame, batch_size, width, height, channel))
    coeff_2 = np.zeros(shape=(frame, batch_size, width, height, channel))
    coeff_3 = np.zeros(shape=(frame, batch_size, width, height, channel))
    coeff_4 = np.zeros(shape=(frame, batch_size, width, height, channel))

    for t in range(frame):
        coeff_1[t, :, :, :, :] = -(1 - t) * t
        coeff_2[t, :, :, :, :] = t * t
        coeff_3[t, :, :, :, :] = (1 - t) * (1 - t)
        coeff_4[t, :, :, :, :] = -t * (1 - t)



if __name__ == '__main__':
    main()