import numpy as np
import cv2

import sys, os, logging
import random

class DataHandler:

    def __init__(self, video_directory):
        # the directory where the videos are located
        self.directory = video_directory
        self.extension = [".mov", ".mp4", ".m4v", ".MOV"]
        # read the video list
        self._read()

    def _read(self):
        # search the directory
        self.video_list = os.listdir(self.directory)

        if len(self.video_list) == 0:
            logging.error("No video files in the directory")
        else:
            logging.info("Successfully loaded the video list")

        self.video_list = [file for file in self.video_list if file.endswith(self.extension[0]) \
                    or file.endswith(self.extension[1]) \
                    or file.endswith(self.extension[2]) \
                    or file.endswith(self.extension[3])]

    def play(self):
        # do not read any files other than video files
        random_file = random.sample(self.video_list, 1)
        cap = cv2.VideoCapture(os.path.join(self.directory, random_file[0]))

        while cap.isOpened():
            ret, frame = cap.read()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
