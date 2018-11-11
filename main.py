"""
    A Tensorflow implementation of Super SloMo

    Copyright (c) 2018, Windforces
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <ORGANIZATION> nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import tensorflow as tf
import numpy as np
import argparse
import sys

from src.LoadVideos import DataHandler
from model.VoxelFlow import VoxelFlow

FLAGS = None


def main():

    worker_hosts    = FLAGS.worker_hosts
    batch_size      = FLAGS.batch_size
    width           = FLAGS.width
    height          = FLAGS.height
    channel         = FLAGS.channel
    frame           = FLAGS.frame
    video_directory = FLAGS.directory
    multiplier = 1e-2
    learning_rate = 1e-4

    with tf.device('/device:CPU:0'):
        # data = DataHandler(video_directory=video_directory)
        with tf.name_scope("network"):
            model = VoxelFlow(width=width, height=height, channel=channel, multiplier=multiplier, scope="VoxelFlow")
            """
            coeff_1 = np.zeros(shape=(frame, batch_size, width, height, channel))
            coeff_2 = np.zeros(shape=(frame, batch_size, width, height, channel))
            coeff_3 = np.zeros(shape=(frame, batch_size, width, height, channel))
            coeff_4 = np.zeros(shape=(frame, batch_size, width, height, channel))

            for t in range(frame):
                oeff_1[t, :, :, :, :] = -(1 - t) * t
                coeff_2[t, :, :, :, :] = t * t
                coeff_3[t, :, :, :, :] = (1 - t) * (1 - t)
                coeff_4[t, :, :, :, :] = -t * (1 - t)    
            """
        with tf.name_scope("train"):
            total_loss = model.loss()
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer.minimize(loss=total_loss)

        with tf.name_scope("tensorboard"):
            tf.summary.scalar(name='total loss', tensor=total_loss)
            tf.summary.image(name='Input Image', tensor=model.input_placeholder)
            tf.summary.image(name='Output Image', tensor=model.inference())
            tf.summary.image(name='Target Image', tensor=model.target_placeholder)

        with tf.name_scope("miscellaneous"):
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()

        with 






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Hostname:port pairs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Default batch size"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="The width of the video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="The height of the video"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=3,
        help="The RGB channel of the video"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=240,
        help="The number of frame in a second in the video"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="",
        help="The location where the videos are located, use s3 or google bucket"
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
