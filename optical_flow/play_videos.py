from skvideo.io import vread
import matplotlib.pyplot as plt
import numpy

numpy.float = numpy.float64
numpy.int = numpy.int_

if __name__ == "__main__":
    frame_rate = 1 / 16
    which_video = "./video2.mp4"
    videodata = vread(which_video)

    for _ in range(5):

        for i in range(videodata.shape[0]):
            frame = videodata[i, ...]

            plt.cla()
            plt.imshow(frame)
            plt.pause(frame_rate)

    plt.close("all")
