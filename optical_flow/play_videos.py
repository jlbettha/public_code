import matplotlib.pyplot as plt

# import skvideo
from skvideo.io import vread

# TODO: skvideo.setFFmpegPath("/.venv/bin/ffmpeg")  # Adjust path as needed
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
