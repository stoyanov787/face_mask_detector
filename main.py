"""Face mask detector in real time"""

import train_model
import live_video
import os


def main():
    """Create the model and use it in real time video"""

    if os.path.exists('mask_recognition.h5'):
        live_video.start_video()
    else:
        train_model.run()
        live_video.start_video()


if __name__ == '__main__':
    main()
