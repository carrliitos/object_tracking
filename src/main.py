import argparse
import object_tracker

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
  ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
  args = vars(ap.parse_args())

  object_tracker.run(args)
