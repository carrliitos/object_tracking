import os
import sys
import time
from pathlib import Path

from utils import logger
from utils import context
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2

directory = context.get_context(os.path.abspath(__file__))
logger_file_name = Path(directory).stem
logger_name = Path(__file__).stem
logger_file = f"{directory}/logs/{logger_file_name}.log"
logger = logger.setup_logger(logger_name, logger_file)

def run(args):
  obj_tracker = object_tracker(args)
  # initialize the bounding box coordinates of the object we are going to track
  init_bounding_box = None
  vs = videostream(args)
  fps = None

  # loop over frames from video stream
  while True:
    # grab the current frame, then handle if we are using a VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check if we have reached the end of stream
    if frame is None:
      break

    # resize the frame and grab the frame dimensions
    frame = imutils.resize(frame, width=500)
    (height, width) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if init_bounding_box is not None:
      (success, box) = obj_tracker.update(frame)

      if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      fps.update()
      fps.stop()

      info = [
        ("Tracker", args["tracker"]),
        ("Success", "Yes" if success else "No"),
        ("FPS", "{:.2f}".format(fps.fps()))
      ]

      # loop over the info tuples and draw them on our frame
      for (i, (k, v)) in enumerate(info):
        cv2.putText(frame, f"{k}: {v}", (10, height - ((i * 20) + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding box to track
    if key == ord("s"):
      # select the bounding box of the object we want to track 
      # (make sure to press ENTER or SPACE after selecting the ROI)
      init_bounding_box = cv2.selectROI("Frame", 
                                        frame, 
                                        fromCenter=False, 
                                        showCrosshair=True)

      # start openCV object tracker using the supplied bounding box coordinates, 
      # then start the FPS throughput estimator as well
      obj_tracker.init(frame, init_bounding_box)
      fps = FPS().start()
    elif key == ord("q"):
      break

  if not args.get("video", False):
    vs.stop()
  else:
    vs.release()

  cv2.destroyAllWindows()

def object_tracker(args):
  (major, minor) = cv2.__version__.split(".")[:2]
  logger.info(f"Current selected object tracker: {args['tracker']}")
  if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
  else:
    OPENCV_OBJECT_TRACKERS = {
      "csrt": cv2.legacy.TrackerCSRT_create,
      "kcf": cv2.legacy.TrackerKCF_create,
      "boosting": cv2.legacy.TrackerBoosting_create,
      "mil": cv2.legacy.TrackerMIL_create,
      "tld": cv2.legacy.TrackerTLD_create,
      "medianflow": cv2.legacy.TrackerMedianFlow_create,
      "mosse": cv2.legacy.TrackerMOSSE_create
    }
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

  return tracker

def videostream(args):
  if not args.get("video", False):
    logger.info("starting video stream...")
    video_stream = VideoStream(src=0).start()
    time.sleep(1.0)
  else:
    video_stream = cv2.VideoCapture(args["video"])

  return video_stream