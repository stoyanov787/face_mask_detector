"""This module detects if person is with mask or without mask in real time"""

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


class Coordinates:
    """Coordinates of start x,y and end x,y"""

    def __init__(self, box):
        self.start_x, self.start_y, self.end_x, self.end_y = box.astype("int")

    def ensure(self, width, height):
        """Ensure that the bounding boxes fit in the frame's dimensions"""

        self.start_x, self.start_y = (max(0, self.start_x), max(0, self.start_y))
        self.end_x, self.end_y = (min(width - 1, self.end_x), min(height - 1, self.end_y))


class Face(Coordinates):
    """Extract and convert the face"""

    def __init__(self, box, width, height):
        super().__init__(box)
        super().ensure(width, height)

    def convert(self, frame):
        """Extract the face ROI, convert the face color and size,
         convert it to array and preprocess it"""

        # extract the face ROI
        face = frame[self.start_y:self.end_y, self.start_x:self.end_x]

        # convert it from BGR to RGB channel ordering
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        return face


def detect_and_predict_mask(frame, face, mask):
    """Detect the faces and predict if they're with or without mask

    :param frame: The current frame of the video
    :type frame: numpy.ndarray
    :param face: Pre-built face detector model
    :type face: cv2.dnn_Net
    :param mask: The face mask recognition model
    :type mask: keras.engine.functional.Functional
    :returns: locations and predictions
    """

    # construct a blob from dimensions of the frame
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # obtain the face detections
    face.setInput(blob)
    detections = face.forward()

    faces = []
    locations = []
    predictions = []

    # loop over the detections
    for i in range(detections.shape[2]):
        # extract the probability associated with the detection
        probability = detections[0, 0, i, 2]

        # ensuring that the probability is greater than the minimum probability
        if probability > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            face = Face(box, width, height)

            # add the face and bounding boxes to their lists
            locations.append((face.start_x, face.start_y, face.end_x, face.end_y))
            face = face.convert(frame)
            faces.append(face)

    # only make a predictions if at least one face is detected
    if len(faces) > 0:
        # make batch predictions on all faces at the same time in the above `for` loop
        faces = np.array(faces, dtype='float32')
        predictions = mask.predict(faces, batch_size=32)

    return locations, predictions


def start_video():
    """Start real time video and detect if person is with or without mask"""

    # load the pre-built face detector models
    deploy = r'deploy.prototxt'
    width = r'width.caffemodel'
    face = cv2.dnn.readNet(deploy, width)

    # load the face mask detector model
    mask = load_model(r'mask_recognition.h5')

    # start the video stream
    vs = VideoStream(src=0).start()

    # loop over the frames from the video stream
    while True:
        # open the frame from the video stream and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect if the face is with or without mask
        locations, predictions = detect_and_predict_mask(frame, face, mask)

        # loop over the detected face locations
        for box, prediction in zip(locations, predictions):
            # unpack the bounding box and predictions
            start_x, start_y, end_x, end_y = box
            with_mask, without_mask = prediction

            # determine the label and color to draw the bounding box and text
            label = 'Mask' if with_mask > without_mask else 'No Mask'
            color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

            # include the probability in the label
            label = f'{label}: {(max(with_mask, without_mask) * 100):.2f}%'

            # display the label and bounding box rectangle on the output
            cv2.putText(frame, label, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        # show the output frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q`, 'Q' or Esc key is pressed, break from the loop
        if key == ord('q') or key == ord('Q') or key == ord(''):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
