import sys
import time
import cv2
import utils
import pydirectinput
import multiprocessing
import threading
from multiprocessing import Pipe

from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet


def leave():
	pass

def leg_glance():
	pydirectinput.press('down')
	pydirectinput.press('d')
	pydirectinput.press('down')
	pydirectinput.press('a')
	pydirectinput.press('down')
	pydirectinput.press('d')
	pydirectinput.press('down')

def cover_drive():
	pydirectinput.press('down')
	pydirectinput.press('a')
	pydirectinput.press('down')
	pydirectinput.press('d')
	pydirectinput.press('down')
	pydirectinput.press('a')
	pydirectinput.press('down')

def check_pose(p1_input):
	estimation_model = 'movenet_lightning'  # ['posenet', 'movenet_lightning', 'movenet_thunder', 'movenet_multipose']
	tracker_type = 'bounding_box'  # ['keypoint', 'bounding_box']
	classification_model = '<your-tflite-model>'
	label_file = '<your-label-file>'
	camera_id = 0
	width = 800
	height = 800

	# Notify users that tracker is only enabled for MoveNet MultiPose model.
	if tracker_type and (estimation_model != 'movenet_lightning'):
		print("Tracker can only be used for MoveNet Lightning model.")

	# Initialize the pose estimator selected.
	if estimation_model in ['movenet_lightning', 'movenet_thunder']:
		pose_detector = Movenet(estimation_model)
		print("MoveNet Lightning model selected.")
	elif estimation_model == 'posenet':
		pose_detector = Posenet(estimation_model)
		print("PoseNet model selected.")
	elif estimation_model == 'movenet_multipose':
		print("MoveNet MultiPose model selected.")
		pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
	else:
		sys.exit("Error: Model not supported.")

	# Variables to calculate FPS.
	counter, fps = 0, 0
	start_time = time.time()

	# Start capturing video input from the camera.
	cap = cv2.VideoCapture(camera_id)

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	# Visualization parameters.
	row_size = 20  # pixels
	left_margin = 24  # pixels
	text_color = (255, 0, 0)  # Blue
	font_size = 1
	font_thickness = 1
	max_detection_results = 3
	fps_avg_frame_count = 10

	# Initialize the classification model.
	if classification_model:
		classifier = Classifier(classification_model, label_file)
		detection_results_to_show = min(max_detection_results, len(classifier.pose_class_names))

	# Continuously capture images from the camera.
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Unable to open camera. Please check your camera settings.")
			sys.exit()

		counter += 1
		image = cv2.flip(image, 1)

		if estimation_model == 'movenet_multipose':
			# Run pose estimation using a MultiPose model.
			list_persons = pose_detector.detect(image)
		else:
			# Run pose estimation using a SinglePose model, and wrap the result in an array.
			list_persons = [pose_detector.detect(image)]

		# Draw keypoints and edges on input image.	
		image = utils.visualize(image, list_persons)

		if classification_model:
			# Run pose classification.
			prob_list = classifier.classify_pose(list_persons[0])

			# Show classification results on the image.
			for i in range(detection_results_to_show):
				class_name = prob_list[i].label
				probability = round(prob_list[i].score, 2)
				result_text = class_name + ' (' + str(probability) + ')'
				text_location = (left_margin, (i + 2) * row_size)

				if class_name == 'leave':
					cv2.putText(image, 'Leave', (75,50), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
				elif class_name == 'cover_drive':
					cv2.putText(image, 'Cover Drive', (75,50), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)
				else:
					cv2.putText(image, 'Leg Glance', (75,50), cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness)

				
		# Calculate the FPS.		
		if counter % fps_avg_frame_count == 0:
			end_time = time.time()
			fps = fps_avg_frame_count / (end_time - start_time)
			start_time = time.time()

		# Show the FPS.
		fps_text = 'FPS = ' + str(int(fps))
		text_location = (left_margin, row_size)

		# Stop the program if the ESC key is pressed.
		if cv2.waitKey(1) == 27:
			break

		cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Pose Classification', 670, 600)
		cv2.imshow('Pose Classification', image)
		cv2.moveWindow('Pose Classification', 0, 0)
		p1_input.send([class_name, probability*100])	
			
	cap.release()
	cv2.destroyAllWindows()

def fetch_class(p1_output):
	while True:
		value = p1_output.recv()
		pose = value[0]
		probability = value[1]
		
		if pose == 'cover_drive':
			pydirectinput.keyDown('ctrl')
			cover_drive()
		elif pose == 'leg_glance':
			pydirectinput.keyDown('ctrl')
			leg_glance()
		else:
			leave()

if __name__ == '__main__':
	p1_output, p1_input = Pipe()

	p1 = threading.Thread(target=check_pose, args=(p1_input, ))
	p2 = threading.Thread(target=fetch_class, args = (p1_output, ))

	p1.start()
	pydirectinput.click(1555, 300)
	time.sleep(15)
	p2.start()