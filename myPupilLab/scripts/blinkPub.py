#!/usr/bin/env python
import zmq
from msgpack import unpackb, packb, loads
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from vocus2_ros.msg import forDemo
from cv_bridge import CvBridge, CvBridgeError
import pyttsx3
from std_msgs.msg import Int64

currentId = -1
engine = pyttsx3.init()
engine.setProperty('rate', 125)
objectToCheck = ["","","cup"]

def currentObjCB(data):
	global currentId
	currentId = data.id
	print("Engine is about to say!")
	engine.say("current object is")
	engine.say("{}".format(objectToCheck[currentId]))
	engine.runAndWait()

if __name__ == "__main__":
	#pub_world_image = rospy.Publisher('world_Image',WorldCameraImage,queue_size=10)
	test_Img_Pub = rospy.Publisher('object_selected',Int64, queue_size=10)
	object_Sub = rospy.Subscriber('forDemo', forDemo, currentObjCB) 

	context = zmq.Context()
	# open a req port to talk to pupil
	addr = '127.0.0.1'  # remote ip or localhost
	req_port = "50020"  # same as in the pupil remote gui
	req = context.socket(zmq.REQ)
	req.setsockopt(zmq.LINGER, 0)
	req.connect("tcp://{}:{}".format(addr, req_port))

	# ask for the sub port
	req.send_string('SUB_PORT')
	sub_port = req.recv_string()
	rospy.loginfo("Starting Blink Confirmation publisher.")
	print("Starting Blink Confirmation publisher")
	rospy.init_node('blinkConfirmationPublisher')
	print("listening for socket message....")

	# open a sub port to listen to pupil
	sub = context.socket(zmq.SUB)
	sub.connect("tcp://{}:{}".format(addr, sub_port))

	# set subscriptions to topics
	# recv just pupil/gaze/notifications
	sub.setsockopt_string(zmq.SUBSCRIBE, "blinks")

	recent_world = None

	FRAME_FORMAT = 'bgr' #Image format, same as the one in Pupil Capture
	eyeCloseFlag = False
	startTime = 0
	endTime = 0	
	
#	rospy.spin()
	while not rospy.is_shutdown():
		#ret, frame = cap.read()
		# print("beforeTry")
		try:
			# print("loop")
			topic = sub.recv_string()
			msg = sub.recv()
			msg = loads(msg, raw=False)
			# print("\n{}: {}".format(topic,msg))
			if msg['type'] == 'onset' and not eyeCloseFlag:
				print('onset')
				eyeCloseFlag = True
				startTime = msg['timestamp']
			if msg['type'] == 'offset' and eyeCloseFlag:
				print('offset')
				eyeCloseFlag = False
				endTime = msg['timestamp']
				duration = endTime - startTime
				print(duration)
				print(currentId)
				if duration > 2.0 and duration < 4.0:
					if currentId >= 0:
						msg = forDemo()
						msg.id = currentId
						msg.header.stamp = rospy.Time.now()
						test_Img_Pub.publish(currentId)
						print('publish')
				else:
					endTime = 0
					startTime = 0
					duration = 0
				
		except KeyboardInterrupt:
			break
		
	print("Terminated")
	req.close()
	sub.close()
	context.term()


	
	

