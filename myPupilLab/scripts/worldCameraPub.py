#!/usr/bin/env python
import zmq
from msgpack import unpackb, packb
import rospy
import numpy as np
from myPupilLab.msg import WorldCameraImage
import cv2
#from PIL import Image
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String

context = zmq.Context()
# open a req port to talk to pupil
addr = '127.0.0.1'  # remote ip or localhost
req_port = "50020"  # same as in the pupil remote gui
req = context.socket(zmq.REQ)
req.connect("tcp://{}:{}".format(addr, req_port))

# ask for the sub port
req.send_string('SUB_PORT')
sub_port = req.recv_string()

# send notification:
def notify(notification):
	"""Sends ``notification`` to Pupil Remote"""
	topic = 'notify.' + notification['subject']
	payload = packb(notification, use_bin_type=True)
	req.send_string(topic, flags=zmq.SNDMORE)
	req.send(payload)
	return req.recv_string()

# open a sub port to listen to pupil
sub = context.socket(zmq.SUB)
sub.connect("tcp://{}:{}".format(addr, sub_port))

# set subscriptions to topics
# recv just pupil/gaze/notifications
sub.setsockopt_string(zmq.SUBSCRIBE, 'frame.')

#pub_world_image = rospy.Publisher('world_Image',WorldCameraImage,queue_size=10)
test_Img_Pub = rospy.Publisher('/camera/rgb/image_raw',Image, queue_size=10)
def recv_from_sub():
	'''Recv a message with topic, payload.
	Topic is a utf-8 encoded string. Returned as unicode object.
	Payload is a msgpack serialized dict. Returned as a python dict.
	Any addional message frames will be added as a list
	in the payload dict with key: '__raw_data__' .
	'''
	topic = sub.recv_string()
	payload = unpackb(sub.recv(), encoding='utf-8')
	extra_frames = []
	while sub.get(zmq.RCVMORE):
		extra_frames.append(sub.recv())
	if extra_frames:
		payload['__raw_data__'] = extra_frames
	return topic, payload

recent_world = None

FRAME_FORMAT = 'bgr' #Image format, same as the one in Pupil Capture

notify({'subject': 'frame_publishing.set_format', 'format': FRAME_FORMAT})

def world_Image_parser(message):
	#Parse gazeInfo into a message.
	if len(message) < 1: return -1 
	#bridge = CvBridge()

	outmsg = WorldCameraImage()

	outmsg.arr = message #uint8 arrays
	#print(message.type)
	
	#h,w,l = message.shape
	#opencv_format = Image.fromarray(message.astype(np.uint8)) #Convert np array  to cv::Mat
	#test_Img_Pub.publish(bridge.cv2_to_imgmsg(message), "bgr8")

	return outmsg 


if __name__ == "__main__":
	rospy.loginfo("Starting World Image publisher.")
	print("Starting World Image publisher")
	rospy.init_node('worldImagePublisher')
	print("listening for socket message....")
	bridge = CvBridge()
	#cap = cv2.VideoCapture(0)

	while not rospy.is_shutdown():
		#ret, frame = cap.read()
		topic, msg = recv_from_sub()
		if topic.startswith('frame.') and msg['format'] != FRAME_FORMAT:
			print(f"different frame format ({msg['format']}); skipping frame from {topic}")
			continue
		
		if topic == 'frame.world':
			recent_world = np.frombuffer(msg['__raw_data__'][0], dtype=np.uint8).reshape(msg['height'], msg['width'], 3)
			outmsg = world_Image_parser(recent_world)
			#if not outmsg == -1:
				# rospy.loginfo(outmsg) 
				# pub_world_image.publish(outmsg)
				# cv2.imshow("world", recent_world)
				# cv2.waitKey(1)
				# pass
			try:
				print("Publish Image to /camera/rgb/image_raw")
				image_msg = bridge.cv2_to_imgmsg(recent_world)
				image_msg.header.stamp = rospy.Time.now()
				test_Img_Pub.publish(image_msg)
			except CvBridgeError as e:
				print(e)
	

