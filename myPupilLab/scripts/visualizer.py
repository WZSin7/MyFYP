#!/usr/bin/env python3
import rospy
from myPupilLab.msg import GazeInfoBino_Array
import message_filters
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


pub_result = rospy.Publisher('visualizer',Image,queue_size=1)

def callback(gaze, image):
	rospy.loginfo("Success!")		
	normalized_X = np.sum(gaze.x)/len(gaze.x)
	normalized_Y = np.sum(gaze.y)/len(gaze.y)

	actual_X = int(normalized_X*1280-1)
	actual_Y = int(((1-normalized_Y)*720)-1)

	raw_x_array = []
	raw_y_array = []
	for i in range(len(gaze.x)):
		raw_x_array.append(int((gaze.x[i] * 1280)-1))
		raw_y_array.append(int(((1-gaze.y[i])*720)-1))

	bridge = CvBridge()
	np_array = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
	cv.line(np_array,(0,actual_Y),(1279,actual_Y),(255,0,0),3)
	cv.line(np_array,(actual_X,0),(actual_X,719),(255,0,0),3)

	for i in range(len(raw_x_array)):
		cv.circle(np_array,(raw_x_array[i],raw_y_array[i]),1,(0,255,0),-1)

	image_message = bridge.cv2_to_imgmsg(np_array, encoding="passthrough")
	pub_result.publish(image_message)



def visualizer():
	rospy.init_node('visualizer', anonymous=True)

	# gaze_sub = message_filters.Subscriber("gaze_array2", GazeInfoBino_Array)
	# image_sub = message_filters.Subscriber("detectron2_ros/image", Image)

	gaze_sub = message_filters.Subscriber("gaze_array", GazeInfoBino_Array)
	image_sub = message_filters.Subscriber("camera/rgb/image_raw", Image)

	ts = message_filters.ApproximateTimeSynchronizer([gaze_sub, image_sub], 10, 0.1)
	ts.registerCallback(callback)
	rospy.spin()

if __name__ == '__main__':
	visualizer()
