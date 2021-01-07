#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from myPupilLab.msg import Result_Detectron2
from myPupilLab.msg import GazeInfoBino_Array
import message_filters
import numpy as np
from cv_bridge import CvBridge


pub_result = rospy.Publisher('detectron2_prediction',String,queue_size=1)

def callback(gaze, detectron2_ros):
	lastValid_X = 0.5 
	lastValid_Y = 0.5
	gazeX = []
	gazeY = []

	for i in range(0,len(gaze.x)-1):
		# To handle if curX,curY [estimated gaze position] is not (0,1)
		if (gaze.x[i]<0 or gaze.x[i]>1):
			gazeX.append(lastValid_X)
		else:
			gazeX.append(gaze.x[i])
			lastValid_X = gaze.x[i]

		if (gaze.y[i]<0 or gaze.y[i]>1):
			gazeY.append(lastValid_Y)
		else:
			gazeY.append(gaze.y[i])
			lastValid_Y = gaze.y[i]
		
	cur_X = np.sum(gazeX)/len(gazeX)
	cur_Y = np.sum(gazeY)/len(gazeY)


	x = int(cur_X*1280-1)
	y = int(((1-cur_Y)*720)-1)
	# rospy.loginfo("x:")
	# rospy.loginfo(x)
	# rospy.loginfo("y:")
	# rospy.loginfo(y)
	flag = False
	for id, mask in enumerate(detectron2_ros.masks):
		bridge = CvBridge()
		np_array = bridge.imgmsg_to_cv2(mask, desired_encoding='passthrough')
		a = np_array[y][x]
		if a!=0 :
			flag = True
			pub_result.publish(detectron2_ros.class_names[id])
			print("x: ",x," y: ",y)
			print("Detectron2 prediction is ",detectron2_ros.class_names[id])
			print("-----------------------------------------------------")
			break
	
	if not flag:
		print(">>>>-----------------------------------------")
		print("x: ",x," y: ",y)
		print(detectron2_ros.class_names)
		print(">>>>-----------------------------------------")
		pub_result.publish("None")

def detectron2_test():
	rospy.init_node('detectron2_test', anonymous=True)

	gaze_sub = message_filters.Subscriber("gaze_array2", GazeInfoBino_Array)
	mask_sub = message_filters.Subscriber("detectron2_ros/result", Result_Detectron2)

	ts = message_filters.ApproximateTimeSynchronizer([gaze_sub, mask_sub], 10, 0.1)
	ts.registerCallback(callback)
	rospy.spin()

if __name__ == '__main__':
	detectron2_test()
