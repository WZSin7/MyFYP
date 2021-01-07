#!/usr/bin/env python
import rospy
from myPupilLab.msg import GazeInfoBino_Array


pub_gaze_transition = rospy.Publisher('gaze_array2',GazeInfoBino_Array,queue_size=1)


def callback(data):
	#rospy.loginfo("Sucess!")
	outmsg = GazeInfoBino_Array()
	outmsg.x = data.x
	outmsg.y = data.y
	outmsg.header.stamp = rospy.Time.now()
	pub_gaze_transition.publish(outmsg)
	
def listener():
	rospy.init_node('test_Sub&Pub', anonymous=True)

	rospy.Subscriber("gaze_array", GazeInfoBino_Array, callback)
	rospy.spin()

if __name__ == '__main__':
	listener()
