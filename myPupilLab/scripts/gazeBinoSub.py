#!/usr/bin/env python
import rospy
from myPupilLab.msg import GazeInfoBino

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %f", data) #Change data.xxx into the variable you want to read
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("gaze_info_bino", GazeInfoBino, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()