#!/usr/bin/env python3
import rospy
from myPupilLab.msg import Result_Detectron2
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def callback(data):
    bridge = CvBridge()
    for mask in data.masks:
        img = bridge.imgmsg_to_cv2(mask, desired_encoding='passthrough')
        kernel = np.ones((100,100),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        image_message = bridge.cv2_to_imgmsg(dilation, encoding="passthrough")
        Img_Pub2.publish(mask)
        Img_Pub.publish(image_message)
        print("Yes!")
    
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('forTesting', anonymous=True)

    rospy.Subscriber("/detectron2_ros/result", Result_Detectron2, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    Img_Pub = rospy.Publisher('/JustForTesting',Image, queue_size=10)
    Img_Pub2 = rospy.Publisher('/JustForTesting_OG',Image, queue_size=10)
    listener()
