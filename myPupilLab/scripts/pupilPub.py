#!/usr/bin/env python
import zmq
import msgpack
import rospy
from myPupilLab.msg import PupilInfo_2d
from myPupilLab.msg import PupilInfo_3d
# from std_msgs.msg import String

ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = ctx.socket(zmq.REQ)

ip = 'localhost'  # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

pupil_remote.connect(f'tcp://{ip}:{port}')

# Request 'SUB_PORT' for reading data
pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

# Request 'PUB_PORT' for writing data
pupil_remote.send_string('PUB_PORT')
pub_port = pupil_remote.recv_string()

# Assumes `sub_port` to be set to the current subscription port
subscriber = ctx.socket(zmq.SUB)
subscriber.connect(f'tcp://{ip}:{sub_port}') #sub_port not port
subscriber.subscribe('pupil.')  # receive pupil info, 'gaze.' is also valid but need world camera

pub3d = rospy.Publisher('pupil_info_3d',PupilInfo_3d,queue_size=10)
pub2d = rospy.Publisher('pupil_info_2d',PupilInfo_2d,queue_size=10)
def pupil_parser_3d(message):
	#Parse pupil_position into a PupilInfo message.
	if len(message) < 1: return -1 
	
	outmsg = PupilInfo_3d()

	outmsg.circle3D_center = message[b'circle_3d'][b'center']
	outmsg.circle3D_normal = message[b'circle_3d'][b'normal']
	outmsg.circle3D_radius = message[b'circle_3d'][b'radius']
	outmsg.confidence = message[b'confidence']
	outmsg.timestamp = message[b'timestamp']
	outmsg.diameter_3d = message[b'diameter_3d']
	outmsg.ellipse_center = message[b'ellipse'][b'center']
	outmsg.ellipse_axes = message[b'ellipse'][b'axes']
	outmsg.ellipse_angle = message[b'ellipse'][b'angle']
	outmsg.location = message[b'location']
	outmsg.diameter = message[b'diameter']
	outmsg.sphere_center = message[b'sphere'][b'center']
	outmsg.sphere_radius = message[b'sphere'][b'radius']
	outmsg.projected_sphere_center = message[b'projected_sphere'][b'center']
	outmsg.projected_sphere_axes = message[b'projected_sphere'][b'axes']
	outmsg.projected_sphere_angle = message[b'projected_sphere'][b'angle']
	outmsg.model_confidence = message[b'model_confidence']
	outmsg.model_id = message[b'model_id']
	outmsg.model_birth_timestamp = message[b'model_birth_timestamp']
	outmsg.theta = message[b'theta']
	outmsg.phi = message[b'phi']
	outmsg.norm_pos = message[b'norm_pos']
	outmsg.id = message[b'id']
	outmsg.method = "3d c++" #Can't use message [b'method'] because it will return byte string, byte has no attribute 'encode'

	return outmsg

def pupil_parser_2d(message):
	#Parse pupil_position into a PupilInfo message.
	if len(message) < 1: return -1 
	
	outmsg = PupilInfo_2d()

	outmsg.confidence = message[b'confidence']
	outmsg.timestamp = message[b'timestamp']
	outmsg.ellipse_center = message[b'ellipse'][b'center']
	outmsg.ellipse_axes = message[b'ellipse'][b'axes']
	outmsg.ellipse_angle = message[b'ellipse'][b'angle']
	outmsg.location = message[b'location']
	outmsg.diameter = message[b'diameter']
	outmsg.norm_pos = message[b'norm_pos']
	outmsg.id = message[b'id']
	outmsg.method = "2d c++"

	return outmsg

if __name__ == "__main__":
	rospy.loginfo("Starting pupil publisher.")
	print("Starting pupil publisher")

	rospy.init_node('pupilPublisher')

	print("listening for socket message....")
	while not rospy.is_shutdown():
		topic, payload = subscriber.recv_multipart()
		message = msgpack.loads(payload)
		if topic == b'pupil.1.3d' or topic == b'pupil.0.3d':
			# print("tcp://{}:{}".format(topic,message))
			# print(message[b'circle_3d'][b'center'])
			outmsg = pupil_parser_3d(message)
			if not outmsg == -1:
				rospy.loginfo(outmsg) 
				pub3d.publish(outmsg)

		elif topic == b'pupil.1.2d' or topic == b'pupil.0.2d':
			outmsg = pupil_parser_2d(message)
			if not outmsg == -1:
				rospy.loginfo(outmsg) 
				pub2d.publish(outmsg)
