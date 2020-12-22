#ifndef VOCUS_ROS_H
#define VOCUS_ROS_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "VOCUS2.h"
#include <mutex>
#include <dynamic_reconfigure/server.h>
#include <vocus2_ros/vocus2_rosConfig.h>

#include <image_geometry/pinhole_camera_model.h>

// darknet_ros_msgs
#include <vocus2_ros/BoundingBox.h>
#include <vocus2_ros/BoundingBoxes.h>
#include <vocus2_ros/GazeInfoBino_Array.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

class VOCUS_ROS
{

public:


    // default constructor
  VOCUS_ROS();
  ~VOCUS_ROS();

  void restoreDefaultConfiguration();

  struct values{
    int row;
    int col;
    float intensity;
  };
  struct finalValues{
    int row;
    int col;
    int euclideanDistance;
  };
  
    // this sets VOCUS's own configuration object's settings to the one given in the ROS configuration 
    // coming from the dynamic_reconfigure module
  void setVOCUSConfigFromROSConfig(VOCUS2_Cfg& vocus_cfg, const vocus2_ros::vocus2_rosConfig &config);

    // callback that is called when a new image is published to the topic
  void imageCb(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::BoundingBoxesConstPtr& mybboxes, const vocus2_ros::GazeInfoBino_ArrayConstPtr& myarray);
  
  void imageCb2(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::BoundingBoxesConstPtr& mybboxes);

    // callback that is called after the configuration in dynamic_reconfigure has been changed
  void callback(vocus2_ros::vocus2_rosConfig &config, uint32_t level);

  //Functions to calc EMD
  void sortIntensity(vector<values>& storage);

  int getClosest(float val1, float val2, int a, int b, float target);

  int findClosestID(vector<values>& storage, int n, float target);

  float calcDistanceF(int x1, int y1, int x2, int y2);

  int calcDistance(int x1, int y1, int x2, int y2);

  static float dist(int F1, int F2);

private:
  // the VOCUS2 object that will 
  // process all the images
	VOCUS2 _vocus;

  ros::NodeHandle _nh;
  image_transport::ImageTransport _it;
  ros::Subscriber _cam_sub;
  image_transport::Publisher _image_pub;
  image_transport::Publisher _image_sal_pub;
  ros::Publisher _poi_pub, _final_verdict_pub, _nums_pub;
  message_filters::Subscriber<sensor_msgs::Image> image_sub;
	message_filters::Subscriber<vocus2_ros::BoundingBoxes> bboxes_sub;
  message_filters::Subscriber<vocus2_ros::GazeInfoBino_Array> array_sub;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, vocus2_ros::BoundingBoxes> MySyncPolicy; //Change between 'Approximate' and 'Exact'
	typedef message_filters::Synchronizer<MySyncPolicy> Sync; //, vocus2_ros::GazeInfoBino_Array ^^^^^^^^^^^^^^^^
  boost::shared_ptr<Sync> sync;
  bool useThres = true; //Use threshold / fixed l_pixels value

  image_geometry::PinholeCameraModel _cam;

  // dynamic reconfigure server and callbacktype
  dynamic_reconfigure::Server<vocus2_ros::vocus2_rosConfig> _server;
  dynamic_reconfigure::Server<vocus2_ros::vocus2_rosConfig>::CallbackType _f;

  // VOCUS configuration
  VOCUS2_Cfg _cfg;
  // ROS configuration
  vocus2_ros::vocus2_rosConfig _config;
  std::mutex cfg_mutex;
  boost::recursive_mutex config_mutex;

  // parameters that are not included in VOCUS's configuration but should be
  // configurable from the dynamic_reconfigure module
  int TOPDOWN_SEARCH = -1;
  int TOPDOWN_LEARN = -1;
  float MSR_THRESH = 0.75; // most salient region
  double CENTER_BIAS = 0.000005;
  bool COMPUTE_CBIAS = false;
  int NUM_FOCI = 1;

  // helper bools to enable button-like behavior of check boxes
  bool HAS_LEARNED = false;
  bool RESTORE_DEFAULT = false;

  const std::string OPENCV_WINDOW = "Image window";


};

  #endif // VOCUS_ROS_H
