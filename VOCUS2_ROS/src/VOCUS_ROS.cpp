#include "VOCUS_ROS.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <sys/stat.h>

#include "ImageFunctions.h"
#include "HelperFunctions.h"
#include "wasserstein.h"

#include <vocus2_ros/BoundingBox.h>
#include <vocus2_ros/BoundingBoxes.h>
#include <opencv2/opencv.hpp>
#include <random>
#include "std_msgs/String.h"
#include <string>

using namespace cv;


VOCUS_ROS::VOCUS_ROS() : _it(_nh) //Constructor [assign '_nh' to '_it']
{
	_f = boost::bind(&VOCUS_ROS::callback, this, _1, _2);
	_server.setCallback(_f);
	//_cam_sub = _nh.subscribe("/darknet_ros/detection_image", 1, &VOCUS_ROS::imageCb, this);

	//Added by me
	image_sub.subscribe(_nh, "/darknet_ros/detection_image", 1);
	bboxes_sub.subscribe(_nh, "/darknet_ros/bounding_boxes", 1);
	sync.reset(new Sync(MySyncPolicy(10), image_sub, bboxes_sub));
	sync->registerCallback(boost::bind(&VOCUS_ROS::imageCb, this, _1, _2));
	//End of added by me

	//_image_sub = _it.subscribe("/usb_cam/image_raw", 1,
	//	&VOCUS_ROS::imageCb, this);
	//_image_pub = _it.advertise("/image_converter/output_video", 1);
	_image_pub = _it.advertise("most_salient_region", 1);
	_image_sal_pub = _it.advertise("saliency_image_out", 1); //Potentially important
        _poi_pub = _nh.advertise<geometry_msgs::PointStamped>("saliency_poi", 1);
	_final_verdict_pub = _nh.advertise<std_msgs::String>("final_verdict",10);
}

VOCUS_ROS::~VOCUS_ROS()
{}


void VOCUS_ROS::restoreDefaultConfiguration()
{
	exit(0);
	boost::recursive_mutex::scoped_lock lock(config_mutex); 
	vocus2_ros::vocus2_rosConfig config;
	_server.getConfigDefault(config);
	_server.updateConfig(config);
	COMPUTE_CBIAS = config.center_bias;
	CENTER_BIAS = config.center_bias_value;
	NUM_FOCI = config.num_foci;
	MSR_THRESH = config.msr_thresh;
	TOPDOWN_LEARN = config.topdown_learn;
	TOPDOWN_SEARCH = config.topdown_search;
	setVOCUSConfigFromROSConfig(_cfg, config);
	_vocus.setCfg(_cfg);
	lock.unlock();

}

void VOCUS_ROS::setVOCUSConfigFromROSConfig(VOCUS2_Cfg& vocus_cfg, const vocus2_ros::vocus2_rosConfig &config)
{
	cfg_mutex.lock();
	vocus_cfg.fuse_feature = (FusionOperation) config.fuse_feature;  
	vocus_cfg.fuse_conspicuity = (FusionOperation) config.fuse_conspicuity;
	vocus_cfg.c_space = (ColorSpace) config.c_space;
	vocus_cfg.start_layer = config.start_layer;
    vocus_cfg.stop_layer = max(vocus_cfg.start_layer,config.stop_layer); // prevent stop_layer < start_layer
    vocus_cfg.center_sigma = config.center_sigma;
    vocus_cfg.surround_sigma = config.surround_sigma;
    vocus_cfg.n_scales = config.n_scales;
    vocus_cfg.normalize = config.normalize;
    vocus_cfg.orientation = config.orientation;
    vocus_cfg.combined_features = config.combined_features;
    vocus_cfg.descriptorFile = "topdown_descriptor";

    // individual weights?
    if (config.fuse_conspicuity == 3)
    {
    	vocus_cfg.weights[10] = config.consp_intensity_on_off_weight;
    	vocus_cfg.weights[11] = config.color_channel_1_weight;
    	vocus_cfg.weights[12] = config.color_channel_2_weight;
    	vocus_cfg.weights[13] = config.orientation_channel_weight;
    }
    if (config.fuse_feature == 3)
    {
    	vocus_cfg.weights[0] = config.intensity_on_off_weight;
    	vocus_cfg.weights[1] = config.intensity_off_on_weight;
    	vocus_cfg.weights[2] = config.color_a_on_off_weight;
    	vocus_cfg.weights[3] = config.color_a_off_on_weight;
    	vocus_cfg.weights[4] = config.color_b_on_off_weight;
    	vocus_cfg.weights[5] = config.color_b_off_on_weight;
    	vocus_cfg.weights[6] = config.orientation_1_weight;
    	vocus_cfg.weights[7] = config.orientation_2_weight;
    	vocus_cfg.weights[8] = config.orientation_3_weight;
    	vocus_cfg.weights[9] = config.orientation_4_weight;

    }

    cfg_mutex.unlock();
}

void VOCUS_ROS::imageCb2(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::BoundingBoxesConstPtr& mybboxes){
	ROS_INFO("Callback working");
}

void VOCUS_ROS::imageCb(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::BoundingBoxesConstPtr& mybboxes)
{
    //_cam.fromCameraInfo(info_msg);
	ROS_INFO("callback");
	cout << "Number of bounding box: " << mybboxes->bounding_boxes.size() << endl;
    if(RESTORE_DEFAULT) // if the reset checkbox has been ticked, we restore the default configuration
	{
		ROS_INFO("RESTORE_DEFAULT");
		restoreDefaultConfiguration();
		RESTORE_DEFAULT = false;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try //Here
	{	
		ROS_INFO("CV_BRIDGE");
		cv_ptr = cv_bridge::toCvCopy(msg); // Change needed here
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}


	Mat mainImg, img;
	float minEMD = INFINITY;
	std_msgs::String finalVerdict;
        // _cam.rectifyImage(cv_ptr->image, img);
	mainImg = cv_ptr->image;

	//Crop Image
	for (uint i=0; i< mybboxes->bounding_boxes.size(); i++){
		int xmin = mybboxes->bounding_boxes[i].xmin;
		int xmax = mybboxes->bounding_boxes[i].xmax;
		int ymin = mybboxes->bounding_boxes[i].ymin;
		int ymax = mybboxes->bounding_boxes[i].ymax;
		int xdiff = xmax - xmin;
		int ydiff = ymax - ymin;  
		img = mainImg(Rect(xmin, ymin, xdiff, ydiff));

		Mat salmap;
		_vocus.process(img);

		if (TOPDOWN_LEARN == 1)
		{	
			ROS_INFO("TOPDOWN");
			Rect ROI = annotateROI(img);

		//compute feature vector
			_vocus.td_learn_featurevector(ROI, _cfg.descriptorFile);

		// to turn off learning after we've learned
		// we disable it in the ros_configuration
		// and set TOPDOWN_LEARN to -1
			cfg_mutex.lock();
			_config.topdown_learn = false;
			_server.updateConfig(_config);
			cfg_mutex.unlock();
			TOPDOWN_LEARN = -1;
			HAS_LEARNED = true;
			return;
		}
		if (TOPDOWN_SEARCH == 1 && HAS_LEARNED)
		{
			ROS_INFO("TOPDOWN//HASLEARNED");
			double alpha = 0;
			salmap = _vocus.td_search(alpha);
			if(_cfg.normalize){

				double mi, ma;
				minMaxLoc(salmap, &mi, &ma);
				cout << "saliency map min " << mi << " max " << ma << "\n";
				salmap = (salmap-mi)/(ma-mi);
			}
		}
		else //Here
		{
			ROS_INFO("TOPDOWN//HASLEARNED--ELSE");
			salmap = _vocus.compute_salmap();
			if(COMPUTE_CBIAS)
				salmap = _vocus.add_center_bias(CENTER_BIAS);
		}
		vector<vector<Point>> msrs = computeMSR(salmap,MSR_THRESH, NUM_FOCI);

		for (const auto& msr : msrs)
		{
			if (msr.size() < 3000)
			{ // if the MSR is really large, minEnclosingCircle sometimes runs for more than 10 seconds,
			// freezing the whole proram. Thus, if it is very large, we 'fall back' to the efficiently
			// computable bounding rectangle
				Point2f center;
				float rad=0;
				minEnclosingCircle(msr, center, rad);
				if(rad >= 5 && rad <= max(img.cols, img.rows)){
					circle(img, center, (int)rad, Scalar(0,0,255), 3);
				}
			}
			else
			{
				Rect rect = boundingRect(msr);
				rectangle(img, rect, Scalar(0,0,255),3);
			}
		}
		
		//My code
		vector<values> storage, tempStorage; //tempStorage for all value, storage for only the highest l_pixels values
		float totalSum = 0, sdSum =0, sd;
		Size s = salmap.size();
		int rows = s.height;
		int cols = s.width;
		int l_pixels = 400; //User defined
		if(l_pixels > rows*cols) l_pixels = rows*cols;
		cout << "No of rows(y): " << rows << ", No of cols(x): " << cols << endl;

		for (int i = 0; i<rows; ++i){
			for(int j =0; j<cols; j++){
				values temp;
				temp.row = i;
				temp.col = j;
				temp.intensity = salmap.at<float>(i,j);
				tempStorage.push_back(temp);
			}
		}

		int totalSize = tempStorage.size()-1;
		sortIntensity(tempStorage); //Sort in according to the function

		for(int i=totalSize-l_pixels; i<totalSize; i++){
			storage.push_back(tempStorage[i]);
			totalSum+=tempStorage[i].intensity;
		}

		float mean = totalSum/float(l_pixels);

		for (int i = 0; i<l_pixels; i++){
			sdSum += pow((storage[i].intensity-mean),2);
		}

		sd = sqrt(sdSum/l_pixels);
		cout << "Mean is " << mean << endl;
		cout << "Standard Deviation is " << sd << endl;
		cout << "Precentage of l_pixels over bounding boxes:" << float(l_pixels)/float((rows*cols))*100 <<"%" << endl;

		//For Gaussian Distrubtion
		std::random_device rd;
		std::mt19937 gen(rd());
		float sample, curEMD;
		int k_pixels = l_pixels*0.2; //User defined
		vector<finalValues> hypoGazePoints;
		vector<int> forEMD, weights,forEMD2;
		float sumEuclDist=0,meanEuclDist;
		std::normal_distribution<float> d(mean, sd);
		for(int i = 0; i<k_pixels; i++){
			while(true){
				sample = d(gen);
				if(sample >= storage[0].intensity) break; //Retry until obtained intensity >= to min
			}
			//cout << "RNG: "<< sample << endl;
			int idx = findClosestID(storage,l_pixels,sample);
			//cout << "IDX: " << idx << endl;
			finalValues temp;
			temp.row = storage[idx].row;
			temp.col = storage[idx].col;
			temp.euclideanDistance = calcDistance(temp.row,temp.col,rows/2,cols/2);
			forEMD.push_back(temp.euclideanDistance);
			forEMD2.push_back(temp.euclideanDistance+5);
			weights.push_back(1);
			sumEuclDist+=temp.euclideanDistance;
			hypoGazePoints.push_back(temp);
		}
		cout << "First Euclidean Distance: " << hypoGazePoints[0].euclideanDistance << endl;
		meanEuclDist = sumEuclDist/float(k_pixels);
		cout << "Average Euclidean Distance: " << meanEuclDist<< endl;
		curEMD = wasserstein(forEMD,weights,forEMD2,weights);
		cout<< "EMD:" << curEMD << endl;

		if (curEMD < minEMD){
			minEMD = curEMD;
			finalVerdict.data = mybboxes->bounding_boxes[i].Class;
		}

		//End of My code



		// Output modified video stream
		cv_ptr->image= img;
		_image_pub.publish(cv_ptr->toImageMsg());
		ROS_INFO("IMG_PUB");

		// Output saliency map
		salmap *= 255.0;
		salmap.convertTo(salmap, CV_8UC1);
		cv_ptr->image = salmap;
		cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
		ROS_INFO("SAL_PUB");
		_image_sal_pub.publish(cv_ptr->toImageMsg());

		// Output 3D point in the direction of the first MSR
		if( msrs.size() > 0 ){
		geometry_msgs::PointStamped point;
		point.header = msg->header;
		cv::Point3d cvPoint = _cam.projectPixelTo3dRay(msrs[0][0]);
		point.point.x = cvPoint.x;
		point.point.y = cvPoint.y;
		point.point.z = cvPoint.z;
		_poi_pub.publish(point);
		}

	}
	// TODO: Published correct object into final verdict topic
	_final_verdict_pub.publish(finalVerdict);

}

void VOCUS_ROS::callback(vocus2_ros::vocus2_rosConfig &config, uint32_t level) 
{
	setVOCUSConfigFromROSConfig(_cfg, config);
	COMPUTE_CBIAS = config.center_bias;
	CENTER_BIAS = config.center_bias_value;

	NUM_FOCI = config.num_foci;
	MSR_THRESH = config.msr_thresh;
	TOPDOWN_LEARN = config.topdown_learn;
	if (config.restore_default) // restore default parameters before the next image is processed
		RESTORE_DEFAULT = true;
	TOPDOWN_SEARCH = config.topdown_search;
	_vocus.setCfg(_cfg);
	_config = config;
}

bool VOCUS_ROS::compareIntensity(const values A, const values B){
	return A.intensity < B.intensity;
}

void VOCUS_ROS::sortIntensity(vector<values>& storage){ //Ascending order, change compareIntensity form '<' to '>' for descending order
	sort(storage.begin(),storage.end(), VOCUS_ROS::compareIntensity); //[](const values& A, const values& B){return A.intensity < B.intensity;}
}

int VOCUS_ROS::getClosest(float val1, float val2, int a, int b, float target){ 
	return (target - val1 >= val2 - target)? b: a;
} 

int VOCUS_ROS::findClosestID(vector<values>& storage, int n, float target){ 
    // Corner cases 
    if (target <= storage[0].intensity)return 0;
    if (target >= storage[n-1].intensity) return n-1; 
	int i = 0, j = n, mid;
	while(i<j){
		mid = (i+j)/2;
		if (storage[mid].intensity == target) return mid;
		//Target less than mid
		if (target < storage[mid].intensity){
			if (mid > 0 && target > storage[mid-1].intensity){
				return getClosest(storage[mid].intensity,storage[mid-1].intensity,mid,mid-1,target);
			}
			j = mid; 
		}
		// Target larger than mid
		else{
			if(mid < n-1 && target< storage[mid+1].intensity){
				return getClosest(storage[mid].intensity,storage[mid+1].intensity,mid,mid+1,target);
			}
			i = mid +1;
		}
	}
    return mid;
} 

float VOCUS_ROS::calcDistanceF(int x1, int y1, int x2, int y2){
	int x = x1 - x2; //calculating number to square in next step
	int y = y1 - y2;

	return sqrtf32(pow(x, 2) + pow(y, 2));
}

int VOCUS_ROS::calcDistance(int x1, int y1, int x2, int y2){
	int x = x1 - x2; //calculating number to square in next step
	int y = y1 - y2;

	return sqrt(pow(x, 2) + pow(y, 2));
}