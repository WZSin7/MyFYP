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
#include "myEMD.h"

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>

#include <vocus2_ros/BoundingBox.h>
#include <vocus2_ros/BoundingBoxes.h>
#include <vocus2_ros/GazeInfoBino_Array.h>
#include <vocus2_ros/Result.h>
#include <vocus2_ros/Result_Detectron2.h>

#include "std_msgs/String.h"
#include <std_msgs/Int16.h>


using namespace cv;


VOCUS_ROS::VOCUS_ROS() : _it(_nh) //Constructor [assign '_nh' to '_it']
{
	_f = boost::bind(&VOCUS_ROS::callback, this, _1, _2);
	_server.setCallback(_f);

	//Added by me
	if (useMaskRCNN == true){
		image_sub.subscribe(_nh, "/detectron2_ros/image", 5);
		rcnn_result_sub.subscribe(_nh, "/detectron2_ros/result", 5);
		array_sub.subscribe(_nh, "gaze_array", 5); //Change accordingly
		sync_MaskRCNN.reset(new Sync_MaskRCNN(MaskRCNN_Policy(10), image_sub, rcnn_result_sub, array_sub)); //10 is queue size
		sync_MaskRCNN->registerCallback(boost::bind(&VOCUS_ROS::imageCb_MaskRCNN, this, _1, _2, _3));
	}
	else{
		image_sub.subscribe(_nh, "/darknet_ros/detection_image", 1);
		bboxes_sub.subscribe(_nh, "/darknet_ros/bounding_boxes", 1);
		array_sub.subscribe(_nh, "gaze_array", 1);
		// sync.reset(new Sync(MySyncPolicy(10), image_sub, bboxes_sub));
		// sync->registerCallback(boost::bind(&VOCUS_ROS::imageCb2, this, _1, _2));
		sync.reset(new Sync(MySyncPolicy(10), image_sub, bboxes_sub, array_sub));
		sync->registerCallback(boost::bind(&VOCUS_ROS::imageCb, this, _1, _2, _3));
	}
	//End of added by me

	_image_pub = _it.advertise("most_salient_region", 1);
	_image_sal_pub = _it.advertise("saliency_image_out", 1); //Potentially important
    _poi_pub = _nh.advertise<geometry_msgs::PointStamped>("saliency_poi", 1);
	_final_verdict_EMD_pub = _nh.advertise<vocus2_ros::Result>("final_verdict_EMD",10);
	_final_verdict_fixation_pub = _nh.advertise<vocus2_ros::Result>("final_verdict_fixation",10);
	_true_final_verdict_pub = _nh.advertise<vocus2_ros::Result>("final_verdict_true",10);
	_truth_pub = _nh.advertise<std_msgs::String>("truth",10);
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

void VOCUS_ROS::imageCb2(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::BoundingBoxesConstPtr& mybboxes){ //For troubleshooting
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
	float minEMD = INFINITY, minFixation = INFINITY;
	//std_msgs::String finalVerdict, finalVerdict_fixation;
	vocus2_ros::Result finalVerdict_EMD, finalVerdict_fixation;
	std_msgs::Int16 nums;
        // _cam.rectifyImage(cv_ptr->image, img);
	mainImg = cv_ptr->image;

	//Crop Image
	for (uint i=0; i< mybboxes->bounding_boxes.size(); i++){
		int xmin = mybboxes->bounding_boxes[i].xmin; //Top left is origin 
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
		//My code
		vector<values> storage, tempStorage; //tempStorage for all value, storage for only the highest l_pixels values
		float totalSum = 0, sdSum =0, sd;
		Size s = salmap.size();
		int rows = s.height;
		int cols = s.width;
		int l_pixels;
		float mean;
		if (useThres){ //Use threshold (Changed in VOCUS_ROS.h)
			float threshold = 0.8;
			//int l_pixels; //User defined
			cout << "No of rows(y): " << rows << ", No of cols(x): " << cols << endl;

			for (int i = 0; i<rows; ++i){
				for(int j =0; j<cols; j++){
					values temp;
					temp.row = i;
					temp.col = j;
					temp.intensity = salmap.at<float>(i,j);
					if(temp.intensity < threshold) continue;
					tempStorage.push_back(temp);
				}
			}

			l_pixels = tempStorage.size();
			sortIntensity(tempStorage); //Sort in according to the function

			for(int i=0; i<l_pixels; i++){
				storage.push_back(tempStorage[i]);
				totalSum+=tempStorage[i].intensity;
			}
		}
		else{ // Use fixed l_pixels
			l_pixels = 4000; //User defined
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
		}
		
		mean = totalSum/float(l_pixels);
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
		vector<finalValues> hypoGazePoints;
		vector<int> forEMD, forEMD2;
		vector<float> weights;
		float sumEuclDist=0,sumEuclDist_gaze = 0, meanEuclDist,meanEuclDist_gaze, lastValid_X =0.5, lastValid_Y =0.5;
		std::normal_distribution<float> d(mean, sd);
		vector<float> arrayX = {0.22371095,0.32982804,0.23055343,0.16278544,0.26971457,0.22807053,0.16255418,0.15598259,0.10242523,0.14069398,0.22127186,0.28384002,0.20125358,0.24224914,0.15061691,0.18899083,0.19496965,0.29235009,0.32375968,0.2438072,0.160521,0.21123199,0.2095943,0.22982817,0.16195227,0.23099075,0.26762711,0.22500883,0.17511318,0.11146625};
		vector<float> arrayY = {0.383161,0.14017015,0.33897986,0.20415037,0.34645933,0.39727204,0.28802226,0.31085289,0.24580363,0.23043759,0.36105778,0.27206512,0.24525784,0.36399496,0.20603097,0.37949472,0.25555136,0.3724595,0.2538986,0.15078471,0.25399853,0.24543093,0.31741116,0.38965459,0.18730317,0.22395852,0.28575449,0.23809562,0.18635813,0.20922096};
		for(int i = 0; i<k_pixels; i++){
			while(true){
				sample = d(gen);
				if(sample >= storage[0].intensity) break; //Retry until obtained intensity >= to min
			}
			int idx = findClosestID(storage,l_pixels,sample);
			finalValues temp;
			temp.row = storage[idx].row;
			temp.col = storage[idx].col;
			temp.euclideanDistance = calcDistance(temp.row,temp.col,rows/2,cols/2);
			forEMD.push_back(temp.euclideanDistance);
			float curX = arrayX[i];
			float curY = arrayY[i];
			
			//To handle if curX,curY [estimated gaze position] is not (0,1), not applicable for test scenario
			// if ((curX < 0)||(curX>1)) curX = lastValid_X;
			// if ((curY < 0)||(curY>1)) curY = lastValid_Y;
			// lastValid_X = curX;
			// lastValid_Y = curY;
			cout << "curX: " << curX <<", curY: "<< curY << endl;

			forEMD2.push_back(calcDistance(curX*1280-1, (1-curY)*720-1, (xmin+xdiff/2), (ymin+ydiff/2))); //Bottom Left is origin[myarray], Top Left is origin[bboxes]
			weights.push_back(1);
			sumEuclDist+=forEMD[i];
			sumEuclDist_gaze+=forEMD2[i];
			hypoGazePoints.push_back(temp);
		}
		meanEuclDist = sumEuclDist/float(k_pixels);
		meanEuclDist_gaze = sumEuclDist_gaze/float(k_pixels);
		cout << "Average Euclidean Distance for saliency map: " << meanEuclDist<< endl;
		cout << "Average Euclidean Distance for gaze points: " << meanEuclDist_gaze<< endl;
		signature_t s1 = {k_pixels, forEMD, weights};
		signature_t s2 = {k_pixels, forEMD2, weights};
		//cout << "For EMD1: " << endl;
		// for (int i=0; i<forEMD.size();i++){
		// 	cout<<forEMD[i]<<endl;
		// }
		// cout << "For EMD2: " << endl;
		// for (int i=0; i<forEMD2.size();i++){ 
		// 	cout<<forEMD2[i]<<endl;
		// }
		curEMD = emd(&s1, &s2, VOCUS_ROS::dist, NULL, NULL);
		cout<< ">>>EMD:" << curEMD << ", Class:" << mybboxes->bounding_boxes[i].Class << endl;

		float temp = (accumulate(forEMD2.begin(),forEMD2.end(),0))/k_pixels;
		cout << "temp: "<< temp << endl;

		if (curEMD < minEMD){
			minEMD = curEMD;
			finalVerdict_EMD.s = mybboxes->bounding_boxes[i].Class;
			nums.data = curEMD;
		}

		if (temp < minFixation){
			minFixation= temp;
			finalVerdict_fixation.s = mybboxes->bounding_boxes[i].Class;
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
	//Set output to none if EMD is too high
	if (minEMD>150) finalVerdict_EMD.s = "None";

	//Published correct object into final verdict topic
	finalVerdict_EMD.header.stamp = ros::Time::now();
	finalVerdict_fixation.header.stamp = ros::Time::now();
	_final_verdict_EMD_pub.publish(finalVerdict_EMD);
	_final_verdict_fixation_pub.publish(finalVerdict_fixation);
	cout << "EMD Final verdict: " << finalVerdict_EMD.s << ", " << nums.data << endl;
	cout << "Fixation final verdict: " << finalVerdict_fixation.s << endl;
	cout << "--------------------------------------------------------" << endl;
	std_msgs::String msg_truth;
    std::stringstream ss;
	if (finalVerdict_EMD.s == finalVerdict_fixation.s){
		ss << myCount++ << " True; "<< finalVerdict_EMD.s << ", " << finalVerdict_fixation.s;
	}
	else {
		ss << myCount++ << " False; " << finalVerdict_EMD.s << ", " << finalVerdict_fixation.s;
	}
	msg_truth.data = ss.str();
	_truth_pub.publish(msg_truth);
	cout << endl;

}

void VOCUS_ROS::imageCb(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::BoundingBoxesConstPtr& mybboxes, const vocus2_ros::GazeInfoBino_ArrayConstPtr& myarray)
{	
	cout << "Number of bounding box: " << mybboxes->bounding_boxes.size() << endl;
    if(RESTORE_DEFAULT) // if the reset checkbox has been ticked, we restore the default configuration
	{
		restoreDefaultConfiguration();
		RESTORE_DEFAULT = false;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try //Here
	{	
		cv_ptr = cv_bridge::toCvCopy(msg); // Change needed here
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	//Correcting gaze array
	float lastValid_X =0.5, lastValid_Y =0.5; 
	int mean_X, mean_Y;
	vector<float> corrected_X, corrected_Y;
	for(int i = 0; i<k_pixels; i++){
		//To handle if curX,curY [estimated gaze position] is not (0,1)
		float curX = myarray->x[i];
		float curY = myarray->y[i];
		if ((curX < 0)||(curX>1)) curX = lastValid_X;
		if ((curY < 0)||(curY>1)) curY = lastValid_Y;
		corrected_X.push_back(curX);
		corrected_Y.push_back(curY);
		lastValid_X = curX;
		lastValid_Y = curY;
	}
	mean_X = (accumulate(corrected_X.begin(),corrected_X.end(),0.0)/k_pixels)*1280-1;
	mean_Y = ((1-(accumulate(corrected_Y.begin(),corrected_Y.end(),0.0)/k_pixels))*720-1);

	Mat mainImg, img;
	float minEMD = INFINITY, minFixation = INFINITY;
	vocus2_ros::Result finalVerdict_EMD, finalVerdict_fixation, true_finalVerdict;
	vector<bool> withinMask;
	vector<float> eudDistanceforFixation;
	std_msgs::Int16 nums;
	mainImg = cv_ptr->image;

	//Crop Image
	for (uint i=0; i< mybboxes->bounding_boxes.size(); i++){
		int xmin = mybboxes->bounding_boxes[i].xmin; //Top left is origin 
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
		int l_pixels;
		float mean;
		if (useThres){ //Use threshold (Changed in VOCUS_ROS.h)
			float threshold = 0.8;
			//int l_pixels; //User defined
			//cout << "No of rows(y): " << rows << ", No of cols(x): " << cols << endl;

			for (int i = 0; i<rows; ++i){
				for(int j =0; j<cols; j++){
					values temp;
					temp.row = i;
					temp.col = j;
					temp.intensity = salmap.at<float>(i,j);
					if(temp.intensity < threshold) continue;
					tempStorage.push_back(temp);
				}
			}

			if(isnan(tempStorage[0].intensity)) continue;
			l_pixels = tempStorage.size();
			//sortIntensity(tempStorage); //Sort in according to the function

			for(int i=0; i<l_pixels; i++){
				storage.push_back(tempStorage[i]);
				totalSum+=tempStorage[i].intensity;
			}
		}
		else{ // Use fixed l_pixels
			l_pixels = rows*cols*0.2; //User defined
			//if(l_pixels > rows*cols) l_pixels = rows*cols;
			//cout << "No of rows(y): " << rows << ", No of cols(x): " << cols << endl;

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
		}
		
		mean = totalSum/float(l_pixels);
		for (int i = 0; i<l_pixels; i++){
			sdSum += pow((storage[i].intensity-mean),2);
		}
		sd = sqrt(sdSum/l_pixels);
		//cout << "Mean is " << mean << endl;
		//cout << "Standard Deviation is " << sd << endl;
		//cout << "Precentage of l_pixels over bounding boxes:" << float(l_pixels)/float((rows*cols))*100 <<"%" << endl;

		//For Gaussian Distrubtion
		std::random_device rd;
		std::mt19937 gen(rd());
		float sample, curEMD;
		vector<int> forEMD, forEMD2;
		vector<float> weights;
		float sumEuclDist=0,sumEuclDist_gaze = 0, meanEuclDist,meanEuclDist_gaze, lastValid_X =0.5, lastValid_Y =0.5;
		std::normal_distribution<float> d(mean, sd);
		for(int i = 0; i<k_pixels; i++){
			while(true){
				sample = d(gen);
				if(sample >= storage[0].intensity) break; //Retry until obtained intensity >= to min
			}
			int idx = findClosestID(storage,l_pixels,sample);
			finalValues temp;
			temp.row = storage[idx].row;
			temp.col = storage[idx].col;
			temp.euclideanDistance = calcDistance(temp.row,temp.col,rows/2,cols/2);
			forEMD.push_back(temp.euclideanDistance);

			forEMD2.push_back(calcDistance(corrected_X[i]*1280-1, (1-corrected_Y[i])*720-1, (xmin+xdiff/2), (ymin+ydiff/2))); //Bottom Left is origin[myarray], Top Left is origin[bboxes]
			weights.push_back(1);
			sumEuclDist+=forEMD[i];
			sumEuclDist_gaze+=forEMD2[i];
		}
		meanEuclDist = sumEuclDist/float(k_pixels);
		meanEuclDist_gaze = sumEuclDist_gaze/float(k_pixels);
		//cout << "Average Euclidean Distance for saliency map: " << meanEuclDist<< endl;
		//cout << "Average Euclidean Distance for gaze points: " << meanEuclDist_gaze<< endl;
		signature_t s1 = {k_pixels, forEMD, weights};
		signature_t s2 = {k_pixels, forEMD2, weights};

		curEMD = emd(&s1, &s2, VOCUS_ROS::dist, NULL, NULL);
		// cout<< ">>>EMD:" << curEMD << ", Class:" << mybboxes->bounding_boxes[i].Class << endl;

		float temp = (accumulate(forEMD2.begin(),forEMD2.end(),0))/k_pixels;
		eudDistanceforFixation.push_back(temp);
		// cout << "temp: "<< temp << endl;

		if (curEMD < minEMD){
			minEMD = curEMD;
			finalVerdict_EMD.s = mybboxes->bounding_boxes[i].Class;
			nums.data = curEMD;
		}

		if (temp < minFixation){
			minFixation= temp;
			finalVerdict_fixation.s = mybboxes->bounding_boxes[i].Class;
		}

		//End of My code

		// Output saliency map
		salmap *= 255.0;
		salmap.convertTo(salmap, CV_8UC1);
		cv_ptr->image = salmap;
		cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
		_image_sal_pub.publish(cv_ptr->toImageMsg());

	}
	//Set output to none if EMD is too high
	if (minEMD>170) finalVerdict_EMD.s = "None";
	if (minFixation>170) finalVerdict_fixation.s = "None";

	//Published correct object into final verdict topic
	finalVerdict_EMD.header.stamp = ros::Time::now();
	finalVerdict_fixation.header.stamp = ros::Time::now();
	_final_verdict_EMD_pub.publish(finalVerdict_EMD);
	_final_verdict_fixation_pub.publish(finalVerdict_fixation);

	cout << "--------------------------------------------------------" << endl;
	cout << "EMD verdict: " << finalVerdict_EMD.s << ", " << nums.data << endl;
	cout << "Fixation verdict: " << finalVerdict_fixation.s << endl;
	cout << "--------------------------------------------------------" << endl;
	std_msgs::String msg_truth;
    std::stringstream ss;
	if (finalVerdict_EMD.s == finalVerdict_fixation.s){
		ss << myCount++ << " True; "<< finalVerdict_EMD.s << ", " << finalVerdict_fixation.s;
	}
	else {
		ss << myCount++ << " False; " << finalVerdict_EMD.s << ", " << finalVerdict_fixation.s;
	}
	msg_truth.data = ss.str();
	_truth_pub.publish(msg_truth);
	cout << endl;

}

void VOCUS_ROS::imageCb_MaskRCNN(const sensor_msgs::ImageConstPtr& msg, const vocus2_ros::Result_Detectron2ConstPtr& detectron2_result, const vocus2_ros::GazeInfoBino_ArrayConstPtr& myarray){
	cout << "Number of bounding box: " << detectron2_result->boxes.size() << endl;
    if(RESTORE_DEFAULT) // if the reset checkbox has been ticked, we restore the default configuration
	{
		restoreDefaultConfiguration();
		RESTORE_DEFAULT = false;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try //Here
	{	
		cv_ptr = cv_bridge::toCvCopy(msg); // Change needed here
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	//Correcting gaze array
	float lastValid_X =0.5, lastValid_Y =0.5; 
	int mean_X, mean_Y;
	vector<float> corrected_X, corrected_Y;
	for(int i = 0; i<k_pixels; i++){
		//To handle if curX,curY [estimated gaze position] is not (0,1)
		float curX = myarray->x[i];
		float curY = myarray->y[i];
		if ((curX < 0)||(curX>1)) curX = lastValid_X;
		if ((curY < 0)||(curY>1)) curY = lastValid_Y;
		corrected_X.push_back(curX);
		corrected_Y.push_back(curY);
		lastValid_X = curX;
		lastValid_Y = curY;
	}
	mean_X = (accumulate(corrected_X.begin(),corrected_X.end(),0.0)/k_pixels)*1280-1; //1280
	mean_Y = ((1-(accumulate(corrected_Y.begin(),corrected_Y.end(),0.0)/k_pixels))*720-1); //720
	// cout << "mean_X: " << mean_X <<", mean_Y: " << mean_Y << endl;

	Mat mainImg, img;
	float minEMD = INFINITY, minFixation = INFINITY, minSDforSalmap;
	vocus2_ros::Result finalVerdict_EMD, finalVerdict_fixation, true_finalVerdict;
	vector<bool> withinMask;
	vector<float> eudDistanceforFixation;
	std_msgs::Int16 nums;
	mainImg = cv_ptr->image;

	//Crop Image
	for (uint i=0; i< detectron2_result->boxes.size(); i++){
		if (find(objectToCheck.begin(),objectToCheck.end(),detectron2_result->class_names[i]) == objectToCheck.end()){
			withinMask.push_back(false);
			eudDistanceforFixation.push_back(INFINITY);
			continue;
		}
		int xmin = detectron2_result->boxes[i].x_offset; //Top left is origin
		int ymin = detectron2_result->boxes[i].y_offset;
		int height = detectron2_result->boxes[i].height;
		int width = detectron2_result->boxes[i].width; 
		img = mainImg(Rect(xmin, ymin, width, height)); 

		cv_bridge::CvImagePtr cv_mask_ptr;
		try //Here
		{	
			cv_mask_ptr = cv_bridge::toCvCopy(detectron2_result->masks[i]); // Change needed here
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		Mat mask_img;
		mask_img = cv_mask_ptr->image;
		if (mask_img.at<uchar>(mean_Y,mean_X) != 0) withinMask.push_back(true); //For determining edge cases
		else withinMask.push_back(false);

		//Apply mask to image, so that background of the mask is filtered
		mask_img = mask_img(Rect(xmin, ymin, width, height))/255; 
		Mat mask_img_3c[] = {mask_img, mask_img, mask_img};
		merge(mask_img_3c, 3, mask_img); //Mask is 1 channel, merge 3 masks into 1 " 3 channel mask" then do element-wise multiplication
		img = img.mul(mask_img);


		Mat salmap;
		_vocus.process(img);

		if (TOPDOWN_LEARN == 1)
		{	
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
		float totalSum = 0, sdSum =0, sd, mean;
		Size s = salmap.size();
		int rows = s.height;
		int cols = s.width;
		int l_pixels;

		if (useThres){ //Use threshold (Changed in VOCUS_ROS.h)
			float threshold = 0.8;
			//int l_pixels; //User defined
			// cout << "No of rows(y): " << rows << ", No of cols(x): " << cols << endl;

			for (int i = 0; i<rows; ++i){
				for(int j =0; j<cols; j++){
					values temp;
					temp.row = i;
					temp.col = j;
					temp.intensity = salmap.at<float>(i,j);
					if(temp.intensity < threshold) continue;
					tempStorage.push_back(temp);
				}
			}

			if(isnan(tempStorage[0].intensity)) continue;
			l_pixels = tempStorage.size();

			for(int i=0; i<l_pixels; i++){
				storage.push_back(tempStorage[i]);
				totalSum+=tempStorage[i].intensity;
			}
		}
		else{ // Use fixed l_pixels
			l_pixels = rows*cols*0.2; //User defined
			//if(l_pixels > rows*cols) l_pixels = rows*cols;
			//cout << "No of rows(y): " << rows << ", No of cols(x): " << cols << endl;

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
			sortIntensity(tempStorage); //Sort in accordance to the function

			for(int i=totalSize-l_pixels; i<totalSize; i++){
				storage.push_back(tempStorage[i]);
				totalSum+=tempStorage[i].intensity;
			}
		}
		
		mean = totalSum/float(l_pixels);
		for (int i = 0; i<l_pixels; i++){
			sdSum += pow((storage[i].intensity-mean),2);
		}
		sd = sqrt(sdSum/l_pixels);

		//For Gaussian Distrubtion
		std::random_device rd;
		std::mt19937 gen(rd());
		float sample, curEMD;
		vector<int> forEMD, forEMD2;
		vector<float> weights;
		float meanEuclDist,meanEuclDist_gaze, sdEuclDist, sdEuclDist_gaze;
		std::normal_distribution<float> d(mean, sd);
		for(int i = 0; i<k_pixels; i++){
			while(true){
				sample = d(gen);
				if(sample >= storage[0].intensity) break; //Retry until obtained intensity >= to min
			}
			int idx = findClosestID(storage,l_pixels,sample);
			finalValues temp;
			temp.row = storage[idx].row;
			temp.col = storage[idx].col;
			temp.euclideanDistance = calcDistance(temp.row,temp.col,rows/2,cols/2);
			forEMD.push_back(temp.euclideanDistance);

			//cout << "curX: " << corrected_X[i] <<", curY: "<< corrected_Y[i] << endl;
			forEMD2.push_back(calcDistance(corrected_X[i]*1280-1, (1-corrected_Y[i])*720-1, (xmin+width/2), (ymin+height/2))); //Bottom Left is origin[myarray], Top Left is origin[bboxes]
			weights.push_back(1);
		}

		vector<float> diff(forEMD.size());
		float sum = accumulate(forEMD.begin(), forEMD.end(), 0.0);
		meanEuclDist = sum / forEMD.size();
		transform(forEMD.begin(), forEMD.end(), diff.begin(), [meanEuclDist](double x) { return x - meanEuclDist; });
		sdEuclDist = sqrt(inner_product(diff.begin(), diff.end(), diff.begin(), 0.0) / forEMD.size()); //Standard deviation of points from saliency map

		signature_t s1 = {k_pixels, forEMD, weights};
		signature_t s2 = {k_pixels, forEMD2, weights};

		curEMD = emd(&s1, &s2, VOCUS_ROS::dist, NULL, NULL);
		//cout<< ">>>EMD:" << curEMD << ", Class:" << detectron2_result->class_names[i]<< endl;

		float temp = (accumulate(forEMD2.begin(),forEMD2.end(),0))/k_pixels;
		eudDistanceforFixation.push_back(temp);
		//cout << "temp: "<< temp << endl;

		if (curEMD < minEMD){
			minEMD = curEMD;
			minSDforSalmap = sdEuclDist;
			finalVerdict_EMD.s = detectron2_result->class_names[i];
			nums.data = curEMD;
		}

		if (temp < minFixation){
			minFixation= temp;
			finalVerdict_fixation.s = detectron2_result->class_names[i];
		}

		// //End of My code

		// Output saliency map
		salmap *= 255.0;
		salmap.convertTo(salmap, CV_8UC1);
		cv_ptr->image = salmap;
		cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
		_image_sal_pub.publish(cv_ptr->toImageMsg());

	}
	//Set output to none if EMD is too high
	if (minEMD>300) finalVerdict_EMD.s = "None";
	if (minFixation>250) finalVerdict_fixation.s = "None";

	//Published correct object into final verdict topic
	bool edgeCase = isEdgeCase(detectron2_result, mean_X, mean_Y, withinMask, eudDistanceforFixation);
	if (finalVerdict_fixation.s == "None") edgeCase = true;
	true_finalVerdict.s = (edgeCase)? finalVerdict_EMD.s : finalVerdict_fixation.s;
	true_finalVerdict.isEdgeCase = edgeCase;
	true_finalVerdict.class_names =  detectron2_result->class_names;

	finalVerdict_EMD.header.stamp = ros::Time::now();
	finalVerdict_fixation.header.stamp = ros::Time::now();
	true_finalVerdict.header.stamp = ros::Time::now();

	_final_verdict_EMD_pub.publish(finalVerdict_EMD);
	_final_verdict_fixation_pub.publish(finalVerdict_fixation);
	_true_final_verdict_pub.publish(true_finalVerdict);

	cout << "--------------------------------------------------------" << endl;
	cout << "EMD verdict: " << finalVerdict_EMD.s << ", " << nums.data << endl;
	cout << "Fixation verdict: " << finalVerdict_fixation.s << endl;
	cout << "Final verdict: " << true_finalVerdict.s << endl;
	cout << "--------------------------------------------------------" << endl;
	std_msgs::String msg_truth;
    std::stringstream ss;
	if (finalVerdict_EMD.s == finalVerdict_fixation.s){
		ss << myCount++ << " True; "<< finalVerdict_EMD.s << ", " << finalVerdict_fixation.s << ", Edge Case: " << edgeCase << " , SD: " << minSDforSalmap;
	}
	else {
		ss << myCount++ << " False; " << finalVerdict_EMD.s << ", " << finalVerdict_fixation.s << ", Edge Case: " << edgeCase << " , SD: " << minSDforSalmap;
	}
	msg_truth.data = ss.str();
	_truth_pub.publish(msg_truth);
	cout << endl;

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

void VOCUS_ROS::sortIntensity(vector<values>& storage){ //Ascending order, change compareIntensity form '<' to '>' for descending order
	sort(storage.begin(),storage.end(), [](const values& A, const values& B){return A.intensity < B.intensity;}); 
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

float VOCUS_ROS::dist(int F1, int F2){
	return abs(F1 - F2);
}

bool VOCUS_ROS::isEdgeCase(const vocus2_ros::Result_Detectron2ConstPtr& detectron2_result, int x, int y, vector<bool>& withinMask, vector<float>& eucDistforFixation){
	// for (int i =0; i < withinMask.size(); i++){
	// 	cout << withinMask[i]<< endl;
	// }
	//First Check: Check if gaze point of min euclidean distances is within the corresponding mask
	if (withinMask.size() == 0) return false;
		
	if (withinMask[std::distance(eucDistforFixation.begin(), std::min_element(eucDistforFixation.begin(),eucDistforFixation.end()))] == false){ 
		// cout << "edgeCase Case 1" << endl;
		return true;
	}

	//Second Check: Check if gaze exist in close proximity to multiple BBoxes
	int count = 0; //No of overlaps, return true if count >=2
	float tolerance = 0.1; // 10% Tolerance
	for (uint i=0; i< detectron2_result->boxes.size(); i++){
		if (find(objectToCheck.begin(),objectToCheck.end(),detectron2_result->class_names[i]) == objectToCheck.end()) continue;
		int xmin = detectron2_result->boxes[i].x_offset; //Top left is origin
		int ymin = detectron2_result->boxes[i].y_offset;
		int height = detectron2_result->boxes[i].height;
		int width = detectron2_result->boxes[i].width; 
		
		// Increase by tolerance specified and check if there is intersection
		int new_xmin = xmin - width*tolerance;
		int new_ymin = ymin - height*tolerance;
		int new_xmax = xmin + width + width*tolerance;
		int new_ymax = ymin + height + height*tolerance;

		if ((x>new_xmin && x < new_xmax) && (y>new_ymin && y< new_ymax)){
			cv_bridge::CvImagePtr cv_mask_ptr;
			Mat mask_img;
			try 
			{	
				cv_mask_ptr = cv_bridge::toCvCopy(detectron2_result->masks[i]);
			}
			catch (cv_bridge::Exception& e)
			{
				ROS_ERROR("cv_bridge exception during edge case detection: %s", e.what());
			}
			mask_img = cv_mask_ptr->image;
			dilate(mask_img,mask_img,cv::getStructuringElement(MORPH_RECT,cv::Size(50,50)));
			if(mask_img.at<uchar>(y,x)) count++; //NumpyArray -> [Rows][Cols]
			if (count >=2) {
				return true;
			}
		}
	}
	return false;
}