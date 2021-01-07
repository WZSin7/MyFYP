#include "ros/ros.h"
#include "std_msgs/String.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <myPupilLab/Result.h>
#include <string.h>
#include <iostream>

using namespace message_filters;
using namespace std;

//Global variables
int emd_true = 0, emd_false=0, fixation_true=0, fixation_false=0;

void callback(const myPupilLab::ResultConstPtr& EMD, const myPupilLab::ResultConstPtr& fixation){
	std::string currentTruth;
	ros::NodeHandle nh;
	if(nh.getParam("/groundTruth", currentTruth)){
		if(EMD->s == currentTruth) emd_true++;
		else emd_false++;

		if(fixation->s == currentTruth) fixation_true++;
		else fixation_false++;

		cout << "EMD:" << endl;
		cout << "Positive: " << emd_true << ", Negative: " << emd_false << endl;

		cout << "Fixation:" << endl;
		cout << "Positive: " << fixation_true << ", Negative: " << fixation_false << endl;
		cout << "-------------------------------------------------------------------" << endl;
		cout << endl;
	}
	else cout<<"Rosparam /groundTruth is not set!";
}

int main(int argc, char **argv){
	ros::init(argc, argv, "resultsTabulator");

	ros::NodeHandle nh;
	message_filters::Subscriber<myPupilLab::Result> EMD_result_sub(nh, "final_verdict", 1);
	message_filters::Subscriber<myPupilLab::Result> fixation_result_sub(nh, "final_verdict_fixation", 1);

	typedef sync_policies::ApproximateTime<myPupilLab::Result, myPupilLab::Result> MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), EMD_result_sub, fixation_result_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();

	return 0;
}
