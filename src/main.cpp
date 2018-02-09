#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "math.h"

using namespace std;
using namespace cv;

int main() {

	VideoCapture cap("H:\\programming\\wlibCV\\test\\IMG_4775.MOV");

	int Width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int Height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int max_frame = cap.get(CV_CAP_PROP_FRAME_COUNT); //フレーム数


	Mat source(Height, Width, CV_8UC1);
	Mat HIS_source(Height, Width, CV_8UC1);

	for (int frame = 0; frame<max_frame; frame++) {
		cap >> source;
		Mat disp = source.clone();
		cvtColor(source, source, CV_BGR2GRAY);

		if (frame>0) {

			vector<cv::Point2f> prev_pts;
			vector<cv::Point2f> next_pts;

			Size flowSize(100, 100); //ベクトルの数
			Point2f center = cv::Point(source.cols / 2., source.rows / 2.);
			for (int i = 0; i<flowSize.width; ++i) {
				for (int j = 0; j<flowSize.width; ++j) {
					Point2f p(i*float(source.cols) / (flowSize.width - 1),
						j*float(source.rows) / (flowSize.height - 1));
					prev_pts.push_back((p - center)*0.95f + center);
				}
			}

			Mat flow;
			vector<float> error;

			calcOpticalFlowFarneback(HIS_source, source, flow, 0.8, 10, 15, 3, 5, 1.1, 0);

			// オプティカルフローの表示
			std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
			for (; p != prev_pts.end(); ++p) {
				const cv::Point2f& fxy = flow.at<cv::Point2f>(p->y, p->x);
				cv::line(disp, *p, *p + fxy * 8, cv::Scalar(0), 1);
			}

			HIS_source = source.clone();

			imshow("vector", disp);
			imshow("source", HIS_source);

			int c = waitKey(1);
			if (c == 27)return 0;
		}

		cout << frame << endl;
		frame += 1;
	}
	waitKey(0);
	return 0;
}