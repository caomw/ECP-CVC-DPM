#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "CFFLD.h"

using namespace std;
using namespace cv;
using namespace FFLD;

#define FFLD_MODE 0
#define OPENCV_MODE 1

class CLSvmDetector
{
// Member variables
public:
	struct DETECTION_RESULT
	{
		Rect 			rootRect;
		vector<Rect> 	partsRect;
		int 			classId;
		string 			className;
		Scalar 			drawColor;
		float 			score;
	}_DETECTION_RESULT_;

	vector<string> 				m_vFileNames;
	vector<DETECTION_RESULT> 	m_vDetectionResult;
	int 		   				m_nMode;

private:
	vector<LatentSvmDetector::ObjectDetection> m_vCvDetectedResult;
	vector<DetectionResult> 				   m_vFFLDResult;

// Member functions
public:
	CLSvmDetector();
	CLSvmDetector( string modelFile, int mode = OPENCV_MODE );
	~CLSvmDetector();

	void setMode(int mode);
	void detect();
	void detect_opencv();
	void detect_ffld();
	
	void drawDetection();
	void drawPartsFilter();
	void drawAll();
	
	void readModelfromFile();
	void readModelfromDirectory();

	void generateRandomColor();
private:

};