#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/core/core.hpp>
#include "opencv2/core/internal.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include "CvLSVMRead/_lsvmparser.h"
#include "CFFLD.h"

#include <sys/time.h>

timeval Start, Stop;

inline void start()
{
	gettimeofday(&Start, 0);
}

inline int stop()
{
	gettimeofday(&Stop, 0);
	
	timeval duration;
	timersub(&Stop, &Start, &duration);
	
	return duration.tv_sec * 1000 + (duration.tv_usec + 500) / 1000;
}

using namespace FFLD;
using namespace std;
using namespace cv;

void drawResult( Mat & img, vector<DetectionResult> vResults);

int main(int argc, char * argv[])
{
	string strXMLFile( "../../inriaperson_final.xml" );
	
	string strImage( "../../000061.jpg"  );
	Mat img;
	img = imread( strImage );
	double threshold =-0.50;

	vector<DetectionResult> vResults;

	CFFLD ffldDetector;

	start();

	ffldDetector.detector( img, strXMLFile, threshold, vResults );

	cout << "FFLD detection cost: " << stop() << " ms" << endl;

	drawResult( img, vResults );
}

void drawResult( Mat & img, vector<DetectionResult> vResults)
{
	Scalar red( 0, 0, 255 ), blue( 255, 0, 0 );

	for( int i = 0; i < vResults.size(); i ++ )
	{
		// Draw root
		rectangle( img, vResults.at(i).rtRoot, blue, 2 );
		for( int j = 0; j < vResults.at(i).vParts.size(); j ++ )
		{
			rectangle( img, vResults.at(i).vParts.at(j), red, 2 );
		}
	}
	imshow( "result", img );
	waitKey(0);
}