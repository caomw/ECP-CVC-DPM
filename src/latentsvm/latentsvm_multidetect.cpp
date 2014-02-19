#define FFLD_MODE 0
#define OPENCV_MODE 1
#define MODE_OPTION FFLD_MODE

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

//#include "CvLSVMRead/_lsvmparser.h"
#include "CFFLD.h"
#include "CLSvmDetector.h"

#include <sys/time.h>

#include <dirent.h>

#ifdef HAVE_CVCONFIG_H
#include <cvconfig.h>
#endif

#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

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
static void detectAndDrawObjects( Mat& image, LatentSvmDetector& detector, const vector<Scalar>& colors, float overlapThreshold, int numThreads );

int main(int argc, char * argv[])
{
    string strXMLFile( "../../data/models/inriaperson_final.xml" );
    string strImage( "../../data/pics/000061.jpg"  );
    double threshold = -0.40;
    
    if( argc == 5 )
    {
        string strMode( argv[0] );
        string strPics( argv[1] );
        string strModel( argv[2] );
        string strThreshold( argv[3] );
        
    }

    Mat img;
    img = imread( strImage );
    int numThreads = 8;

    switch( MODE_OPTION )
    {
    case FFLD_MODE:
    {
        vector<DetectionResult> vResults;

        CFFLD ffldDetector;

        start();

        ffldDetector.detector( img, strXMLFile, threshold, vResults );

        cout << "FFLD detection cost: " << stop() << " ms" << endl;

        drawResult( img, vResults );
        return 0;
    }

    case OPENCV_MODE:
    {
        vector<string> vModelFiles;
        vModelFiles.push_back( strXMLFile );
        LatentSvmDetector detector( vModelFiles );
        vector<LatentSvmDetector::ObjectDetection> detections;

        start();
        detector.detect( img, detections, threshold, numThreads );
        cout << "OpenCV DPM detection cost: " << stop() << " ms" << endl;

        for( size_t i = 0; i < detections.size(); i++ )
        {
            const LatentSvmDetector::ObjectDetection& od = detections[i];
            rectangle( img, od.rect, Scalar(255, 0, 0), 2 );
        }

        imshow( "result", img );
        waitKey(0);
    }
    }

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

static void detectAndDrawObjects( Mat& image, LatentSvmDetector& detector, const vector<Scalar>& colors, float overlapThreshold, int numThreads )
{
    vector<LatentSvmDetector::ObjectDetection> detections;

    TickMeter tm;
    tm.start();
    detector.detect( image, detections, overlapThreshold, numThreads);
    tm.stop();

    cout << "Detection time = " << tm.getTimeSec() << " sec" << endl;

    const vector<string> classNames = detector.getClassNames();
    CV_Assert( colors.size() == classNames.size() );

    for( size_t i = 0; i < detections.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& od = detections[i];
        rectangle( image, od.rect, colors[od.classID], 3 );
    }
    // put text over the all rectangles
    for( size_t i = 0; i < detections.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& od = detections[i];
        putText( image, classNames[od.classID], Point(od.rect.x+4,od.rect.y+13), FONT_HERSHEY_SIMPLEX, 0.55, colors[od.classID], 2 );
    }
}