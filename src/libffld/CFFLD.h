#ifndef _CFFLD_H
#define _CFFLD_H

#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"
#include "HOGPyramid.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>
#include "string.h"

#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/core/core.hpp>
#include "opencv2/core/internal.hpp"
//#include "opencv2/opencv_modules.hpp."
#include "opencv2/highgui/highgui.hpp"

#include "CvLSVMRead/_lsvmparser.h"

using namespace FFLD;
using namespace std;
using namespace cv;

namespace FFLD
{

typedef struct DetectionResult
{
	Rect rtRoot;
	std::vector<Rect> vParts;
}_detectionResult_;

class CFFLD
{
public:
	enum
	{
		OPT_HELP, OPT_MODEL, OPT_NAME, OPT_RESULTS, OPT_IMAGES, OPT_NB_NEG, OPT_PADDING, OPT_INTERVAL,
		OPT_THRESHOLD, OPT_OVERLAP
	};

	struct Detection : public FFLD::Rectangle
	{
		HOGPyramid::Scalar score;
		int l;
		int x;
		int y;
	
		Detection() : score(0), l(0), x(0), y(0)
		{
		}
	
		Detection(HOGPyramid::Scalar score, int l, int x, int y, FFLD::Rectangle bndbox) :
		FFLD::Rectangle(bndbox), score(score), l(l), x(x), y(y)
		{
		}
	
		bool operator<(const Detection & detection) const
		{
			return score > detection.score;
		}
	};



	CFFLD()
	{};
	~CFFLD()
	{};

	void detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid, double threshold, double overlap, const string image, ostream & out, const string & images, vector<Detection> & detections, vector<DetectionResult> & vResult);

	void draw( JPEGImage & image, const FFLD::Rectangle & rect, uint8_t r, uint8_t g, uint8_t b, int linewidth );

	int detector( Mat image, string strModelFile, float threshold, vector<DetectionResult> & result );

private:
};

}

#endif