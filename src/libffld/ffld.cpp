#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/core/core.hpp>
#include "opencv2/core/internal.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "CvLSVMRead/_lsvmparser.h"

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

// SimpleOpt array of valid options
enum
{
	OPT_HELP, OPT_MODEL, OPT_NAME, OPT_RESULTS, OPT_IMAGES, OPT_NB_NEG, OPT_PADDING, OPT_INTERVAL,
	OPT_THRESHOLD, OPT_OVERLAP
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	{ OPT_MODEL, "-m", SO_REQ_SEP },
	{ OPT_MODEL, "--model", SO_REQ_SEP },
	{ OPT_NAME, "-n", SO_REQ_SEP },
	{ OPT_NAME, "--name", SO_REQ_SEP },
	{ OPT_RESULTS, "-r", SO_REQ_SEP },
	{ OPT_RESULTS, "--results", SO_REQ_SEP },
	{ OPT_IMAGES, "-i", SO_REQ_SEP },
	{ OPT_IMAGES, "--images", SO_REQ_SEP },
	{ OPT_NB_NEG, "-z", SO_REQ_SEP },
	{ OPT_NB_NEG, "--nb-negatives", SO_REQ_SEP },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
	{ OPT_THRESHOLD, "-t", SO_REQ_SEP },
	{ OPT_THRESHOLD, "--threshold", SO_REQ_SEP },
	{ OPT_OVERLAP, "-v", SO_REQ_SEP },
	{ OPT_OVERLAP, "--overlap", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage()
{
	cout << "Usage: test [options] image.jpg, or\n       test [options] image_set.txt\n\n"
			"Options:\n"
			"  -h,--help               Display this information\n"
			"  -m,--model <file>       Read the input model from <file> (default \"model.txt\")\n"
			"  -n,--name <arg>         Name of the object to detect (default \"person\")\n"
			"  -r,--results <file>     Write the detection results to <file> (default none)\n"
			"  -i,--images <folder>    Draw the detections to <folder> (default none)\n"
			"  -z,--nb-negatives <arg> Maximum number of negative images to consider (default all)\n"
			"  -p,--padding <arg>      Amount of zero padding in HOG cells (default 12)\n"
			"  -e,--interval <arg>     Number of levels per octave in the HOG pyramid (default 10)\n"
			"  -t,--threshold <arg>    Minimum detection threshold (default -10)\n"
			"  -v,--overlap <arg>      Minimum overlap in non maxima suppression (default 0.5)"
		 << endl;
}

void draw( JPEGImage & image, const FFLD::Rectangle & rect, uint8_t r, uint8_t g, uint8_t b, int linewidth )
{
	if (image.empty() || rect.empty() || (image.depth() < 3))
		return;
	
	const int width = image.width();
	const int height = image.height();
	const int depth = image.depth();
	uint8_t * bits = image.bits();
	
	// Draw 2 horizontal lines
	const int top = min(max(rect.top(), 1), height - linewidth - 1);
	const int bottom = min(max(rect.bottom(), 1), height - linewidth - 1);
	
	for (int x = max(rect.left() - 1, 0); x <= min(rect.right() + linewidth, width - 1); ++x) 
	{
		if ( ( x != max(rect.left() - 1, 0 ) ) && 
			 ( x != min(rect.right() + linewidth, width - 1 ) ) ) 
		{
			for (int i = 0; i < linewidth; ++i) 
			{
				bits[((top + i) * width + x) * depth    ] = r;
				bits[((top + i) * width + x) * depth + 1] = g;
				bits[((top + i) * width + x) * depth + 2] = b;
				bits[((bottom + i) * width + x) * depth    ] = r;
				bits[((bottom + i) * width + x) * depth + 1] = g;
				bits[((bottom + i) * width + x) * depth + 2] = b;
			}
		}
		
		// Draw a white line below and above the line
		if ((bits[((top - 1) * width + x) * depth    ] != 255) &&
			(bits[((top - 1) * width + x) * depth + 1] != 255) &&
			(bits[((top - 1) * width + x) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[((top - 1) * width + x) * depth + i] = 255;
			}
		}
			
		if ((bits[((top + linewidth) * width + x) * depth    ] != 255) &&
			(bits[((top + linewidth) * width + x) * depth + 1] != 255) &&
			(bits[((top + linewidth) * width + x) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[((top + linewidth) * width + x) * depth + i] = 255;
			}
		}

		if ((bits[((bottom - 1) * width + x) * depth    ] != 255) &&
			(bits[((bottom - 1) * width + x) * depth + 1] != 255) &&
			(bits[((bottom - 1) * width + x) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[((bottom - 1) * width + x) * depth + i] = 255;
			}
		}

		if ((bits[((bottom + linewidth) * width + x) * depth    ] != 255) &&
			(bits[((bottom + linewidth) * width + x) * depth + 1] != 255) &&
			(bits[((bottom + linewidth) * width + x) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[((bottom + linewidth) * width + x) * depth + i] = 255;
			}
		}
	}
	
	// Draw 2 vertical lines
	const int left = min(max(rect.left(), 1), width - linewidth - 1);
	const int right = min(max(rect.right(), 1), width - linewidth - 1);
	
	for (int y = max(rect.top() - 1, 0); y <= min(rect.bottom() + linewidth, height - 1); ++y) 
	{
		if ( ( y != max(rect.top() - 1, 0 ) ) && 
			 ( y != min(rect.bottom() + linewidth, height - 1 ) ) ) 
		{
			for (int i = 0; i < linewidth; ++i) 
			{
				bits[(y * width + left + i) * depth    ] = r;
				bits[(y * width + left + i) * depth + 1] = g;
				bits[(y * width + left + i) * depth + 2] = b;
				bits[(y * width + right + i) * depth    ] = r;
				bits[(y * width + right + i) * depth + 1] = g;
				bits[(y * width + right + i) * depth + 2] = b;
			}
		}
		
		// Draw a white line left and right the line
		if ((bits[(y * width + left - 1) * depth    ] != 255) &&
			(bits[(y * width + left - 1) * depth + 1] != 255) &&
			(bits[(y * width + left - 1) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[(y * width + left - 1) * depth + i] = 255;
			}
		}

		if ((bits[(y * width + left + linewidth) * depth    ] != 255) &&
			(bits[(y * width + left + linewidth) * depth + 1] != 255) &&
			(bits[(y * width + left + linewidth) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[(y * width + left + linewidth) * depth + i] = 255;
			}
		}

		if ((bits[(y * width + right - 1) * depth    ] != 255) &&
			(bits[(y * width + right - 1) * depth + 1] != 255) &&
			(bits[(y * width + right - 1) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[(y * width + right - 1) * depth + i] = 255;
			}
		}

		if ((bits[(y * width + right + linewidth) * depth    ] != 255) &&
			(bits[(y * width + right + linewidth) * depth + 1] != 255) &&
			(bits[(y * width + right + linewidth) * depth + 2] != 255))
		{
			for (int i = 0; i < 3; ++i)
			{
				bits[(y * width + right + linewidth) * depth + i] = 255;
			}
		}
	}
}

void detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid,
			double threshold, double overlap, const string image, ostream & out,
			const string & images, vector<Detection> & detections)
{
	// Compute the scores
	vector<HOGPyramid::Matrix> scores;
	vector<Mixture::Indices> argmaxes;
	vector<vector<vector<Model::Positions> > > positions;
	
	if (!images.empty())
	{
		mixture.convolve(pyramid, scores, argmaxes, &positions);
	}
	else
	{
		mixture.convolve(pyramid, scores, argmaxes);
	}
	

	//cout<<"conv"<<endl;
	// Cache the size of the models
	vector<pair<int, int> > sizes(mixture.models().size());
	
	for (int i = 0; i < sizes.size(); ++i)
		sizes[i] = mixture.models()[i].rootSize();
	
	// For each scale
	for (int i = pyramid.interval(); i < scores.size(); ++i) 
	{
		// Scale = 8 / 2^(1 - i / interval)
		const double scale = pow(2.0, static_cast<double>(i) / pyramid.interval() + 2.0);
		
		const int rows = scores[i].rows();
		const int cols = scores[i].cols();
		
		for (int y = 0; y < rows; ++y) 
		{
			for (int x = 0; x < cols; ++x) 
			{
				const float score = scores[i](y, x);
				
				if (score > threshold) 
				{
					if (((y == 0) || (x == 0) || (score > scores[i](y - 1, x - 1))) &&
						((y == 0) || (score > scores[i](y - 1, x))) &&
						((y == 0) || (x == cols - 1) || (score > scores[i](y - 1, x + 1))) &&
						((x == 0) || (score > scores[i](y, x - 1))) &&
						((x == cols - 1) || (score > scores[i](y, x + 1))) &&
						((y == rows - 1) || (x == 0) || (score > scores[i](y + 1, x - 1))) &&
						((y == rows - 1) || (score > scores[i](y + 1, x))) &&
						((y == rows - 1) || (x == cols - 1) || (score > scores[i](y + 1, x + 1)))) {
						FFLD::Rectangle bndbox((x - pyramid.padx()) * scale + 0.5,
											   (y - pyramid.pady()) * scale + 0.5,
											   sizes[argmaxes[i](y, x)].second * scale + 0.5,
											   sizes[argmaxes[i](y, x)].first * scale + 0.5);
						
						// Truncate the object
						bndbox.setX(max(bndbox.x(), 0));
						bndbox.setY(max(bndbox.y(), 0));
						bndbox.setWidth(min(bndbox.width(), width - bndbox.x()));
						bndbox.setHeight(min(bndbox.height(), height - bndbox.y()));
						
						if (!bndbox.empty())
						{
							detections.push_back(Detection(score, i, x, y, bndbox));
						}
					}
				}
			}
		}
	}
	
	// Non maxima suppression
	sort(detections.begin(), detections.end());
	
	for (int i = 1; i < detections.size(); ++i)
		detections.resize(remove_if(detections.begin() + i, detections.end(),
									Intersector(detections[i - 1], overlap, true)) -
						  detections.begin());
	
	// Print the detection
	const size_t lastDot = image.find_last_of('.');
	
	string id = image.substr(0, lastDot);
	
	const size_t lastSlash = id.find_last_of("/\\");
	
	if (lastSlash != string::npos)
		id = id.substr(lastSlash + 1);


	
	if (out) 
	{
#pragma omp critical
		for (int i = 0; i < detections.size(); ++i)
		{
			out << id << ' ' << detections[i].score << ' ' << (detections[i].left() + 1) << ' '
				<< (detections[i].top() + 1) << ' ' << (detections[i].right() + 1) << ' '
				<< (detections[i].bottom() + 1) << endl;
		}
	}
	
	if (!images.empty()) 
	{
		JPEGImage im(image);
		
		for (int j = 0; j < detections.size(); ++j) 
		{
			// The position of the root one octave below
			const int argmax = argmaxes[detections[j].l](detections[j].y, detections[j].x);
			const int x2 = detections[j].x * 2 - pyramid.padx();
			const int y2 = detections[j].y * 2 - pyramid.pady();
			const int l = detections[j].l - pyramid.interval();
			
			// Scale = 8 / 2^(1 - j / interval)
			const double scale = pow(2.0, static_cast<double>(l) / pyramid.interval() + 2.0);
			
			for (int k = 0; k < positions[argmax].size(); ++k) 
			{
				const FFLD::Rectangle bndbox((positions[argmax][k][l](y2, x2)(0) - pyramid.padx()) *
											 scale + 0.5,
											 (positions[argmax][k][l](y2, x2)(1) - pyramid.pady()) *
											 scale + 0.5,
											 mixture.models()[argmax].partSize().second * scale + 0.5,
											 mixture.models()[argmax].partSize().second * scale + 0.5);
				
				draw(im, bndbox, 0, 0, 255, 2);
			}
			
			// Draw the root last
			draw(im, detections[j], 255, 0, 0, 2);
		}
		
		im.save(images + '/' + id + ".jpg");
	}
}

bool compareMixture( Mixture mixture1, Mixture mixture2 );

// Test a mixture model (compute a ROC curve)
int main(int argc, char * argv[])
{
	// For xml model;
    CvLSVMFilterObject** filters = 0;
    int kFilters = 0;
    int kComponents = 0;
    int* kPartFilters = 0;
    float* b = 0;
    float scoreThreshold = 0.f;
    int err_code = 0;
	string strXMLFile( "../../inriaperson_final.xml" );

	// Default parameters
	string model("../../inria_person.txt");
	Object::Name name = Object::PERSON;
	string results;
	string images(".");
	int nbNegativeScenes = -1;
	int padding = 12;
	int interval = 10;
	double threshold =-0.50;
	double overlap = 0.5;

	// Parse the parameters
	CSimpleOpt args(argc, argv, SOptions);
	
	// Try to open the mixture
	ifstream in( model.c_str(), ios::binary );

	// load xml file
    err_code = loadModel(strXMLFile.c_str(), &filters, &kFilters, &kComponents, &kPartFilters, &b, &scoreThreshold);
    //cout<<kPartFilters[2]<<endl;

    if( err_code != 0 )
    {
    	cout<<"Invalid xml model file."<<endl;
    	return 0;
    }

	if ( !in.is_open() ) 
	{
		showUsage();
		cerr << "\nInvalid model file " << model << endl;
		return -1;
	}
	
	Mixture mixture;
	in >> mixture;
	
    Mixture testmixture;
    testmixture.importFromOpenCVModel( kFilters, kComponents, kPartFilters, b, filters );

    compareMixture( mixture, testmixture );

	if ( mixture.empty() ) 
	{
		showUsage();
		cerr << "\nInvalid model file " << model << endl;
		return -1;
	}
	
	// The image/dataset

	// Read image
	const string file( "../../000061.jpg" );
	Mat img;
	img = imread( file );
	//imshow( "img", img );
	//waitKey(0);

	const size_t lastDot = file.find_last_of('.');
	
	if ((lastDot == string::npos) ||
		((file.substr(lastDot) != ".jpg") && 
		(file.substr(lastDot) != ".txt"))) 
	{
		showUsage();
		cerr << "\nInvalid file " << file << ", should be .jpg or .txt" << endl;
		return -1;
	}
	
	// Try to open the results
	ofstream out;
	
	if (!results.empty()) 
	{
		out.open(results.c_str(), ios::binary);
		
		if (!out.is_open()) 
		{
			showUsage();
			cerr << "\nInvalid results file " << results << endl;
			return -1;
		}
	}
	

	// Main process start
	
	JPEGImage image( img );
	
	if (image.empty()) 
	{
		showUsage();
		cerr << "\nInvalid image " << file << endl;
		return -1;
	}
	
	// Compute the HOG features
	start();
	
	HOGPyramid pyramid(image, padding, padding, interval);
	
	if (pyramid.empty()) 
	{
		showUsage();
		cerr << "\nInvalid image " << file << endl;
		return -1;
	}
	
	cout << "Computed HOG features in " << stop() << " ms" << endl;
	
	// Initialize the Patchwork class
	start();
	
	if ( !Patchwork::Init((pyramid.levels()[0].rows() - padding + 15) & ~15,
						  (pyramid.levels()[0].cols() - padding + 15) & ~15) ) 
	{
		cerr << "\nCould not initialize the Patchwork class" << endl;
		return -1;
	}
	
	cout << "Initialized FFTW in " << stop() << " ms" << endl;
	
	start();
	
	mixture.cacheFilters();
	
	cout << "Transformed the filters in " << stop() << " ms" << endl;
	
	// Compute the detections
	start();
	
	vector<Detection> detections;
	
	detect(mixture, image.width(), image.height(), pyramid, threshold, overlap, file, out,
		   images, detections);
	
	cout << "Computed the convolutions and distance transforms in " << stop() << " ms" << endl;
	
	return EXIT_SUCCESS;
}


bool compareMixture( Mixture mixture1, Mixture mixture2 )
{
	// compare size
	int size1 = mixture1.models_.size();
	int size2 = mixture2.models_.size();

	if( (size1 - size2) != 0 )
	{
		cout<<"size diff."<<endl;
		return false;
	}

	// parts comparison
	for( int i = 0; i < size1; i ++ )
	{
		cout<<"Model "<<i<<":"<<endl;
		if( mixture1.models_.at(i).bias_ - mixture2.models_.at(i).bias_ != 0 )
		{
			cout<<"Mix 1: "<<mixture1.models_.at(i).bias_<<", "
			    <<"Mix 2: "<<mixture2.models_.at(i).bias_<<"."<<endl;
			cout<<mixture1.models_.at(i).bias_ - mixture2.models_.at(i).bias_<<endl;
			cout<<"Bias diff."<<endl;
		}

		// Parts size
		if( mixture1.models_.at(i).parts_.size() -
			mixture2.models_.at(i).parts_.size() != 0 )
		{
			cout<<"Mix 1: "<<mixture1.models_.at(i).parts_.size()<<", "
			    <<"Mix 2: "<<mixture1.models_.at(i).parts_.size()<<"."<<endl;
			cout<< mixture1.models_.at(i).parts_.size() -
				   mixture2.models_.at(i).parts_.size()<<endl;
			cout<<"Parts number diff."<<endl;
		}
		else
		{
			for( int j = 0; j < mixture1.models_.at(i).parts_.size(); j ++ )
			{
				cout<<"Part number: "<<j<<"."<<endl;

				// cell size
				if( mixture1.models_.at(i).parts_.at(j).filter.cols() - mixture2.models_.at(i).parts_.at(j).filter.cols() != 0 )
				{
					cout<<"Cell diff."<<endl;			
				}

				if( mixture1.models_.at(i).parts_.at(j).filter.rows() - mixture2.models_.at(i).parts_.at(j).filter.rows() != 0 )
				{
					cout<<"Cell diff."<<endl;			
				}

				// offset
				if( mixture1.models_.at(i).parts_.at(j).offset(0) - mixture2.models_.at(i).parts_.at(j).offset(0) != 0 )
				{
					cout<<mixture1.models_.at(i).parts_.at(j).offset(0)<<"Mix1"<<endl;;
					cout<<mixture2.models_.at(i).parts_.at(j).offset(0)<<"Mix2"<<endl;;
					cout<<"X Offset diff."<<endl;			
				}
				if( mixture1.models_.at(i).parts_.at(j).offset(1) - mixture2.models_.at(i).parts_.at(j).offset(1) != 0 )
				{
					cout<<mixture1.models_.at(i).parts_.at(j).offset(1)<<"Mix1"<<endl;;
					cout<<mixture2.models_.at(i).parts_.at(j).offset(1)<<"Mix2"<<endl;;
					cout<<"Y Offset diff."<<endl;			
				}

				// deformation
				// for( int k = 0; k < 4; k ++ )
				// {
				// 	if( mixture1.models_.at(i).parts_.at(j).deformation(k) - mixture2.models_.at(i).parts_.at(j).deformation(k) != 0 )
				// 	{
				// 		cout<<mixture1.models_.at(i).parts_.at(j).deformation(k)<<"Mix1"<<endl;;
				// 		cout<<mixture2.models_.at(i).parts_.at(j).deformation(k)<<"Mix2"<<endl;;
				// 		cout<<"Deformation diff."<<k<<endl;			
				// 	}
				// }

				
			}
		}
	}
}