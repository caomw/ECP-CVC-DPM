#include "CFFLD.h"

int CFFLD::detector( Mat img, string strModelFile, float threshold, vector<DetectionResult> & result )
{
	// For xml model;
    CvLSVMFilterObject** filters = 0;
    int kFilters = 0;
    int kComponents = 0;
    int* kPartFilters = 0;
    float* b = 0;
    float scoreThreshold = 0.f;
    int err_code = 0;

    Object::Name name = Object::PERSON;
	string results;
	string images(".");
    int nbNegativeScenes = -1;
	int padding = 12;
	int interval = 10;
	//double threshold =-0.50;
	double overlap = 0.5;

	// load xml file
    err_code = loadModel( strModelFile.c_str(), &filters, &kFilters, &kComponents, &kPartFilters, &b, &scoreThreshold );

    if( err_code != 0 )
    {
    	cout<<"Invalid xml model file."<<endl;
    	return 0;
    }

 	Mixture mixture;
 	mixture.importFromOpenCVModel( kFilters, kComponents, kPartFilters, b, filters );

	JPEGImage image( img );
	HOGPyramid pyramid(image, padding, padding, interval);

	// Initialize the Patchwork class
	if ( !Patchwork::Init((pyramid.levels()[0].rows() - padding + 15) & ~15,
						  (pyramid.levels()[0].cols() - padding + 15) & ~15) ) 
	{
		cerr << "\nCould not initialize the Patchwork class" << endl;
		return -1;
	}

	mixture.cacheFilters();
	
	// Compute the detections
	vector<Detection> detections;
	ofstream out;
	string file;
	detect(mixture, image.width(), image.height(), pyramid, threshold, overlap, file, out,
		   images, detections, result );
}

void CFFLD::detect(const Mixture & mixture, int width, int height, const HOGPyramid & pyramid, double threshold, double overlap, const string image, ostream & out, const string & images, vector<Detection> & detections, vector<DetectionResult> & vResult )
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
	{
		sizes[i] = mixture.models()[i].rootSize();
	}
	
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
						((y == rows - 1) || (x == cols - 1) || (score > scores[i](y + 1, x + 1)))) 
					{
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

	// Output the result to OpenCV Rect
	for( int j = 0; j < detections.size(); j ++ )
	{
			DetectionResult result;
			// The position of the root one octave below
			const int argmax = argmaxes[detections[j].l](detections[j].y, detections[j].x);
			const int x2 = detections[j].x * 2 - pyramid.padx();
			const int y2 = detections[j].y * 2 - pyramid.pady();
			const int l = detections[j].l - pyramid.interval();

			const double scale = pow(2.0, static_cast<double>(l) / pyramid.interval() + 2.0);
			//cout<<positions[argmax].size()<<endl;	
			for (int k = 0; k < positions[argmax].size(); ++k) 
			{
				const FFLD::Rectangle bndbox((positions[argmax][k][l](y2, x2)(0) - 
					pyramid.padx()) * scale + 0.5, 
					(positions[argmax][k][l](y2, x2)(1) - pyramid.pady()) * scale + 0.5,
					mixture.models()[argmax].partSize().second * scale + 0.5,
					mixture.models()[argmax].partSize().second * scale + 0.5 );
				Rect rtPart( bndbox.x_, bndbox.y_, bndbox.width_, bndbox.height_ );	
				//cout<<rtPart<<endl;
				result.vParts.push_back( rtPart );
			}

			Rect rtRoot( detections[j].x_, detections[j].y_, detections[j].width_, detections[j].height_ );
			result.rtRoot = rtRoot;

			vResult. push_back(result);
	}
}