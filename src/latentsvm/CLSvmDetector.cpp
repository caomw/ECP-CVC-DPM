#include "CLSvmDetector.h"

CLSvmDetector::CLSvmDetector()
{

}

CLSvmDetector::CLSvmDetector( string modelFile, int mode )
{
	if ( modelFile.empty() )
	{
		cout<<"Invalid model files."<<endl;
	}

	int i = 0;
	for( i = modelFile.size() - 1; i >= 0; i ++  )
	{

	}
}