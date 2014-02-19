#ifndef CSTOPWATCH_H_
#define CSTOPWATCH_H_

#include <ctime>

using namespace std;
using std::clock;

class CStopwatch
{
public:
	explicit CStopwatch( bool start_immediately = false );

	void Start( bool reset = false );
	void Stop();

	float Elapsed() const;

private:
	clock_t m_tStart, m_tStop;
	bool m_bRunning;
};

#endif
