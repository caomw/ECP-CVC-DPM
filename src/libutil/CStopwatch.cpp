#include "CStopwatch.h"

CStopwatch::CStopwatch( bool start_immediately )
	: m_tStart(0), m_tStop(0), m_bRunning(false)
{
	if( start_immediately )
	{
		Start(true);
	}
}

void CStopwatch::Start( bool reset )
{
	if( !m_bRunning )
	{
		if( reset )
		{
			m_tStart = clock();
		}

		m_bRunning = true;
	}
}

void CStopwatch::Stop()
{
	if( m_bRunning )
	{
		m_tStop = clock();
		m_bRunning = false;
	}
}

float CStopwatch::Elapsed() const
{
	float fTime;
	if( m_bRunning )
	{
		fTime = ( clock() - m_tStart ) / ( 1.0 * CLOCKS_PER_SEC ); 
	}
	else
	{
		fTime = ( m_tStop - m_tStart ) / ( 1.0 * CLOCKS_PER_SEC ); 
	}

	return fTime;
}
