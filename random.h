#include<iostream>
#include<cmath>
using namespace std;
double Gauss()
{
	static double v1 ,v2, fac;
    static int phase = 0 ;
	double x , s;
	if(phase == 0 )
	{
		do
		{
			double u1 = (double) rand() / RAND_MAX;
			double u2 = (double) rand() / RAND_MAX;
			v1 = 1 - 2 * u1 ;
			v2 = 1 - 2 * u2 ;
			s = v1 * v1 + v2 * v2 ;
		}while(s >= 1 || s <= 0);
		fac = sqrt( -2.0 * log(s) / s) ;
		x = v1 * fac ;
	}
	else 
		x = v2 * fac ;
	phase = 1 - phase ;
	return x ;
}
