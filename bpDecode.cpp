#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include "random.h"
using namespace std;

int **A,**B,*z;
double **mes,**E,*r,*L;
int N,M,col,row;
double var;

int read_file()
{
	int i,j;
	ifstream fip("155_64_3_5.txt");
	if(!fip)
	{
		cout<<"can not open!"<<endl;
	}
	fip>>N;
	fip>>M;
	fip>>col;
	fip>>row;

	A=new int*[N+1];
	for(i=0;i<=N;i++)
		A[i]=new int[col+1];

	B=new int*[M+1];
	for(i=0;i<=M;i++)
		B[i]=new int[row+1];

	mes=new double*[M+1];
	for(i=0;i<=M;i++)
		mes[i]=new double[N+1];

	E=new double*[M+1];
	for(i=0;i<=M;i++)
		E[i]=new double[N+1];

	r=new double[N+1];

	L=new double[N+1];

	z=new int[N+1];

	for(i=1;i<N+1;i++)
		for(j=1;j<col+1;j++)
			fip>>A[i][j];

	for(i=1;i<M+1;i++)
		for(j=1;j<row+1;j++)
			fip>>B[i][j];

	return 1;

}

void init()
{
	int i,j;
	for(i = 0; i <= M; i ++)
	{
		for(j = 0; j <= N; j ++)
			E[i][j] = mes[i][j] = 0;
	}

	for(i = 0; i <= N; i ++)
		L[i] = r[i] = z[i] = 0;

	for(i = 1; i <= N; i ++) 
		r[i] = pow(-1, r[i]);

	for(i = 1; i <= N; i ++) 
	{
		r[i] += Gauss() * var ;
		r[i] = 2 * r[i] / (var * var);
	}

	for(i = 1; i <= N; i ++) 
	{
		for(j = 1; j <= M; j ++)
		{
			mes[j][i]=r[i];
		}
	}
}

void step1()
{
	int i,j;

	for(j=1;j<=M;j++)
	{
		for(i=1;i<=row;i++)
		{
			double den=1.0;
			for(int k=1;k<=row;k++)
			{
				if(k==i)
					continue;
				else
					den*=tanh(mes[j][B[j][k]]/2);
			}
			E[j][B[j][i]] = log( ( 1 + den ) / ( 1 - den ) );
		}
	}
}


void test()
{
	int i,j;
	for(i=1;i<=N;i++)
	{
		double t=0.0;
		for(j=1;j<=M;j++)
		{
			t+=E[j][i];
		}
		z[i]=(t+r[i]<=0)?1:0;
	}
}



void step2()
{
	int i,j,k;
	double m=0.0;
	for(i=1;i<=N;i++)
	{
		for(j=1;j<=col;j++)
		{
			double m=0.0;
			for(k=1;k<=col;k++)
			{
				if(k==j)
					continue;
				else
					m+=E[A[i][k]][i];
			}
			mes[A[i][j]][i]=m+r[i];
		}
	}
}


void main()
{
	ofstream fos("result.txt",ios::out);
	if(!fos)
	{
		cout<<"open error!"<<endl;
	}
	srand((unsigned)time(NULL));

	int i;
	if(read_file())
	{
		for(double SNR=2.0;SNR<=5;SNR+=0.1)
		{
			double No=0.0;
			cout<<SNR<<endl;
			No=pow(10,SNR/10);
			cout<<No<<endl;
			var=sqrt(1.0/((2.0*(N-M)/N)*No));
			cout<<var<<endl;
			int error=0;
			int frame=0;
			int tmp_ERR = 0;
			int t = 0;

			while(error<50)
			{
				frame++;
				init();
				int flag=0;
				int iter=0;
				while((!flag)&&(iter<1000))
				{
					t++;
					iter++;
					step1();
					test();
					int number_of_error=0;
					for(i=1;i<=N;i++)
					{
						if(z[i]){
							number_of_error++;
							tmp_ERR ++;
						}
					}
					if(number_of_error==0)
					{
						flag=1;
					}
					if(!flag)
					{
						step2();
					}
				}
				if(iter==1000)
					error++;				
			}
			for(i=1;i<=N;i++)
			{
				//cout<<z[i]<<" ";
			}
			cout<<"信噪比为"<<SNR<<"时,误帧率为 "<<(50.0/frame) <<endl;
			//cout << "信噪比为" << SNR << ",误码率为：" << tmp_ERR/(t*N) << endl;
			cout<<endl;
			fos<<SNR<<"\t"<<((double)50/frame)<<endl;
			fos << SNR << "\t" << ((double)tmp_ERR/50/N) << endl;
		}
		delete A,B,z,mes,E,r,L;

	}
	fos.close();
}
