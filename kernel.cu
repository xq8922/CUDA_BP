#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#define FILE_NAME "155_64_3_5.txt"
__constant__  int MaxCheckdegree = 6;//检验节点的度
__constant__ int MaxVarDegree = 3;//变量节点的度
using namespace std;

int **A,**B,*z;
float **mes,**E,*r,*L;
int N,M,col,row;
float var;

int readFile();
void init();

//设置校验信息
__global__ void SetCheckMessage(float *E,float *Mes,int *B,int width)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//计算当前行号

	float den=1.0;
	if(xIndex < width){
		for(int i = 1;i <= MaxCheckdegree;i++){//MaxCheckedegree is row .
			int innerIdx = xIndex*MaxCheckdegree+B[i];
			//den *= tanh(mes[xIndex][B[xIndex][i]]/2);
			//convert to one dimension
			den *= tanh(Mes[innerIdx]/2);
		}
		for(int i = 1;i <= MaxCheckdegree;i ++){
			int innerIdx = xIndex*MaxCheckdegree+B[i];
			den /= tanh(Mes[innerIdx]/2);
			E[innerIdx] = log( ( 1 + den ) / ( 1 - den ) );
		}
	}
}

//为z赋值，z为得到的译码结果
__global__ void setZ(float *E,float *z,float *r,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(xIndex < width){
		float t = 0.0;
		//计算当前列号
		int column = 0;
		for(int i = 1;i <= M;i ++){
			t += E[i*MaxCheckdegree + xIndex];
		}
		z[column] = (t + r[column] <= 0)?1:0;
	}
}

//计算mes
__global__ void setBitMes(float *A,int col,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(xIndex < width){
		//计算列号
		int column = 0;
		float m = 0.0;
		for(int i = 1;i <= col;i ++){
			m += E[A[xIndex * col+i]+xIndex];
		}
	}
	__syncthreads();
	if(xIndex < width){
		for(int i = 1;i <= col;i ++){
			m -= E[A[xIndex * col+i]+xIndex];
			mes[A[xIndex * col+i]+xIndex] = m + r[i];
		}	
	}
}

int main()
{	
	srand((unsigned)time(NULL));
	clock_t begin = clock();
	if(readFile()){
		ofstream fos("result.txt",ios::out);
		if(!fos){
			cout << "create file failed" << endl;
			return 0;
		}
		float* h_varToCheck;
		h_varToCheck = (float*)malloc(sizeof(float)*M*row);

		float* h_checkToVar;
		h_checkToVar = (float*)malloc(sizeof(float)*M*row);

		float* h_mes;
		h_mes = (float*)malloc(sizeof(float)*M*N);
		float* d_mes;
		cudaMalloc((void**)&d_mes,sizeof(float)*M*N);

		float* h_E;
		h_E = (float*)malloc(sizeof(float)*M*N);
		float* d_E;
		d_E = cudaMalloc((void**)&d_E,sizeof(float)*M*N);

		for(float snr = 2.0;snr <= 5;snr += 0.1){
			cout << snr << endl;
			float var_temp = 0.0;
			var_temp = pow(10,snr/10);
			var = sqrt(1.0/((2.0*(N-M)/N)*var_temp));
			int error = 0;
			int frame = 0;
			int tmpErr = 0;
			int t = 0;
			
			while(error < 50){
				frame++;
				init();
				int flag = 0;
				int iter = 0;
				memset(h_varToCheck,0,sizeof(float)*M*row);
				memset(h_checkToVar,0,sizeof(float)*N*col);
				while((!flag) && (iter < 1000)){
					t ++;
					iter ++;
					for(int i = 1;i <= M;i ++){
						for(int j = 1;j <= row;j ++){
							int index = (i-1)*row+j;
							h_varToCheck[index] = B[i][j];
						}
					}
					for(int i = 1;i <= N;i ++){
						for(int j = 1;j <= col;j ++){
							int index = (i-1)*col+j;
							h_checkToVar[index] = A[i][j];
						}
					}
					//set Mes
					for(int i = 1;i <= M;i ++){
						for(int j = 1;j <= N;j ++){
							int index = (i-1)*N+j;
							h_mes[index] = mes[i][j];
						}
					}
					
					//主要的gpu逻辑.

					
					cudaThreadSynchronize();
					//set E
					for(int i = 1;i <= M*N;i ++){
						E[i/N+1][i%N] = h_E[i];
					}
					int codeErrNum = 0;
					//此处还可以改为并行
					for(int i = 1;i <= N;i ++){
						if(z[i]){
							codeErrNum ++;
							tmpErr ++;
						}
					}
					if(codeErrNum == 0){
						flag = 1;
					}
					if(flag == 0){
						//gpu
					}
				}
				if(iter == 1000){
					error++;
				}
			}
			cout << "snr = " << snr << ",error frame:" << (50.0/frame) << endl;
			fos << snr << "\t" << ((float)50/frame) << endl;
			fos << snr << "\t" << ((float)tmpErr/50/N) << endl;
		}
		clock_t end = clock();
		float cost = (float)(end - begin)/CLOCKS_PER_SEC;
		delete A,B,z,mes,E,r,L;
		fos.close();
	}	

    return 0;
}

//读文件以及赋初值
int readFile(){
	ifstream fip(FILE_NAME);
	if(!fip){
		cout << "Can not find txt file!" << endl;
		return 0;
	}
	//此处忘记标准格式了，应该还有两行直接删掉了
	fip >> N;
	fip >> M;
	fip >> col;
	fip >> row;
	A = new int*[N+1];
	for(int i = 0;i <= N;i ++){
		A[i] = new int[col+1];
	}
	B = new int*[M+1];
	for(int i = 0;i <= M;i ++){
		B[i] = new int[row+1];
	}
	mes = new float*[M+1];
	for(int i = 0;i <= M;i ++){
		mes[i] = new float[N+1];
	}
	E = new float*[M+1];
	for(int i = 0;i <= M;i ++){
		E[i] = new float[N+1];
	}
	r = new float[N+1];
	L = new float[N+1];
	z = new int[N+1];
	for(int i = 1;i < N+1;i ++){
		for(int j = 1;j < col +1;j ++){
			fip >> A[i][j];
		}
	}
	for(int i = 1;i < M +1;i ++){
		for(int j = 1;j < row +1;j ++){
			fip >> B[i][j];
		}
	}
	return 1;
}

//初始化各变量
void init(){
	for(int i = 0;i <= M;i ++){
		for(int j = 0;j <= N;j ++){
			E[i][j] = mes[i][j] = 0;
		}
	}
	for(int i = 0;i <= N;i ++){
		L[i] = r[i] = z[i] = 0;
	}
	for (int i = 1;i <= N;i ++){
		r[i] = pow(-1,r[i]);
	}
	for(int i = 1;i <= N;i ++){
		r[i] += Gauss() * var;
		r[i] = 2 * r[i] / (var*var);
	}
	for(int i = 1;i <= N;i ++){
		for(int j = 1;j <= M;j ++){
			mes[j][i] = r[i];
		}
		//将mes分解为行
	}
}