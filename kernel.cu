#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#define FILE_NAME "155_64_3_5.txt"
#define BLOCK_DIM_32 32
#define BLOCK_DIM_64 64
//#define BLOCK_DIM_X 6;//校验节点的度
//#define BLOCK_DIM_Y 3;//变量节点的度
//__constant__  int MaxCheckdegree = 5;//检验节点的度
//__constant__ int MaxVarDegree = 3;//变量节点的度
using namespace std;

int **A,**B,*z;
float **mes,**E,*r,*L;
int N,M,col,row;
float var;

void init();
//输出二位矩阵
void printx(float**x,int m,int n);
//输出一维浮点型数组
void print_1dimension_float(float* x,int m);
//输出一维整形数组
void print_1dimension_int(int* x,int m);
//读取H矩阵
int readH();


//设置校验信息
__global__ void SetCheckMessage(float *E,float *Mes,int *B,int degree,int N,int width)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(xIndex < width){
		for(int j = 1;j <= degree;j ++){
			float temp = 1.0;
			for(int i = 1;i <= degree;i ++){
				if(i != j){
					int b_idx = xIndex*degree+i;
					int mes_idx = xIndex*N+B[b_idx];
					temp *= tanh(Mes[mes_idx]/2);
				}
			}
			int b_idx = xIndex*degree+j;
			int e_idx = xIndex*N+B[b_idx];
			E[e_idx] = log( ( 1 + temp ) / ( 1 - temp ) );
		}
	}
}

//为z赋值，z为得到的译码结果.重新分配空间
__global__ void setZ(float *E,int *z,float *r,int M,int N,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(xIndex < width){
		float t = 0.0;
		for(int i = 0;i < M;i ++){
			int e_idx = i*N+xIndex+1;
			t += E[e_idx]; 
		}
		z[xIndex+1] = (t + r[xIndex+1] <= 0)?1:0;
	}
}

__global__ void setZ_float(float *E,float *z,float *r,int M,int N,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(xIndex < width){
		float t = 0.0;
		for(int i = 0;i < M;i ++){
			int e_idx = i*N+xIndex+1;
			t += E[e_idx];
		}
		z[xIndex+1] = t;
	}
}

//计算mes
__global__ void setBitMes(int *A,float* mes,float* r,float *E,int degree,int N,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(xIndex < width){	
		for(int i = 1;i <= degree;i ++){
			float m = 0.0;
			for(int j = 1;j <= degree;j ++){
				if(i != j){
					int a_idx = xIndex*degree+j;
					int e_idx = (A[a_idx]-1)*N+xIndex+1;
					m = m+E[e_idx];
				}
			}
			int a_idx = xIndex*degree+i;
			int mes_idx = (A[a_idx]-1)*N+xIndex+1;
			//mes[mes_idx] = m;
			mes[mes_idx] = m + r[xIndex+1];
		}		
	}
}

int main()
{	
	srand((unsigned)time(NULL));
	clock_t begin = clock();
	
	if(readH()){	
		int* h_varToCheck;
		h_varToCheck = (int*)malloc(sizeof(int)*((M+1)*(row+1)));
		int* d_varToCheck;
		cudaMalloc((void**)&d_varToCheck,sizeof(int)*((M+1)*(row+1)));

		int* h_checkToVar;
		h_checkToVar = (int*)malloc(sizeof(int)*((N+1)*(col+1)));
		int* d_checkToVar;
		cudaMalloc((void**)&d_checkToVar,sizeof(int)*((N+1)*(col+1)));

		float* h_mes;
		h_mes = (float*)malloc(sizeof(float)*((M+1)*(N+1)));
		float* d_mes;
		cudaMalloc((void**)&d_mes,sizeof(float)*((M+1)*(N+1)));

		float* h_E;
		h_E = (float*)malloc(sizeof(float)*((M+1)*(N+1)));
		float* d_E;
		cudaMalloc((void**)&d_E,sizeof(float)*((M+1)*(N+1)));
		float* h_E_tmp;
		h_E_tmp = (float*)malloc(sizeof(float)*((M+1)*(N+1)));

		int* d_z;
		cudaMalloc((void**)&d_z,sizeof(int)*(N+1));

		float* d_r;
		cudaMalloc((void**)&d_r,sizeof(float)*(N+1));

		memset(h_varToCheck,0,sizeof(float)*((M+1)*(row+1)));
		memset(h_checkToVar,0,sizeof(int)*((N+1)*(col+1)));
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
			
		//test h_checktovar
		//print_1dimension_int(h_checkToVar,N*col);
		dim3 threads,grid;

		for(float snr = 2.0;snr <= 4.5;snr += 0.1){
			printf("snr = %f\n",snr);
			var = sqrt(1.0/((2.0*(N-M)/N)*pow(10,snr/10)));

			int error = 0;
			int frame = 0;
			int tmpErr = 0;
			
			while(error < 50){
				frame++;
				init();
				memset(h_E,0,sizeof(float)*((M+1)*(N+1)));
				memset(h_mes,0,sizeof(float)*((M+1)*(N+1)));
				//set E ,mes
				for(int i = 1;i <= M;i ++){
					for(int j = 1;j <= N;j ++){
						int index = (i-1)*N + j;
						h_E[index] = E[i][j];
						h_mes[index] = mes[i][j];
					}
				}
				
				cudaMemset(d_E,0.0,sizeof(float)*(M+1)*(N+1));
				cudaMemset(d_mes,0.0,sizeof(float)*(M+1)*(N+1));
				cudaMemset(d_varToCheck,0.0,sizeof(float)*((M+1)*(row+1)));
				cudaMemset(d_r,0,sizeof(float)*(N+1));
				cudaMemset(d_checkToVar,0.0,sizeof(float)*((N+1)*(col+1)));

				cudaMemcpy(d_E,h_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyHostToDevice);
				cudaMemcpy(d_mes,h_mes,sizeof(float)*((M+1)*(N+1)),cudaMemcpyHostToDevice);
				cudaMemcpy(d_varToCheck,h_varToCheck,sizeof(float)*((M+1)*(row+1)),cudaMemcpyHostToDevice);
				cudaMemcpy(d_r,r,sizeof(float)*(N+1),cudaMemcpyHostToDevice);
				cudaMemcpy(d_checkToVar,h_checkToVar,sizeof(float)*((N+1)*(col+1)),cudaMemcpyHostToDevice);

				
				int flag = 0;
				int iter = 0;
				while((!flag) && (iter < 1000)){
					iter ++; 
					//主要的gpu逻辑.
					threads.x = BLOCK_DIM_32;
					threads.y = 1;
					grid.x = (M+BLOCK_DIM_32-1)/BLOCK_DIM_32;
					grid.y = 1;					

					/*cudaMemcpy(h_checkToVar,d_checkToVar,sizeof(int)*(N*col+1),cudaMemcpyDeviceToHost);
					print_1dimension_int(h_checkToVar,N*col);*/
					
					//test
					/*memset(h_E_tmp,0,sizeof(float)*((M+1)*(N+1)));
					cudaMemcpy(h_E_tmp,d_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
					print_1dimension_float(h_E_tmp,M*N);*/


					SetCheckMessage<<<grid,threads>>>(d_E,d_mes,d_varToCheck,row,N,M);	
					cudaThreadSynchronize();


					//test d_mes
					/*cudaMemcpy(h_mes,d_mes,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
					print_1dimension_float(h_mes,M*N);*/
					
					//test
					/*memset(h_E_tmp,0,sizeof(float)*((M+1)*(N+1)));
					cudaMemcpy(h_E_tmp,d_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
					print_1dimension_float(h_E_tmp,M*N);*/
					
					threads.x = BLOCK_DIM_64;
					threads.y = 1;
					grid.y = 1;
					grid.x = (N+BLOCK_DIM_64-1)/BLOCK_DIM_64;

					setZ<<<grid,threads>>>(d_E,d_z,d_r,M,N,N);

					/*float* h_z_temp;
					h_z_temp = (float*)malloc(sizeof(float)*(N+1));
					float* d_z_temp;
					cudaMalloc((void**)&d_z_temp,sizeof(float)*(N+1));
					cudaMemset(d_z_temp,0,sizeof(float)*(N+1));
					setZ_float<<<grid,threads>>>(d_E,d_z_temp,d_r,M,N,N);*/
					cudaThreadSynchronize();
					/*cudaMemcpy(h_z_temp,d_z_temp,sizeof(int)*(N+1),cudaMemcpyDeviceToHost);
					print_1dimension_float(h_z_temp,N);*/ 
									

					//test 
					/*cudaMemcpy(z,d_z,sizeof(int)*(N+1),cudaMemcpyDeviceToHost);
					print_1dimension_int(z,N);*/  //z收敛速度过快
					//cudaMemcpy(r,d_r,sizeof(float)*(N+1),cudaMemcpyDeviceToHost);
					//print_1dimension_float(r,N);

					//test
					/*memset(h_E_tmp,0,sizeof(float)*((M+1)*(N+1)));
					cudaMemcpy(h_E_tmp,d_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
					print_1dimension_float(h_E_tmp,M*N);*/

					cudaMemcpy(z,d_z,sizeof(int)*(N+1),cudaMemcpyDeviceToHost);
					int codeErrNum = 0;
					//此处还可以改为并行?
					for(int i = 1;i <= N;i ++){
						if(z[i] == 1){
							codeErrNum ++;
							tmpErr ++;
						}
					}
					if(codeErrNum == 0){
						flag = 1;
					}					

					if(flag == 0){
						//test
						/*memset(h_E_tmp,0,sizeof(float)*((M+1)*(N+1)));
						cudaMemcpy(h_E_tmp,d_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
						print_1dimension_float(h_E_tmp,M*N);*/


						//test d_mes
						/*cudaMemcpy(h_mes,d_mes,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
						print_1dimension_float(h_mes,M*N);*/

						//检测显示在传入d_E时就是0
						/*cudaMemcpy(h_checkToVar,d_checkToVar,sizeof(int)*(N*col+1),cudaMemcpyDeviceToHost);
						print_1dimension_int(h_checkToVar,N*col);*/

						
						setBitMes<<<grid,threads>>>(d_checkToVar,d_mes,d_r,d_E,col,N,N);
						cudaThreadSynchronize();


						//test r access
						/*cudaMemcpy(r,d_r,sizeof(float)*(N+1),cudaMemcpyDeviceToHost);
						print_1dimension_float(r,N);*/

						//test d_mes
						/*cudaMemcpy(h_mes,d_mes,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
						print_1dimension_float(h_mes,M*N);*/

						//test
						/*memset(h_E_tmp,0,sizeof(float)*((M+1)*(N+1)));
						cudaMemcpy(h_E_tmp,d_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
						print_1dimension_float(h_E_tmp,M*N);*/
					}
				}
				if(iter == 1000){
					error++;
				}
			}
			double fer = 50.0/frame;	
			printf("snr = %f,totalframes = %d,fer:%lf\n",snr,frame,fer);

			FILE* fp;
			if((fp = fopen("result.txt","a+")) == NULL){
				printf("create result.txt failed");
				return 0;
			}					
			fprintf(fp,"snr = %f,totalframes = %d,fer = %lf\n",snr,frame,fer);
			fclose(fp);
		}
		clock_t end = clock();
		float cost = (float)(end - begin)/CLOCKS_PER_SEC;

		FILE* fp;
		if((fp = fopen("result.txt","a+")) == NULL){
			printf("create result.txt failed");
			return 0;
		}			
		fprintf(fp,"cost = %d\n",cost);
		fclose(fp);

		cudaFree(d_varToCheck);
		cudaFree(d_checkToVar);
		cudaFree(d_mes);
		cudaFree(d_E);
		cudaFree(d_z);
		cudaFree(d_r);
		delete A,B,z,mes,E,r,L;		
	}	
	
    return 0;
}

void printx(float** x,int m,int n){
	FILE* fp;
	if((fp = fopen("x.txt","wt")) == NULL){
		printf("error");
		exit(0);
	}
	for(int i = 1;i <= m;i ++){
		for(int j = 1;j <= n;j ++){
			fprintf(fp,"%f,",x[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void print_1dimension_float(float* x,int m){
	FILE* fp;
	if((fp = fopen("x.txt","wt")) == NULL){
		exit(0);
	}
	for(int i =1;i <= m;i ++){
		fprintf(fp,"%f,",x[i]);
		if(i % N == 0){
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
}

void print_1dimension_int(int* x,int m){
	FILE* fp;
	if((fp = fopen("x.txt","wt")) == NULL){
		exit(0);
	}
	for(int i =1;i <= m;i ++){
		fprintf(fp,"%d,",x[i]);
		if(i % N == 0){
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
}

int readH(){
	FILE* fp;
	fp = fopen(FILE_NAME,"rt");
	if (fp == NULL)
	{
		printf("can't open this file");
		return 0;
	}
	fscanf(fp,"%d %d\n",&N,&M);
	fscanf(fp,"%d %d\n",&col,&row);
	
	A = (int **)malloc((N+1) * sizeof(int *));
	for (int i = 1;i <= N;i++)
	{
		A[i] = (int *)malloc((col+1) * sizeof(int));
		for (int j = 1;j <= col;j++)
			fscanf(fp,"%d ",&A[i][j]);
	}

	B = (int **)malloc((M+1) * sizeof(int *));
	for (int i = 1;i <= M;i++)
	{
		B[i] = (int *)malloc((row+1) * sizeof(int));
		for (int j = 1;j <= row;j++)
			fscanf(fp,"%d ",&B[i][j]);  //*(B+i)+j
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
	memset(z,0,sizeof(int)*(N+1));
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
	}
}