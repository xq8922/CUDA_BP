#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#define FILE_NAME "155_64_3_5.txt"
#define BLOCK_DIM_32 32
#define BLOCK_DIM_64 64
#define BLOCK_DIM_X 6;//校验节点的度
#define BLOCK_DIM_Y 3;//变量节点的度
__constant__  int MaxCheckdegree = 6;//检验节点的度
__constant__ int MaxVarDegree = 3;//变量节点的度
__constant__ int MaxRow = 155;
__constant__ int MaxCol = 64;
using namespace std;

int **A,**B,*z;
float **mes,**E,*r,*L;
int N,M,col,row;
float var;

int readFile();
void init();
void printE();

//设置校验信息
__global__ void SetCheckMessage(float *E,float *Mes,int *B,int width)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//计算当前行号,如何确定innerIdx正确性

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
__global__ void setZ(float *E,float *z,float *r,int* B,int M,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(xIndex < width){
		float t = 0.0;
		//计算当前列号
		//int column = xIndex*MaxCheckdegree+B[i];
		for(int i = 1;i <= M;i ++){
			t += E[i*MaxCheckdegree + xIndex];
		}
		z[xIndex] = (t + r[xIndex] <= 0)?1:0;
	}
}

//计算mes
__global__ void setBitMes(float *E,int *A,float* Mes,float* r,int col,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	float m = 0.0;
	if(xIndex < width){
		//计算列号		
		for(int i = 1;i <= col;i ++){
			m += E[A[xIndex * col+i]+xIndex];
		}
	}
	__syncthreads();
	if(xIndex < width){
		for(int i = 1;i <= col;i ++){
			m -= E[A[xIndex * col+i]+xIndex];
			Mes[A[xIndex * col+i]+xIndex] = m + r[i];
		}	
	}
}


int main()
{	
	srand((unsigned)time(NULL));
	clock_t begin = clock();
	if(readFile()){
		/*ofstream fos("result.txt",ios::out);
		if(!fos){
			cout << "create file failed" << endl;
			return 0;
		}*/
		FILE* fp;
		if((fp = fopen("result.txt","wt")) == NULL){
			printf("create result.txt failed");
			return 0;
		}
		int* h_varToCheck;
		h_varToCheck = (int*)malloc(sizeof(int)*(M*row+1));
		int* d_varToCheck;
		cudaMalloc((void**)&d_varToCheck,sizeof(int)*(M*row+1));

		int* h_checkToVar;
		h_checkToVar = (int*)malloc(sizeof(int)*(M*col+1));
		int* d_checkToVar;
		cudaMalloc((void**)&d_checkToVar,sizeof(int)*(M*col+1));

		float* h_mes;
		h_mes = (float*)malloc(sizeof(float)*(M+1)*(N+1));
		float* d_mes;
		cudaMalloc((void**)&d_mes,sizeof(float)*(M+1)*(N+1));

		float* h_E;
		h_E = (float*)malloc(sizeof(float)*(M+1)*(N+1));
		float* d_E;
		cudaMalloc((void**)&d_E,sizeof(float)*(M+1)*(N+1));

		float* d_z;
		cudaMalloc((void**)&d_z,sizeof(float)*(N+1));

		float* d_r;
		cudaMalloc((void**)&d_r,sizeof(float)*(N+1));

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
				memset(h_varToCheck,0,sizeof(float)*(M*row+1));
				memset(h_checkToVar,0,sizeof(float)*(N*col+1));
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
				//set E
				for(int i = 1;i <= M;i ++){
					for(int j = 1;j <= N;j ++){
						int index = (i-1)*N + j;
						h_E[index] = E[i][j];
					}
				}

				//set Mes
				for(int i = 1;i <= M;i ++){
					for(int j = 1;j <= N;j ++){
						int index = (i-1)*N+j;
						h_mes[index] = mes[i][j];
					}
				}

				cudaMemcpy(d_E,h_E,sizeof(float)*(M+1)*(N+1),cudaMemcpyHostToDevice);
				cudaMemcpy(d_mes,h_mes,sizeof(float)*(M+1)*(N+1),cudaMemcpyHostToDevice);
				cudaMemcpy(d_varToCheck,h_varToCheck,sizeof(float)*(M*row+1),cudaMemcpyHostToDevice);
				cudaMemcpy(d_r,r,sizeof(float)*(N+1),cudaMemcpyHostToDevice);

				dim3 threads,grid;
				threads.x = BLOCK_DIM_32;
				threads.y = 1;
				grid.x = (M+BLOCK_DIM_32-1)/BLOCK_DIM_32;
				grid.y = 1;

				int flag = 0;
				int iter = 0;				
				while((!flag) && (iter < 1000)){
					t ++;
					iter ++;
					//主要的gpu逻辑.
					cudaMemcpy(d_mes,h_mes,sizeof(float)*(M+1)*(N+1),cudaMemcpyHostToDevice);

					SetCheckMessage<<<grid,threads>>>(d_E,d_mes,d_varToCheck,MaxCheckdegree);					
					cudaThreadSynchronize();
					setZ<<<grid,threads>>>(d_E,d_z,d_r,d_checkToVar,M,MaxVarDegree);
					cudaThreadSynchronize();
					cudaMemcpy(h_mes,d_mes,sizeof(float)*(M*N+1),cudaMemcpyDeviceToHost);
					cudaMemcpy(h_E,d_E,sizeof(float)*(M*N+1),cudaMemcpyDeviceToHost);
					//set E
					for(int i = 1;i <= M*N;i ++){
						E[i/(N+1)+1][i-N*i/(N+1)] = h_E[i];
					}
					//printE();
					//set z
					cudaMemcpy(z,d_z,sizeof(float)*(N+1),cudaMemcpyDeviceToHost);
					int codeErrNum = 0;
					//此处还可以改为并行
					for(int i = 1;i <= N;i ++){
						if(z[i]){
							codeErrNum ++;
							tmpErr ++;
						}
					}
					//printZ();
					if(codeErrNum == 0){
						flag = 1;
					}
					if(flag == 0){
						//gpu
						threads.x = BLOCK_DIM_64;
						threads.y = 1;
						grid.y = 1;
						grid.x = (N+BLOCK_DIM_64-1)/BLOCK_DIM_64;
						cudaMemcpy(d_checkToVar,h_checkToVar,sizeof(float)*N*col,cudaMemcpyHostToDevice);
						cudaMemcpy(d_E,h_E,sizeof(float)*N*M,cudaMemcpyHostToDevice);
						cudaMemcpy(d_mes,h_mes,sizeof(float)*M*N,cudaMemcpyHostToDevice);
						cudaMemcpy(d_r,r,sizeof(float)*N,cudaMemcpyHostToDevice);
						setBitMes<<<grid,threads>>>(d_E,d_checkToVar,d_mes,d_r,col,MaxCheckdegree);
						cudaThreadSynchronize();
						cudaMemcpy(h_mes,d_mes,sizeof(float)*(M+1)*(N+1),cudaMemcpyDeviceToHost);
					}
				}
				if(iter == 1000){
					error++;
				}
			}
			printf("snr = %f,error frame:%f\n",snr,(50/frame));
			/*fos << snr << "\t" << ((float)50/frame) << endl;
			fos << snr << "\t" << ((float)tmpErr/50/N) << endl;*/
			fprintf(fp,"snr = %f,error frame = %f\n",snr,(50.0/frame));
		}
		clock_t end = clock();
		float cost = (float)(end - begin)/CLOCKS_PER_SEC;
		cudaFree(d_varToCheck);
		cudaFree(d_checkToVar);
		cudaFree(d_mes);
		cudaFree(d_E);
		cudaFree(d_z);
		cudaFree(d_r);
		delete A,B,z,mes,E,r,L;
		//fos.close();
		fclose(fp);
	}	

    return 0;
}

void printE(){
	FILE* fp;
	if((fp = fopen("E.txt","wt+")) == NULL){
		printf("can not create E.txt\n");
		return ;
	}
	for(int i = 1;i <= M;i ++){
		for(int j = 1;j <= N;j ++){
			fprintf(fp,"%f ,",E[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
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
		//将mes分解为行
	}
}