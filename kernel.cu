#include <stdio.h>
//#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#define FILE_NAME "155_64_3_5.txt"
#define BLOCK_DIM_32 32
#define BLOCK_DIM_64 64
#define BLOCK_DIM_X 6;//У��ڵ�Ķ�
#define BLOCK_DIM_Y 3;//�����ڵ�Ķ�
__constant__  int MaxCheckdegree = 5;//����ڵ�Ķ�
__constant__ int MaxVarDegree = 3;//�����ڵ�Ķ�
using namespace std;

int **A,**B,*z;
float **mes,**E,*r,*L;
int N,M,col,row;
float var;

//int readFile();
void init();
void printZ();
int write_alist();

void printE(){
	FILE* fp;
	if((fp = fopen("E.txt","wt")) == NULL){
		printf("can not create E.txt\n");
		exit(0) ;
	}
	for(int i = 1;i <= M;i ++){
		for(int j = 1;j <= N;j ++){
			if(E[i][j] != 0){
				fprintf(fp,"%.3f ,",E[i][j]);
			}
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
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

int readLXL(){
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

//����У����Ϣ
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

//Ϊz��ֵ��zΪ�õ���������.���·���ռ�
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

//����mes
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
	FILE* fp;
	if((fp = fopen("result.txt","wt")) == NULL){
		printf("create result.txt failed");
		return 0;
	}
	if(readLXL()){
		/*ofstream fos("result.txt",ios::out);
		if(!fos){
			cout << "create file failed" << endl;
			return 0;
		}*/		
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

		for(float snr = 2.0;snr <= 5;snr += 0.1){
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

				dim3 threads,grid;
				int flag = 0;
				int iter = 0;
				while((!flag) && (iter < 1000)){
					iter ++; 
					//��Ҫ��gpu�߼�.
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
					print_1dimension_int(z,N);*/  //z�����ٶȹ���
					//cudaMemcpy(r,d_r,sizeof(float)*(N+1),cudaMemcpyDeviceToHost);
					//print_1dimension_float(r,N);

					//test
					/*memset(h_E_tmp,0,sizeof(float)*((M+1)*(N+1)));
					cudaMemcpy(h_E_tmp,d_E,sizeof(float)*((M+1)*(N+1)),cudaMemcpyDeviceToHost);
					print_1dimension_float(h_E_tmp,M*N);*/

					cudaMemcpy(z,d_z,sizeof(int)*(N+1),cudaMemcpyDeviceToHost);
					int codeErrNum = 0;
					//�˴������Ը�Ϊ����?
					for(int i = 1;i <= N;i ++){
						if(z[i] == 1){
							codeErrNum ++;
							tmpErr ++;
						}
					}
					//printZ();
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

						//�����ʾ�ڴ���d_Eʱ����0
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
			float fer = 50.0/frame;
			printf("snr = %f,totalframes = %d,fer:%f\n",snr,frame,fer);
			fprintf(fp,"snr = %f,totalframes = %d,fer = %f\n",snr,frame,fer);
		}
		clock_t end = clock();
		float cost = (float)(end - begin)/CLOCKS_PER_SEC;
		fprintf(fp,"cost = %d\n",cost);
		cudaFree(d_varToCheck);
		cudaFree(d_checkToVar);
		cudaFree(d_mes);
		cudaFree(d_E);
		cudaFree(d_z);
		cudaFree(d_r);
		delete A,B,z,mes,E,r,L;		
	}	
	fclose(fp);
    return 0;
}


void printZ(){
	FILE* fp;
	if((fp = fopen("Z.txt","wt")) == NULL){
		printf("cannot create Z.txt");
		exit(0);
	}
	for(int i = 1;i <= N;i ++){
		fprintf(fp,"%f,",z[i]);
	}
	fclose(fp);
}



//���ļ��Լ�����ֵ
//int readFile(){
//	ifstream fip(FILE_NAME);
//	if(!fip){
//		cout << "Can not find txt file!" << endl;
//		return 0;
//	}
//	//�˴����Ǳ�׼��ʽ�ˣ�Ӧ�û�������ֱ��ɾ����
//	fip >> N;
//	fip >> M;
//	fip >> col;
//	fip >> row;
//	A = new int*[N+1];
//	for(int i = 0;i <= N;i ++){
//		A[i] = new int[col+1];
//	}
//	B = new int*[M+1];
//	for(int i = 0;i <= M;i ++){
//		B[i] = new int[row+1];
//	}
//	mes = new float*[M+1];
//	for(int i = 0;i <= M;i ++){
//		mes[i] = new float[N+1];
//	}
//	E = new float*[M+1];
//	for(int i = 0;i <= M;i ++){
//		E[i] = new float[N+1];
//	}
//	r = new float[N+1];
//	L = new float[N+1];
//	z = new int[N+1];
//	for(int i = 1;i < N+1;i ++){
//		for(int j = 1;j < col +1;j ++){
//			fip >> A[i][j];
//		}
//	}
//	for(int i = 1;i < M +1;i ++){
//		for(int j = 1;j < row +1;j ++){
//			fip >> B[i][j];
//		}
//	}
//	memset(z,0,sizeof(int)*(N+1));
//	return 1;
//}

//��ʼ��������
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


#define NUM_1 12
#define NUM_2 20
#define NUM_3 200
#define NUM_4 330
#define derect "first.txt"
int biggest_num_n,biggest_num_m;
int write_alist ()
{
	/* this assumes that A and B have the form of a rectangular
	matrix in the file; if lists have unequal lengths, then the
	entries should be present (eg zero values) but are ignored
	*/
	FILE *fp;
	fp = fopen(derect,"r");
	if(!fp)
	{
		printf("open file failed!");
		return 0;
	}
	char tmp[NUM_4];
	memset(tmp,0,sizeof(tmp));
	fgets(tmp,NUM_1,fp);
	int i=0,t = 10;
	M = N = 0;
	while(tmp[i] != ' ')
	{
		N = N * t + (tmp[i] - '0');
		++i;
	}
	++i;
	while((tmp[i] != ' ')&&(tmp[i] != '\n'))
	{
		M = M * t + (tmp[i] - '0');
		++i;
	}
	memset(tmp,0,sizeof(tmp));
	fgets(tmp,NUM_1,fp);
	i = 0;
	biggest_num_n = biggest_num_m = 0;
	while(tmp[i] != ' ')
	{
		biggest_num_n = biggest_num_n * t + (tmp[i] - '0');
		++i;
	}
	++i;
	while(tmp[i] != ' ')
	{
		biggest_num_m = biggest_num_m * t + (tmp[i] - '0');
		++i;
	}
	memset(tmp,0,sizeof(tmp));
	fgets(tmp,NUM_4,fp);
	fgets(tmp,NUM_4,fp);
	memset(tmp,0,sizeof(tmp));
	int j = 0,k,_tmp;
	int tmp_N = N,tmp_M = M;
	//matrix->B = (int **)malloc(sizeof(int) * )
	while(tmp_N--)//�洢�����е�B����
	{
		i = 0;
		k = 0;
		fgets(tmp,NUM_3,fp);
		while(tmp[i] != '\n')
		{
			_tmp = 0;
			while(tmp[i] != ' ')
			{
				_tmp = _tmp * t + (tmp[i] - '0');
				++i;
			}
			A[j][k] = _tmp-1;
			E[j][_tmp-1] = 1;
			++k;
			++i;
		}
		++j;
	}
	j=0;
	while(tmp_M--)//�洢�����е�A����
	{
		i = 0;
		k = 0;
		fgets(tmp,NUM_2,fp);
		while(tmp[i] != '\n' && tmp[i] != '\0')
		{
			_tmp = 0;
			while(tmp[i] != ' ')
			{
				_tmp = _tmp * t + (tmp[i] - '0');
				++i;
			}
			B[j][k++] = _tmp-1;
			E[_tmp-1][j] = 1;
			++i;
		}
		++j;
	}
	fclose(fp);
	return 1;
}

