/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
//#define accuracy  	0.05	
#define accuracy  	0.0005 


////////////////////////////////////////////////////////////////////////////////
// Row convolution kernel
////////////////////////////////////////////////////////////////////////////////


__global__ void ConvolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, int imageW, int imageH, int filterR){
  int k;
  double sum=0;
  int row=blockDim.y*blockIdx.y+threadIdx.y;
  int col=blockDim.x*blockIdx.x+threadIdx.x;
  
  for (k = -filterR; k <= filterR; k++) {
	
        int d = col+ k;

        if (d >= 0 && d < imageW) {
          sum += d_Src[row * imageW + d] * d_Filter[filterR - k];
        }     

        d_Dst[row * imageW + col] = sum;
      }
     
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution kernel
////////////////////////////////////////////////////////////////////////////////

     
__global__ void ConvolutionColGPU(double *d_Dst, double *d_Src, double *d_Filter, int imageW, int imageH, int filterR){
  int k;
  double sum=0;
  int row=blockDim.y*blockIdx.y+threadIdx.y;
  int col=blockDim.x*blockIdx.x+threadIdx.x;
  
  for (k = -filterR; k <= filterR; k++) {
	
        int d = row+ k;
        
        if (d >= 0 && d < imageW) {
          sum += d_Src[col +imageW* d] * d_Filter[filterR - k];
        }     

        d_Dst[row * imageW + col] = sum;
      }
}

 
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;
    
    double
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;
    
    
   

    int imageW;
    int imageH;
    unsigned int i;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    
    struct timespec  tv1, tv2;
    
    printf("Enter filter radius : ");
    scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;
    
    
    dim3 grid_dim(imageW/8,imageH/8);
    dim3 block_dim(8,8);
    
    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
     if(h_Filter==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
     if(h_Input==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
     if(h_Buffer==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
     if(h_OutputCPU==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_OutputGPU	= (double *)malloc(imageW * imageH * sizeof(double));
     if(h_OutputGPU==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    
    
  

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++)
    {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 16);
    }


    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    
    printf ("CPU TIME = %g seconds\n",(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +(double) (tv2.tv_sec - tv1.tv_sec));
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
	//orizw to block ws imageW * imageH
    
    
    //desmeusi mnimis stin GPU
    cudaMalloc((void**)&d_Filter,FILTER_LENGTH * sizeof(double));
    cudaMalloc((void**)&d_Input,imageW * imageH * sizeof(double));
    cudaMalloc((void**)&d_Buffer,imageW * imageH * sizeof(double));
    cudaMalloc((void**)&d_OutputGPU,imageW * imageH * sizeof(double));
    
    
    if(d_Filter==NULL || d_Input==NULL || d_Buffer==NULL || d_OutputGPU==NULL){
      
      printf("Cuda Malloc Failed\n");
      return 0;
    }
    
    cudaEventRecord(start,0);
    
    cudaMemcpy(d_Filter,h_Filter,FILTER_LENGTH * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input,h_Input,imageW * imageH * sizeof(double),cudaMemcpyHostToDevice);
  
    
    //kernel launch
    ConvolutionRowGPU<<<grid_dim,block_dim>>>(d_Buffer,d_Input, d_Filter, imageW, imageH, filter_radius); 
    
    cudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
   
    if(error != cudaSuccess){
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      
      return 1;
    }
    //kernel launch
    ConvolutionColGPU<<<grid_dim,block_dim>>>(d_OutputGPU,d_Buffer, d_Filter, imageW, imageH, filter_radius);
    
    
    //metafora dedomenwn apo tin GPU
    cudaMemcpy(h_OutputGPU,d_OutputGPU,imageW * imageH * sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    //elegxos gia sfalmata
    cudaThreadSynchronize();
    error = cudaGetLastError();
    
    if(error != cudaSuccess){
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      
      return 1;
    }
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU TIME = %f seconds \n",milliseconds/1000);
   
     //elegxos apotelesmatos
    i=0;
    while (i<imageW*imageH){
      
      if(ABS(h_OutputGPU[i]-h_OutputCPU[i])>accuracy){
	    printf("Accuracy Error, at element %d\n GPU result - CPU result =  %f\n Aborting...\n",i,h_OutputGPU[i]-h_OutputCPU[i]);
	    break;    
	}
	i++;
    }
   
    
    

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    free(h_OutputGPU);
    
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputGPU);
    cudaFree(d_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();

    return 0;
}
