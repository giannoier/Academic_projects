/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.0005

////////////////////////////////////////////////////////////////////////////////
// Row convolution kernel
////////////////////////////////////////////////////////////////////////////////



__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, int imageW, int imageH, int filterR){
  int k;
  float sum=0;
  int row=blockDim.y*blockIdx.y+threadIdx.y+filterR;
  int col=blockDim.x*blockIdx.x+threadIdx.x+filterR;
  int newImageW=imageW+filterR*2;
  
  for (k = -filterR; k <= filterR; k++) {
	
        int d = col+ k;


          sum += d_Src[row *newImageW + d] * d_Filter[filterR - k];


       
      }
      d_Dst[row *newImageW + col] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, int imageW, int imageH, int filterR){
  int k;
  float sum=0;
  int row=blockDim.y*blockIdx.y+threadIdx.y+filterR;
  int col=blockDim.x*blockIdx.x+threadIdx.x+filterR;
  int newImageW =imageW+filterR*2;
  for (k = -filterR; k <= filterR; k++) {
	
        int d = row+ k;
        
          sum += d_Src[col +newImageW* d] * d_Filter[filterR - k];

        
      }
  d_Dst[row * newImageW + col] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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
    
    
    float
    *h_Filter,
    *h_Input,
    *h_PaddingMatrix,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;
    
    struct timespec  tv1, tv2;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int imageW;
    int imageH;
    unsigned int i,j;
    

    printf("Enter filter radius : ");
    scanf("%d", &filter_radius);
    
    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

//    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  //  scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    if(h_Filter==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    if(h_Input==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_PaddingMatrix     = (float *)malloc((imageW+filter_radius*2 )*(2*filter_radius+ imageH) * sizeof(float));
    if(h_Input==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    if(h_Buffer==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    if(h_OutputCPU==NULL){
	printf("Allocation failed\n");
	return 0;
    }
    h_OutputGPU=(float *)malloc((imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(float));
    if(h_OutputGPU==NULL){
      printf("Allocation failed \n");
      cudaDeviceReset();
      return 0;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
// Desmeush mnhmhs sto device
////////////////////////////////////////////////////////////////////////////////
    cudaMalloc(&d_Filter,FILTER_LENGTH*sizeof(float));
    cudaMalloc(&d_Input,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(float));
    cudaMalloc(&d_Buffer,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(float));
    cudaMalloc(&d_OutputGPU,(imageW+2*filter_radius)*(imageH+2*filter_radius)*sizeof(float));
        
    if(d_Filter==NULL || d_Input==NULL || d_Buffer==NULL || d_OutputGPU==NULL){
      
      printf("Cuda Malloc Failed\n");
      return 0;
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 16);
    }
   
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("CPU time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
 
    dim3 dimGrid(imageW/8,imageH/8);
    dim3 dimBlock(8,8);
    
    for(i=0;i<(imageW+2*filter_radius)*(imageW+2*filter_radius);i++){
      h_PaddingMatrix[i]=0;
    }
    for(i=0;i<imageW;i++){
      for(j=0;j<imageW;j++){
	h_PaddingMatrix[(i+filter_radius)*(2*filter_radius+imageW)+j+filter_radius]=h_Input[i*imageW+j];
      }
    }
      
    printf("GPU computation... \n");
    
    cudaMemcpy(d_Filter,h_Filter,FILTER_LENGTH*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input,h_PaddingMatrix,(imageH+2*filter_radius)*(imageW+2*filter_radius)*sizeof(float),cudaMemcpyHostToDevice);
    cudaEventRecord(start,0);
    convolutionRowGPU <<< dimGrid,dimBlock >>>(d_Buffer,d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaThreadSynchronize();
    cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess){
      printf("Cuda Error:%s\n",cudaGetErrorString(error));
      cudaDeviceReset();
      return 0;
      
    }
    convolutionColumnGPU <<< dimGrid,dimBlock >>>(d_OutputGPU,d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaThreadSynchronize();
    error=cudaGetLastError();
    if(error!=cudaSuccess){
      printf("Cuda Error:%s\n",cudaGetErrorString(error));
      cudaDeviceReset();
      return 0;
      
    }
    
    
    cudaEventRecord(stop,0);
   
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed,start,stop);
    printf("GPU time %f seconds.\n",elapsed/1000);
    

    cudaMemcpy(h_OutputGPU,d_OutputGPU,(imageH+2*filter_radius)*(imageW+2*filter_radius)*sizeof(float),cudaMemcpyDeviceToHost);
   
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
   
    


    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    cudaFree(d_Input);
    cudaFree(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
     cudaDeviceReset();


    return 0;
}
