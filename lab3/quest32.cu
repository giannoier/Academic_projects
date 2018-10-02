/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

//unsigned int radius;

#define radius 16
#define FILTER_LENGTH 	(2 * radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	6

#define tileRH 1
#define tileRW 512

#define tileCH 16
#define tileCW 16


typedef float numid;

__constant__ numid d_Filter[FILTER_LENGTH];

__global__ void tiledConvRowGPU(numid *d_Dst, numid *d_Src, int imageW, int imageH){
	int k;
	numid sum = 0;
	
	int tx=threadIdx.x; 
	int ty=threadIdx.y;
	int bx=blockIdx.x; 
	int by=blockIdx.y;
	  
	
	int row = blockDim.y*by + ty ;
	int col = blockDim.x*bx + tx;
	
	int newImageW = imageW + radius * 2;
	__shared__ numid ShMemory[tileRH] [tileRW + 2 * radius];
	
	if(tx-radius<0){											//Near Left Bounds
		ShMemory[ty][tx] = d_Src[(row+radius) * newImageW + col];
	}
	
	ShMemory[ty][tx+radius] = d_Src[(row+radius) * newImageW + col + radius];				//Center
	
	if(tx >= (tileRW - radius)){							
		ShMemory[ty] [tx + 2 * radius] = d_Src[(row+radius) * newImageW + col + 2 * radius];		//Near Right Bounds
	}
	__syncthreads();
	
	for (k = -radius; k <= radius; k++) {
		
		sum += ShMemory[ty][tx+k+radius] * d_Filter[radius - k];
		
	}
	
	d_Dst[(row+radius) * newImageW + col+radius] = sum;
	
}
__global__ void tiledConvColGPU(numid *d_Dst, numid *d_Src, int imageW, int imageH){
	int k;
	numid sum = 0;
	
	int tx=threadIdx.x; 
	int ty=threadIdx.y;
	int bx=blockIdx.x; 
	int by=blockIdx.y;
	
	
	int row = blockDim.y*by + ty ;
	
	int col = blockDim.x*bx + tx;
	int newImageW = imageW + radius * 2;

	__shared__ numid ShMemory[tileCH + 2 * radius][ tileCW];
	
	if(ty-radius<0){											//Upper Bounds
		ShMemory[ty]  [tx] = d_Src[row * newImageW + col + radius];
	}
	
	ShMemory[ty + radius][ tx ] = d_Src[(row + radius) * newImageW + col + radius ];			//Center
	
	
	ShMemory[ty + 2 * radius ][ tx ] = d_Src[(row + 2* radius) * newImageW + col + radius ];		//Lower Bounds
	
	__syncthreads();
	
	for (k = -radius; k <= radius; k++) {
		
		sum += ShMemory[(ty + k + radius)][tx] * d_Filter[radius - k];
		
	}
	
	d_Dst[ (row + radius) * newImageW + col + radius] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(numid *h_Dst, numid *h_Src, numid *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      numid sum = 0;

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
void convolutionColumnCPU(numid *h_Dst, numid *h_Src, numid *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      numid sum = 0;

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
    cudaSetDevice(0);
    numid
    *h_Filter,
    *h_Input,
    *h_PadInput,
    *h_Buffer,
    *h_OutputCPU,
    
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *result;
    struct timespec  tv1, tv2;
    cudaEvent_t start;
    cudaEvent_t stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int imageW;
    int imageH;
    unsigned int i,j;
    
    if(argc<2){
      printf("Please specify the image size as execution arguments\n");
      return 0;
      
    }
	
    imageW=atoi(argv[1]);
    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

//    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  //  scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (numid *)malloc(FILTER_LENGTH * sizeof(numid));
   
    
    h_Input     = (numid *)malloc(imageW * imageH * sizeof(numid)); 
    h_PadInput  = (numid *)malloc((imageW+radius*2 )*(2*radius+ imageH) * sizeof(numid))  ;
    h_Buffer    = (numid *)malloc(imageW * imageH * sizeof(numid));  
    h_OutputCPU = (numid *)malloc(imageW * imageH * sizeof(numid));   
    result	= (numid *)malloc((imageW+2*radius) * (imageH+2*radius) * sizeof(numid));

    
    
    cudaMalloc(&d_Input,(imageW+2*radius)*(imageH+2*radius)*sizeof(numid));
    cudaMalloc(&d_Buffer,(imageW+2*radius)*(imageH+2*radius)*sizeof(numid));
    cudaMemset(d_Buffer,0,(imageW+2*radius)*(imageH+2*radius)*sizeof(numid));
    cudaMalloc(&d_OutputGPU,(imageW+2*radius)*(imageH+2*radius)*sizeof(numid));
    
    if(d_Filter==NULL || d_Input==NULL || d_Buffer==NULL || d_OutputGPU==NULL){
      
      printf("Cuda Malloc Failed\n");
      return 0;
    }
    
    
    cudaMemset(d_OutputGPU,0,(imageW+2*radius)*(imageH+2*radius)*sizeof(numid));

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (numid)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (numid)rand() / ((numid)RAND_MAX / 255) + (numid)rand() / (numid)RAND_MAX;
    }
   
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("CPU time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
  
    dim3 dimGridR(imageW/tileRW,imageH/tileRH);
    dim3 dimBlockR(tileRW,tileRH);
    dim3 dimGridC(imageW/tileCW,imageH/tileCH);
    dim3 dimBlockC(tileCW,tileCH);
	
    for(i=0;i<(imageW+2*radius)*(imageW+2*radius);i++){
      h_PadInput[i]=0;
    }
    for(i=0;i<imageW;i++){
      for(j=0;j<imageW;j++){
	h_PadInput[(i+radius)*(2*radius+imageW)+j+radius]=h_Input[i*imageW+j];
      }
    }
    

    printf("GPU computation... \n");
    cudaMemcpyToSymbol(d_Filter, h_Filter,FILTER_LENGTH*sizeof(numid));
    cudaMemcpy(d_Input,h_PadInput,(imageH+2*radius)*(imageW+2*radius)*sizeof(numid),cudaMemcpyHostToDevice);
	cudaEventRecord(start,0);
	tiledConvRowGPU <<< dimGridR, dimBlockR >>>(d_Buffer, d_Input, imageW, imageH);
    cudaThreadSynchronize();
    cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess){
      printf("Cuda Error:%s\n",cudaGetErrorString(error));
      cudaDeviceReset();
      return 0;
      
	}
	
	tiledConvColGPU <<< dimGridC, dimBlockC >>>(d_OutputGPU, d_Buffer , imageW, imageH);
    cudaThreadSynchronize();
    error=cudaGetLastError();
    if(error!=cudaSuccess){
      printf("Cuda Error:%s\n",cudaGetErrorString(error));
      cudaDeviceReset();
      return 0;
      
    }
	cudaEventRecord(stop,0);
    cudaMemcpy(result,d_OutputGPU,(imageH+2*radius)*(imageW+2*radius)*sizeof(numid),cudaMemcpyDeviceToHost);
    
   
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed,start,stop);
	
    printf("GPU time :%f ms.\n",elapsed);
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
   
    
   for(i=0;i<imageW;i++){
     for(j=0;j<imageH;j++){
      numid diff= h_OutputCPU[i*imageW+j]-result[(i+radius)*(imageW+2*radius)+j+radius];
	if(ABS(diff)>accuracy){
	    printf("sfalma akriveias %f",ABS(diff));
	}
    }
           
  }
  
  
 // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    cudaFree(d_Input);
    cudaFree(d_Filter);


     cudaDeviceReset();


    return 0;
}
