#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "hist-equ.h"

__global__ void histogram( int * hist_out, unsigned char * img_in, int img_w,int img_h,  int nbr_bin){
  
    int tx=threadIdx.x; 
    int ty=threadIdx.y;
    int bx=blockIdx.x; 
    int by=blockIdx.y;
    
    __shared__ int smem[256];
    smem[threadIdx.x]=0;
    __syncthreads();
    
    unsigned int col= tx + blockDim.x * bx;
    unsigned int row= ty + blockDim.y * by;    
    
  
    int grid_width = gridDim.x * blockDim.x;
    int id = row * grid_width + col;                
   
    if(row<img_w && col<img_h)
	 atomicAdd( &(smem[img_in[id]]) ,1);
    
    __syncthreads();
    
    atomicAdd(&(hist_out[threadIdx.x]),smem[threadIdx.x]);
    
    
}


__global__ void histogram_equalization( int * lut, unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
  
    int tx=threadIdx.x; 
    int ty=threadIdx.y;
    int bx=blockIdx.x; 
    int by=blockIdx.y;
   
    
    unsigned int col= tx + blockDim.x * bx;
    
    unsigned int row= ty + blockDim.y * by;    
  
    int grid_width = gridDim.x * blockDim.x;
    int id = row * grid_width + col;  
    
    // Get the result image 
    if(id<img_size){
      
	if(lut[img_in[id]] > 255){
		img_out[id] = 255;
        }
	else{
		img_out[id] = (unsigned char)lut[img_in[id]];
        }        
    }
    
}

int main(int argc, char *argv[]){
    PGM_IMG img_in;
   
  
      
    PGM_IMG result, d_img_in, d_result;
    int hist[256];
    int * d_hist, *d_lut;
    
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed;
	
    dim3 block_size;
    dim3 grid_size;
  
    
	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_in = read_pgm(argv[1]);
   
  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
   
    cudaEventRecord(start,0);
    
    cudaMalloc((void**)&d_hist , 256* sizeof(int));
    cudaMalloc((void**)&d_img_in.img, result.w * result.h * sizeof(unsigned char));
    cudaMalloc((void **)&d_result.img, result.w * result.h * sizeof(unsigned char));
    
    if(d_hist==NULL || d_img_in.img==NULL || d_result.img==NULL) {
      
      printf("Cuda Malloc Failed\n");
      return 0;
    }
    
    cudaMemcpy(d_img_in.img, img_in.img, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result.img, result.img, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
   
    block_size.x = 256;
    block_size.y = 1;
  
    grid_size.x=(img_in.w/256)+1;
    grid_size.y=(img_in.h)+1;
  
    histogram<<<grid_size, block_size>>>(d_hist, d_img_in.img, img_in.w, img_in.h,  256);
    
    cudaMemcpy(hist,d_hist,256* sizeof(int), cudaMemcpyDeviceToHost);
    
    
    cudaMalloc((void**)&d_lut,256*sizeof(int));
    int *lut = (int *)malloc(sizeof(int)*256);
    
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = result.w*result.h - min;
    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }   
    }
    
    cudaMemcpy(d_lut, lut, 256*sizeof(int), cudaMemcpyHostToDevice);
    
    histogram_equalization<<<grid_size, block_size>>>(d_lut, d_result.img,d_img_in.img,d_hist,img_in.h * img_in.w, 256);

    cudaMemcpy(result.img, d_result.img,  result.w * result.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop,0);//execution time stops
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed,start,stop);
    printf("GPU time : %g ms",elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);	

    write_pgm(result, argv[2]);
    free_pgm(result);
    free_pgm(img_in);

    return 0;
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
 
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
   
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}
