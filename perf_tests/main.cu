/***
Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
***/

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include "../src/include/xmp.h"
#include <omp.h>
#include <gmp.h>
using namespace std;

#define XMP_CHECK_ERROR(fun) \
{                             \
  xmpError_t error=fun;     \
  if(error!=xmpErrorSuccess){ \
    if(error==xmpErrorCuda)   \
      printf("CUDA Error %s, %s:%d\n",cudaGetErrorString(cudaGetLastError()),__FILE__,__LINE__); \
    else  \
      printf("XMP Error %s, %s:%d\n",xmpGetErrorString(error),__FILE__,__LINE__); \
    exit(EXIT_FAILURE); \
  } \
}

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

double GetCurrentTimeInSeconds()
{
  timespec current_time;
  clock_gettime( CLOCK_REALTIME, &current_time );
  return ( current_time.tv_sec + 1e-9 * current_time.tv_nsec );
}


uint32_t rand32() {
  uint32_t lo=rand() & 0xffff; 
  uint32_t hi=rand() & 0xffff; 

  return (hi<<16)|lo; 
}

double run_gpu_performance_test(const vector<int> &devices, uint32_t iterations, uint32_t bits, uint32_t N, uint32_t bN, uint32_t eN, uint32_t mN, xmpAlgorithm_t alg) {
  double ops;
  int numDev=devices.size();
  if(eN==0) eN=N;
  if(mN==0) mN=N;
  if(bN==0) bN=N;
#pragma omp parallel num_threads(numDev)
  {
    int d=omp_get_thread_num();
    cudaSetDevice(devices[d]);

    xmpHandle_t handle;
    xmpIntegers_t x_c, x_b, x_e, x_m;
    uint32_t *c,*b,*e,*m;
    uint32_t limbs=bits/8/sizeof(uint32_t);
    float time_s;

    double start,stop;
    srand(0);

    c=(uint32_t*)malloc(N*limbs*sizeof(uint32_t));
    b=(uint32_t*)malloc(bN*limbs*sizeof(uint32_t));
    e=(uint32_t*)malloc(eN*limbs*sizeof(uint32_t));
    m=(uint32_t*)malloc(mN*limbs*sizeof(uint32_t));

    //initialize base, exp, and mod
    for(int i=0;i<bN*limbs;i++) b[i]=rand32();
    for(int i=0;i<eN*limbs;i++) e[i]=rand32();
    for(int i=0;i<mN*limbs;i++) m[i]=rand32();
    //ensure modulus is odd
    for(int i=0;i<mN;i++) {
      m[i*limbs]|=1;
    }
    xmpExecutionPolicy_t policy;

    XMP_CHECK_ERROR(xmpHandleCreate(&handle));
    XMP_CHECK_ERROR(xmpExecutionPolicyCreate(handle,&policy));
    XMP_CHECK_ERROR(xmpExecutionPolicySetParameter(handle,policy,xmpAlgorithm,alg));
    XMP_CHECK_ERROR(xmpHandleSetExecutionPolicy(handle,policy));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle,&x_c,bits,N));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle,&x_b,bits,bN));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle,&x_e,bits,eN));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle,&x_m,bits,mN));

    XMP_CHECK_ERROR(xmpIntegersImport(handle,x_b,limbs,-1,sizeof(uint32_t),0,0,b,bN));
    XMP_CHECK_ERROR(xmpIntegersImport(handle,x_e,limbs,-1,sizeof(uint32_t),0,0,e,eN));
    XMP_CHECK_ERROR(xmpIntegersImport(handle,x_m,limbs,-1,sizeof(uint32_t),0,0,m,mN));

    //warmp up
    xmpError_t error=xmpIntegersPowm(handle,x_c,x_b,x_e,x_m,N);
    cudaDeviceSynchronize();
    if(error==xmpErrorSuccess) {
      #pragma omp barrier
      start=GetCurrentTimeInSeconds();

      for(int i=0;i<iterations;i++) {
        XMP_CHECK_ERROR(xmpIntegersPowm(handle,x_c,x_b,x_e,x_m,N));
      }
      cudaDeviceSynchronize();
      #pragma omp barrier
      stop=GetCurrentTimeInSeconds();
      time_s=stop-start;

      if(d==0) ops=(double)devices.size()*N*iterations/time_s;
    } else if (error==xmpErrorUnsupported) {
      if(d==0) ops = 0;
    } else {
      exit(EXIT_FAILURE);
    }

    XMP_CHECK_ERROR(xmpIntegersDestroy(handle,x_c));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle,x_b));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle,x_e));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle,x_m));
    XMP_CHECK_ERROR(xmpExecutionPolicyDestroy(handle,policy));

    XMP_CHECK_ERROR(xmpHandleDestroy(handle));

    free(c);
    free(b);
    free(m);
    free(e);
  }
  return ops;
}

double run_cpu_performance_test(uint32_t iterations, uint32_t bits, uint32_t N, uint32_t bN, uint32_t eN, uint32_t mN) {
  mpz_t *g_c, *g_b, *g_e, *g_m;
  uint32_t *c,*b,*e,*m;
  uint32_t limbs=bits/8/sizeof(uint32_t);
  double start,end,time_s;
  
  if(eN==0) eN=N;
  if(mN==0) mN=N;
  if(bN==0) bN=N;

  srand(0);

  c=(uint32_t*)malloc(N*limbs*sizeof(uint32_t));
  b=(uint32_t*)malloc(bN*limbs*sizeof(uint32_t));
  e=(uint32_t*)malloc(eN*limbs*sizeof(uint32_t));
  m=(uint32_t*)malloc(mN*limbs*sizeof(uint32_t));
  
  g_c=(mpz_t*)malloc(sizeof(mpz_t)*N);
  g_b=(mpz_t*)malloc(sizeof(mpz_t)*bN);
  g_e=(mpz_t*)malloc(sizeof(mpz_t)*eN);
  g_m=(mpz_t*)malloc(sizeof(mpz_t)*mN);

  for(int i=0;i<N;i++) mpz_init(g_c[i]);
  for(int i=0;i<bN;i++) mpz_init(g_b[i]);
  for(int i=0;i<eN;i++) mpz_init(g_e[i]);
  for(int i=0;i<mN;i++) mpz_init(g_m[i]);

  //initialize base, exp, and mod
  for(int i=0;i<bN*limbs;i++) b[i]=rand32();
  for(int i=0;i<eN*limbs;i++) e[i]=rand32();
  for(int i=0;i<mN*limbs;i++) m[i]=rand32();
  //ensure modulus is odd
  for(int i=0;i<mN;i++) {
    m[i*limbs]|=1;
  }
  
  #pragma omp parallel for
  for(int i=0;i<bN;i++) mpz_import(g_b[i],limbs,-1,sizeof(uint32_t),-1,0,&b[i*limbs]);
  #pragma omp parallel for
  for(int i=0;i<eN;i++) mpz_import(g_e[i],limbs,-1,sizeof(uint32_t),-1,0,&e[i*limbs]);
  #pragma omp parallel for
  for(int i=0;i<mN;i++) mpz_import(g_m[i],limbs,-1,sizeof(uint32_t),-1,0,&m[i*limbs]);

  start=GetCurrentTimeInSeconds();
  //perform operation on the host
  #pragma omp parallel for
  for(int j=0;j<N;j++) {
    for(int i=0;i<iterations;i++) {
      mpz_t *cc=&g_c[j], *bb, *ee, *mm;
      bb= &g_b[j % bN]; 
      ee= &g_e[j % eN]; 
      mm= &g_m[j % mN ]; 
      mpz_powm_sec(*cc,*bb,*ee,*mm);
    }
  }
  end=GetCurrentTimeInSeconds();
  time_s=end-start;

  double ops=(double)N*iterations/time_s;

  for(int i=0;i<N;i++) mpz_clear(g_c[i]);
  for(int i=0;i<bN;i++) mpz_clear(g_b[i]);
  for(int i=0;i<eN;i++) mpz_clear(g_e[i]);
  for(int i=0;i<mN;i++) mpz_clear(g_m[i]);

  free(g_c); 
  free(g_b); 
  free(g_e); 
  free(g_m); 
  
  free(c);
  free(b);
  free(m);
  free(e);

  return ops;
}

void printUsageAndExit(int returnCode) {
  
  printf("./perf [-h] [-c threads] [-i iters] [-d device_list] [-s size_list] [-b bit_list] [-l start,end,increment,op], [-m count] [-e count]\n");
  printf("    order of command line parameters does not matter\n");
  printf("    -h:  prints this screen\n");
  printf("    -c:  eables a CPU run and specifies how many threads to use\n");
  printf("    -i:  the number of iterations\n");
  printf("    -d:  , or + separated list of device ordinals\n");
  printf("         , creates a new device group with the following device ordinals\n");
  printf("         + joins the next device ordinal to the current device group\n");
  printf("             for example 0,1+2 would create two device groups, the first with 0 and the second with 1 & 2.\n");
  printf("             devices in the same group are ran concurrently and their performance is aggregated.\n");
  printf("    -s:  , separated list sizes\n");
  printf("    -b:  , separated list bit sizes (precision)\n");
  printf("    -l:  specifies sizes to be set via a loop.  start is first size, end is bounds on max size, increment is the factor we add or multiply by, op can be '+' for addition and '*' for multiplication\n");
  printf("            for example  -l 1000,2000,100,+ specifies sizes 1000 to 2000 in increments of 100  (i.e 1000,1100,1200,...,2000)\n");
  printf("            and          -l 1000,2000,1.1,* specifies sizes 1000 to 2000 with multiplication factors 1.1 (i.e 1000,1100,1210,...,1948)\n");
  printf("    -m   overwrite the size parameter for the modulus with count\n");
  printf("    -e   overwrite the size parameter for the exponent with count\n");


  exit(returnCode);
}

int main(int argc, char **argv) {
  vector<vector<int> > devices;
  vector<int> bits;
  vector<int> sizes;
  int numDev; 
  int iters=10;
  int start, end;
  float inc;
  char op;

  int mCount=0;
  int eCount=0;

  cudaGetDeviceCount(&numDev);
  if(numDev==0) {
    printf("Error no CUDA device found\n");
    return 1;
  }

  //parse sizes
  int cpu=0;
  int c, val, a;
  char* str, *str2, *save;
  xmpAlgorithm_t alg = xmpAlgorithmDefault;

  xmpAlgorithm_t algorithms[4] = {xmpAlgorithmDefault, xmpAlgorithmRegMP, xmpAlgorithmDigitMP, xmpAlgorithmDistributedMP};

  while((c=getopt(argc,argv,"i:b:s:d:l:c:hm:e:a:"))!=-1) {
    switch(c) {
      case 'i':
        iters=atoi(optarg);
        break;
      case 'm':
        mCount=atoi(optarg);
        break;
      case 'e':
        eCount=atoi(optarg);
        break;
      case 'd':
        //grab devices in current set of commas
        str=strtok_r(optarg,",",&save);      
        while(str != NULL) {
          vector<int> vec;
          //grab first device
          str2=strtok(str,"+");
          while(str2!=NULL) {
            val=atoi(str2);
            if(val>=numDev) {
              printf("Invalid device ordinal: %d\n",val);
              return 1;
            }
            vec.push_back(val);
            str2=strtok(NULL,"+");
          }
          str=strtok_r(NULL,",",&save);
          devices.push_back(vec);
        }
        break;
      case 'b':
        str=strtok(optarg,",");      
        while(str != NULL) {
          val=atoi(str);
          str=strtok(NULL,",");
          bits.push_back(val);
        }
        break;
      case 's':
        str=strtok(optarg,",");      
        while(str != NULL) {
          val=atoi(str);
          str=strtok(NULL,",");
          sizes.push_back(val);
        }
        break;
      case 'l':
        str=strtok(optarg,",");
        if(str==NULL) {
          printf("Invalid -l parameter\n");
          printUsageAndExit(1);
        }
        start=atoi(str);
        str=strtok(NULL,",");
        if(str==NULL) {
          printf("Invalid -l parameter\n");
          printUsageAndExit(1);
        }
        end=atoi(str);
        str=strtok(NULL,",");
        if(str==NULL) {
          printf("Invalid -l parameter\n");
          printUsageAndExit(1);
        }
        inc=atof(str);
        str=strtok(NULL,",");
        if(str==NULL) {
          printf("Invalid -l parameter\n");
          printUsageAndExit(1);
        }
        op=str[0];
        switch(op) {
          case '+':
            for(int i=start;i<end;i+=inc) {
              sizes.push_back(i);
            }
            break;
          case '*':
            for(int i=start;i<end;i*=inc) {
              sizes.push_back(i);
            }
            break;
          default:
            printf("Invalid -l parameter\n");
            printUsageAndExit(1);
        }
        break;
      case 'h':
        printUsageAndExit(0);
      case 'c':
        cpu=atoi(optarg);
        break;
      case 'a':
        a=atoi(optarg);
        if(a<0 || a>3) printUsageAndExit(1);
        alg=algorithms[a];
        break;
      default:
        printf("Error invalid command line switch '%c'\n",c);
        printUsageAndExit(1);
    }
  }

  if(cpu==0 && devices.size()==0) {
    vector<int> vec;
    vec.push_back(0);
    devices.push_back(vec);
  }

  if(bits.size()==0) {
    bits.push_back(128);
    bits.push_back(256);
    bits.push_back(512);
  }

  if(sizes.size()==0) {
    sizes.push_back(8192);
    sizes.push_back(12288);
    sizes.push_back(16384);
    sizes.push_back(24576);
    sizes.push_back(32768);
    sizes.push_back(49152);
    sizes.push_back(65536);
    sizes.push_back(98304);
    sizes.push_back(131072);
  }

  for(int d=0;d<devices.size();d++) {
    vector<int> devs=devices[d];
    
    for(int j=0;j<devs.size();j++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop,devs[j]);
      printf("Device: %d, %s\n", devs[j], prop.name);
    }
    printf("Iters=%d\n",iters);
    printf("N\\BITS");

    for(int b=0;b<bits.size();b++) {
      printf(",%d",bits[b]);
    }
    for(int s=0;s<sizes.size();s++) {
      printf("\n%d",sizes[s]);
      for(int b=0;b<bits.size();b++) {

        //call all, generate tables
        printf(",%.3e",run_gpu_performance_test(devs, iters, bits[b],  sizes[s], sizes[s], eCount, mCount, alg));
      }
    }
    printf("\n");
  }
  
  if(cpu>0) {
    printf("cpu=%d\n",cpu);
    omp_set_num_threads(cpu);
    printf("CPU, Threads=%d, Iters=%d\n", omp_get_max_threads(), iters);
    printf("N\\BITS");
    for(int b=0;b<bits.size();b++) {
      printf(",%d",bits[b]);
    }
    for(int s=0;s<sizes.size();s++) {
      printf("\n%d",sizes[s]);
      for(int b=0;b<bits.size();b++) {
        //call all, generate tables
        printf(",%.3e",run_cpu_performance_test(iters, bits[b],  sizes[s], sizes[s], eCount, mCount));
      }
    }
    printf("\n");
  }
}
