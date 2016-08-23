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

#include <xmp.h>
#include <xmp_internal.h>
#include <operators.h>
#include <powm_operators.h>
#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;

//#define CHECK

#define xmpCheckError(fun) \
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

int WAVES = 1;

uint32_t arch;
uint32_t smCount;

uint32_t rand32() {
  uint32_t lo=rand() & 0xffff; 
  uint32_t hi=rand() & 0xffff; 

  return (hi<<16)|lo; 
}


//sort by algorithm and then by instances (high to low)
bool operator<(Latency a, Latency b) {
  if(a.alg_index<b.alg_index) 
    return true;
  else if(b.alg_index<a.alg_index)
    return false;
  else
    return a.instances_per_sm>b.instances_per_sm;
}

void compute_and_add_latencies(xmpHandle_t handle, int alg, vector<Latency>& latencies, xmpIntegers_t out, xmpIntegers_t a, xmpIntegers_t exp, xmpIntegers_t mod ) {

  xmpPowmAlgorithm algorithm=xmpPowmAlgorithms[alg];
  uint32_t precision=out->precision;
  uint32_t count=out->count;
  uint32_t nlimbs=out->nlimbs;
  uint32_t* gold = (uint32_t*)malloc(sizeof(uint32_t)*count*nlimbs);
  uint32_t* test = (uint32_t*)malloc(sizeof(uint32_t)*count*nlimbs);
  uint32_t* zero = (uint32_t*)malloc(sizeof(uint32_t)*count*nlimbs); 

  assert(gold!=0);
  assert(test!=0);
  assert(zero!=0);

  int ITERS;
  if(precision>1024)
    ITERS=1;
  else 
    ITERS=5;

  for(int i=0;i<count*nlimbs;i++)
    zero[i]=0;

#ifdef CHECK
  //compute gold standard
  xmpCheckError(xmpIntegersPowm(handle,out,a,exp,mod,count));
  xmpCheckError(xmpIntegersExport(handle,gold,&nlimbs,-1,sizeof(uint32_t),-1,0,out,count));
#endif

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //compute max blocks per sm
  uint32_t max_blocks_sm, instances_per_block;
  algorithm.pfunc(handle,out,a,exp,mod,0,count,out->slimbs,&instances_per_block,&max_blocks_sm);

  for(int blocks_sm=1;blocks_sm<=max_blocks_sm*WAVES;blocks_sm++) {
    //compute instances
    uint32_t count = smCount*blocks_sm*instances_per_block;

    //Temporary work around for indexing bug
    if(count> 0x7FFFFF*8/precision) {
      continue;
    }

    xmpCheckError(xmpIntegersImport(handle,out,nlimbs,-1,sizeof(xmpLimb_t),-1,0,zero,count));

    //warm up
    xmpCheckError(algorithm.pfunc(handle,out,a,exp,mod,0,count,out->slimbs,NULL,NULL));
    //read back gold standard
    xmpCheckError(xmpIntegersExport(handle,test,&nlimbs,-1,sizeof(uint32_t),-1,0,out,count)); //may change format type
  
    cudaEventRecord(start);
    for(int i=0;i<ITERS;i++) {
      xmpCheckError(algorithm.pfunc(handle,out,a,exp,mod,0,count,out->slimbs,NULL,NULL));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaCheckError();

#ifdef CHECK
    //validate results
    for(int i=0;i<count*nlimbs;i++) {
      if(gold[i]!=test[i]) {
        printf("             ERROR i: %d (%d, %d), alg: %d, precision: %d, count: %d, nlimbs: %d, gold: %08x, test: %08\n", i, i/nlimbs,i%nlimbs, alg, precision, count, nlimbs, gold[i], test[i]);
        goto next;
      }
    }
    //validate results
#endif

    float time_ms, time_s;
    cudaEventElapsedTime(&time_ms,start,stop);
    time_s = time_ms / 1e3;
    latencies.push_back(Latency(alg,time_s/ITERS,count/smCount));
#ifdef CHECK
next:
#endif
    cudaCheckError();
  }

  free(zero);
  free(test);
  free(gold);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


int main() {

  xmpHandle_t handle;

  xmpCheckError(xmpHandleCreate(&handle));
  xmpExecutionPolicy_t policy;
  xmpCheckError(xmpExecutionPolicyCreate(handle, &policy));
  //xmpCheckError(xmpExecutionPolicySetParameter(handle,policy,xmpAlgorithm,xmpAlgorithmDistributedMP));
  //xmpCheckError(xmpExecutionPolicySetParameter(handle,policy,xmpAlgorithm,xmpAlgorithmDigitMP));
  xmpCheckError(xmpHandleSetExecutionPolicy(handle,policy));
  //get arch, get #sms
  arch = handle->arch;
  smCount = handle->smCount;
  
  char sarch[100];

  if(arch<50) 
    sprintf(sarch,"kepler");
  else
    sprintf(sarch,"maxwell");


  char filename[100];
  sprintf(filename,"tune_%s.h",sarch);

  FILE *file = fopen(filename,"w");
  
  vector<uint32_t> counts(xmpPowmPrecisionsCount);

  for(int p=0;p<xmpPowmPrecisionsCount;p++) {
    uint32_t P=xmpPowmPrecisions[p];
    float max_occupancy=1;
    if(P>4096)
      max_occupancy=.5;
    //max instances in a single wave at x% occupancy
    uint32_t max_instances = smCount * 2048 * WAVES * max_occupancy;
    uint32_t a_instances= max_instances;
    uint32_t exp_instances = 1;
    uint32_t mod_instances = 1;
    uint32_t out_instances = max_instances;

    vector<Latency> latencies;
    xmpIntegers_t a, exp, mod, out;
    uint32_t nlimbs = P/(sizeof(uint32_t)*8);

    uint32_t *limbs = (uint32_t*)malloc(nlimbs*sizeof(uint32_t)*a_instances);
    assert(limbs!=0);

    xmpCheckError(xmpIntegersCreate(handle, &a, P, a_instances));
    xmpCheckError(xmpIntegersCreate(handle, &exp, P, exp_instances));
    xmpCheckError(xmpIntegersCreate(handle, &mod, P, mod_instances));
    xmpCheckError(xmpIntegersCreate(handle, &out, P, out_instances));

    //generate random data
    for(int i=0;i<nlimbs*a_instances;i++) limbs[i]=rand32();
    xmpCheckError(xmpIntegersImport(handle,a,nlimbs,-1,sizeof(xmpLimb_t),-1,0,limbs,a_instances));
    
    for(int i=0;i<nlimbs*exp_instances;i++) limbs[i]=rand32();
    xmpCheckError(xmpIntegersImport(handle,exp,nlimbs,-1,sizeof(xmpLimb_t),-1,0,limbs,exp_instances));
    
    for(int i=0;i<nlimbs*mod_instances;i++) limbs[i]=rand32();
    
    for(int i=0;i<mod_instances;i++) limbs[i*nlimbs]|=0x1;  //ensure mod is odd

    xmpCheckError(xmpIntegersImport(handle,mod,nlimbs,-1,sizeof(xmpLimb_t),-1,0,limbs,mod_instances));

    for(int i=0;i<xmpPowmAlgorithmsCount;i++) {
      xmpPowmAlgorithm alg=xmpPowmAlgorithms[i];
      if(P>=alg.min_precision && P<=alg.max_precision) {
        compute_and_add_latencies(handle,i,latencies,out,a,exp,mod);
      }
    }
    xmpCheckError(xmpIntegersDestroy(handle, a));
    xmpCheckError(xmpIntegersDestroy(handle, exp));
    xmpCheckError(xmpIntegersDestroy(handle, mod));
    xmpCheckError(xmpIntegersDestroy(handle, out));

    
    free(limbs);

    sort(latencies.begin(),latencies.end());
#if 1
    printf("Precision: %d, latency list:\n",P);
    for(int i=0;i<latencies.size();i++) {
      printf("%d alg: %d, count: %d, instances_per_sm: %d: latency %f, throughput: %e\n", i, latencies[i].alg_index, smCount*latencies[i].instances_per_sm, latencies[i].instances_per_sm, latencies[i].latency,  (smCount*latencies[i].instances_per_sm)/latencies[i].latency);
    }
#endif
    fprintf(file,"const Latency powm_%d_%s[]={\n",P,sarch);
    for(int i=0;i<latencies.size();i++) {
      fprintf(file,"    Latency(%d,%f,%d),\n",latencies[i].alg_index, latencies[i].latency, latencies[i].instances_per_sm);
    }
    fprintf(file,"};\n");

    counts[p]=latencies.size();
   }


  fprintf(file,"const Latency* powm_tbl_%s[]={",sarch);
  for(int p=0;p<xmpPowmPrecisionsCount;p++) {
    uint32_t P=xmpPowmPrecisions[p];
    fprintf(file,"powm_%d_%s,",P,sarch);
  }
  fprintf(file,"};\n");

  fprintf(file,"const uint32_t powm_tbl_%s_counts[]={",sarch);
  for(int p=0;p<xmpPowmPrecisionsCount;p++) {
    fprintf(file,"%d,",counts[p]);
  }
  fprintf(file,"};\n");
 
  fclose(file);

  printf("File: %s successfully written\n",filename);
  xmpCheckError(xmpExecutionPolicyDestroy(handle, policy));
  xmpCheckError(xmpHandleDestroy(handle));
}
