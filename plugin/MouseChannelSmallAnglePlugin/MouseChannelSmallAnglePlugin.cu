#include "cublas_v2.h"
#include "cuda_runtime.h"
namespace nvinfer1
{
namespace plugin
{
__global__ void DoSelect(float const* input, float const* onehot, float* output)
{
    int cur_index = threadIdx.x;
    if (onehot[cur_index] > 0.5f)
    {
        // int index1 = cur_index / 4;
        // int index2 = cur_index % 4;
        // output[cur_index] = input[cur_index * 4];
        // output[cur_index + 1] = input[cur_index * 4 + 1];
        // output[cur_index + 2] = input[cur_index * 4 + 2];
        // output[cur_index + 3] = input[cur_index * 4 + 3];
        

        int out_index = (cur_index / 4) *4;
        output[out_index] = input[cur_index * 4];
     
        output[out_index + 1] = input[cur_index * 4 + 1];
        
        output[out_index + 2] = input[cur_index * 4 + 2];
       
        output[out_index + 3] = input[cur_index * 4 + 3];
        // output[out_index] = cur_index;
        
        // printf("%d  \n", cur_index);
    }
    // output[0] = 1.11f;
}
void RealSelect(float const* input, float const* onehot, float* output,int onehot_count)
{
    DoSelect<<<1, onehot_count>>>(input, onehot, output);
}
} // namespace plugin
} // namespace nvinfer1