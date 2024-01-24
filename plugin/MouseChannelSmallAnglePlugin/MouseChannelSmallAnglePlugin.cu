#include "cublas_v2.h"
#include "cuda_runtime.h"
namespace nvinfer1
{
namespace plugin
{
__global__ void DoHandle(float* sin_half_angles_over_angles, float const* small_angles, const float* angles,float* output)
{
    int cur_index = threadIdx.x;
    if (small_angles[cur_index] > 0.5f)
    {
        output[cur_index] = 0.5f - (angles[cur_index] * angles[cur_index]) / 48.f;
    }
    else
    {
        output[cur_index] = sin(angles[cur_index] / 2) / angles[cur_index];
    }
    // sin_half_angles_over_angles[cur_index] = cur_index;
    // output[0] = 1.11f;
}
void RealHandle(float* sin_half_angles_over_angles, float const* small_angles, const float* angles,float* output, int onehot_count)
{
    DoHandle<<<1, onehot_count>>>(sin_half_angles_over_angles, small_angles, angles,  output);
}
} // namespace plugin
} // namespace nvinfer1