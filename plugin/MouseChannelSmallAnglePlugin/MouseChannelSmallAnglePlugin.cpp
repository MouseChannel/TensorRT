

#include "cublas_v2.h"
#include <MouseChannelSmallAnglePlugin.h>
#include <numeric>
#include <vector>
using namespace nvinfer1;
using nvinfer1::plugin::MouseChannelSmallAngle;
using nvinfer1::plugin::MouseChannelSmallAnglePluginCreater;
namespace
{
char const* const kMouseChannelSmallAngle_PLUGIN_VERSION{"1"};
char const* const kMouseChannelSmallAngle_PLUGIN_NAME{"MouseChannelSmallAngle"};
} // namespace
PluginFieldCollection MouseChannelSmallAnglePluginCreater::mFC{};
REGISTER_TENSORRT_PLUGIN(MouseChannelSmallAnglePluginCreater);
MouseChannelSmallAngle::MouseChannelSmallAngle()
{
    
    std::cout << "⛑️⛑️"
              << "I'm MouseChannelSmallAngle Plugin" << std::endl;
}
char const* MouseChannelSmallAngle::getPluginType() const noexcept
{
    return kMouseChannelSmallAngle_PLUGIN_NAME;
}
char const* MouseChannelSmallAngle::getPluginVersion() const noexcept
{
    return kMouseChannelSmallAngle_PLUGIN_VERSION;
}
int32_t MouseChannelSmallAngle::getNbOutputs() const noexcept
{
    return 1;
}
DimsExprs MouseChannelSmallAngle::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{

    DimsExprs out_dim{};
    out_dim.nbDims = 2;
    // std::vector<int> input_dim;
    // int v = 1;
    // for (int i = 0; i < inputs[1].nbDims - 1; i++)
    // {
    //     v *= inputs[1].d[i]->getConstantValue();
    // }
    // std::accumulate(inputs[1].d, inputs[1].d + inputs[1].nbDims - 1, int{1}, std::multiplies<int>{});

    // out_dim.d[0] = exprBuilder.constant(2);
    // out_dim.d[1] = exprBuilder.constant(4);

    out_dim.d[0] = inputs[0].d[0];
    // out_dim.d[0] = inputs[0].d[0];
    out_dim.d[1] = inputs[0].d[1];
    // std::cout << out_dim.d[0] << std::endl;

    // out_dim.d[2] = inputs[0].d[2];

    return out_dim;
}
int32_t MouseChannelSmallAngle::initialize() noexcept
{
    return STATUS_SUCCESS;
}
void MouseChannelSmallAngle::terminate() noexcept {}
size_t MouseChannelSmallAngle::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 120;
}
int32_t MouseChannelSmallAngle::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    auto dims = inputDesc[0].dims.d[0];

    auto sin_half_angles_over_angles = (float*) inputs[0];
    auto small_angles = (float*) inputs[1];
    auto angles = (float*) inputs[2];
    // , float const* small_angles, const float* angles
    RealHandle(sin_half_angles_over_angles, small_angles, angles, (float*) outputs[0], dims);
    // const int onehot_count = std::accumulate(dims.d, dims.d + dims.nbDims, int{1}, std::multiplies<int>{});
    // RealSelect((float*) inputs[0], (float*) inputs[1], (float*) outputs[0], onehot_count);

    // if (int r = cudaMemcpy(outputs[0], inputs[0], sizeof(float) * 2 * 4,
    // cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    //     r != cudaSuccess)
    // {
    //     std::cout << "err" << std::endl;
    //
    //     throw std::runtime_error("err3");
    // }
    //
    // // auto matinv = cublasSmatinvBatched(mCublas, 3, dd_A, 3, dd_InvA, 3, info, 1);
    // // std::cout << matinv << std::endl;
    // std::vector<float> temp1(2 * 4);
    // if (cudaMemcpy(temp1.data(), inputs[1], sizeof(float) * 2 * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost)
    //     != cudaSuccess)
    // {
    //     std::cout << "err" << std::endl;
    //
    //     throw std::runtime_error("err3");
    // }
    // std::vector<float> temp2(2 * 4 * 4);
    // if (cudaMemcpy(temp2.data(), inputs[0], sizeof(float) * 2 * 4 * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost)
    //     != cudaSuccess)
    // {
    //     std::cout << "err" << std::endl;
    //
    //     throw std::runtime_error("err3");
    // }
    // // float temp = *(float*) inputs[0];
    // auto t = inputDesc[1];
    return STATUS_SUCCESS;
}
size_t MouseChannelSmallAngle::getSerializationSize() const noexcept
{
    return 0;
}
void MouseChannelSmallAngle::serialize(void* buffer) const noexcept {}
// bool MouseChannelSmallAngle::supportsFormat(DataType type, PluginFormat format) const noexcept
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
// }
void MouseChannelSmallAngle::destroy() noexcept
{
    delete this;
}
void MouseChannelSmallAngle::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}
char const* MouseChannelSmallAngle::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}
DataType MouseChannelSmallAngle::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}
// bool MouseChannelSmallAngle::isOutputBroadcastAcrossBatch(
//     int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
// {
//     return false;
// }
// bool MouseChannelSmallAngle::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
// {
//     return false;
// }
void MouseChannelSmallAngle::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    mCublas = cublasContext;
}
void MouseChannelSmallAngle::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}
bool MouseChannelSmallAngle::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut[0].type == DataType::kFLOAT;
}

IPluginV2DynamicExt* MouseChannelSmallAngle::clone() const noexcept
{
    try
    {
        IPluginV2DynamicExt* plugin = new MouseChannelSmallAngle;
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
    return nullptr;
}

MouseChannelSmallAnglePluginCreater::MouseChannelSmallAnglePluginCreater() {}
char const* MouseChannelSmallAnglePluginCreater::getPluginName() const noexcept
{
    return kMouseChannelSmallAngle_PLUGIN_NAME;
}
char const* MouseChannelSmallAnglePluginCreater::getPluginVersion() const noexcept
{
    return kMouseChannelSmallAngle_PLUGIN_VERSION;
}
void MouseChannelSmallAnglePluginCreater::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}
char const* MouseChannelSmallAnglePluginCreater::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
PluginFieldCollection const* MouseChannelSmallAnglePluginCreater::getFieldNames() noexcept
{
    return &mFC;
}
IPluginV2DynamicExt* MouseChannelSmallAnglePluginCreater::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {

        auto* obj = new MouseChannelSmallAngle;
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
    return nullptr;
}
IPluginV2DynamicExt* MouseChannelSmallAnglePluginCreater::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new MouseChannelSmallAngle;
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
}
