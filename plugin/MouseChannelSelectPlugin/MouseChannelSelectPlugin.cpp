

#include "cublas_v2.h"
#include <MouseChannelInversePlugin.h>
using namespace nvinfer1;
using nvinfer1::plugin::MouseChannelInverse;
using nvinfer1::plugin::MouseChannelInversePluginCreater;
namespace
{
char const* const kMouseChannelInverse_PLUGIN_VERSION{"1"};
char const* const kMouseChannelInverse_PLUGIN_NAME{"MouseChannelInverse"};
} // namespace
PluginFieldCollection MouseChannelInversePluginCreater::mFC{};
REGISTER_TENSORRT_PLUGIN(MouseChannelInversePluginCreater);
MouseChannelInverse::MouseChannelInverse()
{
    auto dd_1 = cudaMalloc(&dd_A, sizeof(float*));
    auto dd_2 = cudaMalloc(&dd_InvA, sizeof(float*));
    auto a_info = cudaMalloc((void**) &info, 4);
    std::cout << "🎉🎉" << dd_1 << dd_1 << a_info << std::endl;
}
char const* MouseChannelInverse::getPluginType() const noexcept
{
    return kMouseChannelInverse_PLUGIN_NAME;
}
char const* MouseChannelInverse::getPluginVersion() const noexcept
{
    return kMouseChannelInverse_PLUGIN_VERSION;
}
int32_t MouseChannelInverse::getNbOutputs() const noexcept
{
    return 1;
}
DimsExprs MouseChannelInverse::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out_dim;
    out_dim.nbDims = 3;
    out_dim.d[0] = inputs[0].d[0];
    
    out_dim.d[1] = inputs[0].d[1];
    
    out_dim.d[2] = inputs[0].d[2];
    
    return out_dim;
}
int32_t MouseChannelInverse::initialize() noexcept
{
    return STATUS_SUCCESS;
}
void MouseChannelInverse::terminate() noexcept {}
size_t MouseChannelInverse::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 120;
}
int32_t MouseChannelInverse::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    if (cudaMemcpy(dd_A, inputs, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "err" << std::endl;
        throw std::runtime_error("err3");
    }
    if (cudaMemcpy(dd_InvA, outputs, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "err" << std::endl;

        throw std::runtime_error("err3");
    }
    auto matinv = cublasSmatinvBatched(mCublas, 3, dd_A, 3, dd_InvA, 3, info, 1);
    // std::cout << matinv << std::endl;
    return STATUS_SUCCESS;
}
size_t MouseChannelInverse::getSerializationSize() const noexcept
{
    return 0;
}
void MouseChannelInverse::serialize(void* buffer) const noexcept {}
// bool MouseChannelInverse::supportsFormat(DataType type, PluginFormat format) const noexcept
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
// }
void MouseChannelInverse::destroy() noexcept
{
    delete this;
}
void MouseChannelInverse::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}
char const* MouseChannelInverse::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}
DataType MouseChannelInverse::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}
// bool MouseChannelInverse::isOutputBroadcastAcrossBatch(
//     int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
// {
//     return false;
// }
// bool MouseChannelInverse::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
// {
//     return false;
// }
void MouseChannelInverse::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    mCublas = cublasContext;
}
void MouseChannelInverse::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}
bool MouseChannelInverse::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut[0].type == DataType::kFLOAT;
}

IPluginV2DynamicExt* MouseChannelInverse::clone() const noexcept
{
    try
    {
        IPluginV2DynamicExt* plugin = new MouseChannelInverse;
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
    return nullptr;
}

MouseChannelInversePluginCreater::MouseChannelInversePluginCreater() {}
char const* MouseChannelInversePluginCreater::getPluginName() const noexcept
{
    return kMouseChannelInverse_PLUGIN_NAME;
}
char const* MouseChannelInversePluginCreater::getPluginVersion() const noexcept
{
    return kMouseChannelInverse_PLUGIN_VERSION;
}
void MouseChannelInversePluginCreater::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}
char const* MouseChannelInversePluginCreater::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
PluginFieldCollection const* MouseChannelInversePluginCreater::getFieldNames() noexcept
{
    return &mFC;
}
IPluginV2DynamicExt* MouseChannelInversePluginCreater::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {

        auto* obj = new MouseChannelInverse;
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
    return nullptr;
}
IPluginV2DynamicExt* MouseChannelInversePluginCreater::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new MouseChannelInverse;
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
}
