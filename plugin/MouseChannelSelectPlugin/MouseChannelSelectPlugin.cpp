

#include "cublas_v2.h"
#include <MouseChannelSelectPlugin.h>
#include <numeric>
#include <vector>
using namespace nvinfer1;
using nvinfer1::plugin::MouseChannelSelect;
using nvinfer1::plugin::MouseChannelSelectPluginCreater;
namespace
{
char const* const kMouseChannelSelect_PLUGIN_VERSION{"1"};
char const* const kMouseChannelSelect_PLUGIN_NAME{"MouseChannelSelect"};
} // namespace
PluginFieldCollection MouseChannelSelectPluginCreater::mFC{};
REGISTER_TENSORRT_PLUGIN(MouseChannelSelectPluginCreater);
MouseChannelSelect::MouseChannelSelect()
{
    std::cout << "🐒🐒" << "I'm MouseChannelSelect Plugin" << std::endl;
}
char const* MouseChannelSelect::getPluginType() const noexcept
{
    return kMouseChannelSelect_PLUGIN_NAME;
}
char const* MouseChannelSelect::getPluginVersion() const noexcept
{
    return kMouseChannelSelect_PLUGIN_VERSION;
}
int32_t MouseChannelSelect::getNbOutputs() const noexcept
{
    return 1;
}
DimsExprs MouseChannelSelect::getOutputDimensions(
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
    auto v = exprBuilder.constant(1);
    for (int i = 0; i < inputs[0].nbDims - 2; i++)
    {
        v = exprBuilder.operation(DimensionOperation::kPROD, *v, *inputs[0].d[i]);
    }
    out_dim.d[0] = v;
    // out_dim.d[0] = inputs[0].d[0];
    out_dim.d[1] = inputs[0].d[inputs[0].nbDims - 1];
    std::cout << out_dim.d[0] << std::endl;

    // out_dim.d[2] = inputs[0].d[2];

    return out_dim;
}
int32_t MouseChannelSelect::initialize() noexcept
{
    return STATUS_SUCCESS;
}
void MouseChannelSelect::terminate() noexcept {}
size_t MouseChannelSelect::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 120;
}
int32_t MouseChannelSelect::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    auto dims = inputDesc[1].dims;
    const int onehot_count = std::accumulate(dims.d, dims.d + dims.nbDims, int{1}, std::multiplies<int>{});
    RealSelect((float*) inputs[0], (float*) inputs[1], (float*) outputs[0], onehot_count);

    // if (int r = cudaMemcpy(outputs[0], inputs[0], sizeof(float) * 2 * 4, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    //     r != cudaSuccess)
    // {
    //     std::cout << "err" << std::endl;
    //
    //     throw std::runtime_error("err3");
    // }
    //
    // // auto matinv = cublasSmatinvBatched(mCublas, 3, dd_A, 3, dd_InvA, 3, info, 1);
    // // std::cout << matinv << std::endl;
    std::vector<float> temp1(2 * 4);
    if (cudaMemcpy(temp1.data(), inputs[1], sizeof(float) * 2 * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost)
        != cudaSuccess)
    {
        std::cout << "err" << std::endl;

        throw std::runtime_error("err3");
    }
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
size_t MouseChannelSelect::getSerializationSize() const noexcept
{
    return 0;
}
void MouseChannelSelect::serialize(void* buffer) const noexcept {}
// bool MouseChannelSelect::supportsFormat(DataType type, PluginFormat format) const noexcept
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
// }
void MouseChannelSelect::destroy() noexcept
{
    delete this;
}
void MouseChannelSelect::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}
char const* MouseChannelSelect::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}
DataType MouseChannelSelect::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}
// bool MouseChannelSelect::isOutputBroadcastAcrossBatch(
//     int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
// {
//     return false;
// }
// bool MouseChannelSelect::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
// {
//     return false;
// }
void MouseChannelSelect::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    mCublas = cublasContext;
}
void MouseChannelSelect::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}
bool MouseChannelSelect::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut[0].type == DataType::kFLOAT;
}

IPluginV2DynamicExt* MouseChannelSelect::clone() const noexcept
{
    try
    {
        IPluginV2DynamicExt* plugin = new MouseChannelSelect;
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
    return nullptr;
}

MouseChannelSelectPluginCreater::MouseChannelSelectPluginCreater() {}
char const* MouseChannelSelectPluginCreater::getPluginName() const noexcept
{
    return kMouseChannelSelect_PLUGIN_NAME;
}
char const* MouseChannelSelectPluginCreater::getPluginVersion() const noexcept
{
    return kMouseChannelSelect_PLUGIN_VERSION;
}
void MouseChannelSelectPluginCreater::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}
char const* MouseChannelSelectPluginCreater::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
PluginFieldCollection const* MouseChannelSelectPluginCreater::getFieldNames() noexcept
{
    return &mFC;
}
IPluginV2DynamicExt* MouseChannelSelectPluginCreater::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {

        auto* obj = new MouseChannelSelect;
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
    return nullptr;
}
IPluginV2DynamicExt* MouseChannelSelectPluginCreater::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new MouseChannelSelect;
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }
}
