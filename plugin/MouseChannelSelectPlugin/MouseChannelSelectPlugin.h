#ifndef MOUSECHANNEL_SELECT_PLUGIN
#define MOUSECHANNEL_SELECT_PLUGIN
#include "common/plugin.h"
#include <cublas_v2.h>
namespace nvinfer1
{
namespace plugin
{
void RealSelect(float const* input, float const* onehot, float* output,int onehot_count);
class MouseChannelSelect : public IPluginV2DynamicExt
{
public:
    MouseChannelSelect();
    ~MouseChannelSelect() override = default;
    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    // bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    void destroy() noexcept override;

    IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // bool isOutputBroadcastAcrossBatch(
    //     int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    // bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    // void detachFromContext() noexcept override;
private:
    cublasHandle_t mCublas;
    std::string mPluginNamespace;
};

class MouseChannelSelectPluginCreater : public nvinfer1::IPluginCreator
{
public:
    MouseChannelSelectPluginCreater();
    ~MouseChannelSelectPluginCreater() override = default;
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
     char const*  getPluginNamespace() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1
#endif