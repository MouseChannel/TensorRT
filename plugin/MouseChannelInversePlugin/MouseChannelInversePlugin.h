#ifndef MOUSECHANNEL_INVERSE_PLUGIN
#define MOUSECHANNEL_INVERSE_PLUGIN
#include "common/plugin.h"
#include <cublas_v2.h>
namespace nvinfer1
{
namespace plugin
{

class MouseChannelInverse : public IPluginV2Ext
{
public:
    MouseChannelInverse();
    ~MouseChannelInverse() override = default;
    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;
    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

    // void detachFromContext() noexcept override;
private:
    cublasHandle_t mCublas;
    float** dd_A = NULL;
     
    float** dd_InvA = NULL;
    int * info;
    std::string mPluginNamespace;
};

class MouseChannelInversePluginCreater : public pluginInternal::BaseCreator
{
public:
    MouseChannelInversePluginCreater();
    ~MouseChannelInversePluginCreater() override = default;
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;
    private:
    static PluginFieldCollection mFC;
};
} // namespace plugin
} // namespace nvinfer1
#endif