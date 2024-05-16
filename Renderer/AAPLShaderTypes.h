/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The header that contains types and enumerated constants that the Metal shaders and the C/ObjC source share.
*/
#ifndef AAPLShaderTypes_h
#define AAPLShaderTypes_h

#include <simd/simd.h>

typedef enum AAPLConstantIndex
{
    AAPLConstantIndexRayTracingEnabled
} AAPLConstantIndex;

typedef enum RTReflectionKernelImageIndex
{
    OutImageIndex                   = 0,
    ThinGBufferPositionIndex        = 1,
    ThinGBufferDirectionIndex       = 2,
    IrradianceMapIndex              = 3,
    RefectionMapIndex               = 4,
} RTReflectionKernelImageIndex;

typedef enum RTReflectionKernelBufferIndex
{
    SceneIndex                      = 0,
    AccelerationStructureIndex      = 1
} RTReflectionKernelBufferIndex;

typedef enum BufferIndex
{
    BufferIndexMeshPositions        = 0,
    BufferIndexMeshGenerics         = 1,
    BufferIndexInstanceTransforms   = 2,
    BufferIndexCameraData           = 3,
    BufferIndexLightData            = 4,
    BufferIndexSubmeshKeypath       = 5
} BufferIndex;

typedef enum VertexAttribute
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
} VertexAttribute;

// The attribute index values that the shader and the C code share to ensure Metal
// shader vertex attribute indices match the Metal API vertex descriptor attribute indices.
typedef enum AAPLVertexAttribute
{
    AAPLVertexAttributePosition  = 0,
    AAPLVertexAttributeTexcoord  = 1,
    AAPLVertexAttributeNormal    = 2,
    AAPLVertexAttributeTangent   = 3,
    AAPLVertexAttributeBitangent = 4
} AAPLVertexAttribute;

// The texture index values that the shader and the C code share to ensure
// Metal shader texture indices match indices of Metal API texture set calls.
typedef enum AAPLTextureIndex
{
    AAPLTextureIndexBaseColor        = 0,
    AAPLTextureIndexMetallic         = 1,
    AAPLTextureIndexRoughness        = 2,
    AAPLTextureIndexNormal           = 3,
    AAPLTextureIndexAmbientOcclusion = 4,
    AAPLTextureIndexIrradianceMap    = 5,
    AAPLTextureIndexReflections      = 6,
    AAPLTextureIndexGI               = 7,
    AAPLTextureIndexIrrGI               = 8,
    AAPLSkyDomeTexture               = 9,
    AAPLTextureIndexGBufferAlbedo = 10,
    AAPLMaterialTextureCount = AAPLTextureIndexAmbientOcclusion,
} AAPLTextureIndex;

// The buffer index values that the shader and the C code share to
// ensure Metal shader buffer inputs match Metal API buffer set calls.
typedef enum AAPLBufferIndex
{
    AAPLBufferIndexMeshPositions    = 0,
    AAPLBufferIndexMeshGenerics     = 1,
} AAPLBufferIndex;

typedef struct AAPLInstanceTransform
{
    matrix_float4x4 modelViewMatrix;
} AAPLInstanceTransform;

typedef struct AAPLCameraData
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 invProjectionMatrix;
    matrix_float4x4 viewMatrix;
    matrix_float4x4 invViewMatrix;
    vector_float3 cameraPosition;
    float metallicBias;
    float roughnessBias;
    
    matrix_float4x4 prevProjectionMatrix;
    matrix_float4x4 prevViewMatrix;
    vector_float3 prevCameraPosition;
    
    vector_float2 jitter;
    vector_float2 prev_jitter;
    unsigned int width;
    unsigned int height;
    unsigned int frameIndex;
    
} AAPLCameraData;

// The structure that the shader and the C code share to ensure the layout of
// data accessed in Metal shaders matches the layout of data set in C code.
typedef struct
{
    // Per Light Properties
    vector_float3 directionalLightInvDirection;
    float lightIntensity;

} AAPLLightData;

typedef struct AAPLSubmeshKeypath
{
    uint32_t instanceID;
    uint32_t submeshID;
} AAPLSubmeshKeypath;

#ifdef __METAL_VERSION__
#define CONSTANT constant
#define vector2 float2

// Halton(2, 3) sequence used for temporal antialiasing and shadow ray sampling
CONSTANT vector_float2 haltonSamples[] = {
    vector2(0.5f, 0.333333333333f),
    vector2(0.25f, 0.666666666667f),
    vector2(0.75f, 0.111111111111f),
    vector2(0.125f, 0.444444444444f),
    vector2(0.625f, 0.777777777778f),
    vector2(0.375f, 0.222222222222f),
    vector2(0.875f, 0.555555555556f),
    vector2(0.0625f, 0.888888888889f),
    vector2(0.5625f, 0.037037037037f),
    vector2(0.3125f, 0.37037037037f),
    vector2(0.8125f, 0.703703703704f),
    vector2(0.1875f, 0.148148148148f),
    vector2(0.6875f, 0.481481481481f),
    vector2(0.4375f, 0.814814814815f),
    vector2(0.9375f, 0.259259259259f),
    vector2(0.03125f, 0.592592592593f),
};

#endif

#endif /* ShaderTypes_h */

