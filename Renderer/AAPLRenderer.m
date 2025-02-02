/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The implemenation of the renderer class that performs Metal setup and per-frame rendering.
*/

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <ModelIO/ModelIO.h>
#import "AAPLMathUtilities.h"

#import "AAPLRenderer.h"

#import "AAPLMesh.h"

// Include the headers that share types between the C code here, which executes
// Metal API commands, and the .metal files, which use the types as inputs to the shaders.
#include <simd/simd.h>
#import "AAPLShaderTypes.h"
#import "AAPLArgumentBufferTypes.h"

MTLPackedFloat4x3 matrix4x4_drop_last_row(matrix_float4x4 m)
{
    return (MTLPackedFloat4x3){
        MTLPackedFloat3Make( m.columns[0].x, m.columns[0].y, m.columns[0].z ),
        MTLPackedFloat3Make( m.columns[1].x, m.columns[1].y, m.columns[1].z ),
        MTLPackedFloat3Make( m.columns[2].x, m.columns[2].y, m.columns[2].z ),
        MTLPackedFloat3Make( m.columns[3].x, m.columns[3].y, m.columns[3].z )
    };
}

static const NSUInteger kMaxBuffersInFlight = 3;

// How to add a new instance:
// 1. Increase kMaxInstances to include the new instance.
// 2. Create the mesh in method loadAssets.
// 3. Modify initializeModelInstances to reference your mesh and set its transform.

// The maximum number of objects in the world (not counting the skybox).37
static const NSUInteger kMaxInstances = 2;

static const size_t kAlignedInstanceTransformsStructSize = (sizeof(AAPLInstanceTransform) & ~0xFF) + 0x100;

typedef enum AccelerationStructureEvents : uint64_t
{
    kPrimitiveAccelerationStructureBuild = 1,
    kInstanceAccelerationStructureBuild = 2
} AccelerationStructureEvents;

typedef struct ModelInstance
{
    uint32_t meshIndex;     // The mesh corresponding to this instance.
    vector_float3 position; // The position of this instance in the world.
    float rotationRad;      // The Y rotation of this instance in the world.
} ModelInstance;

typedef struct ThinGBuffer
{
    id<MTLTexture> positionTexture;
    id<MTLTexture> depthNormalTexture;
    id<MTLTexture> PrevDepthNormalTexture;
    id<MTLTexture> motionVectorTexture;
    id<MTLTexture> albedoTexture;
} ThinGBuffer;

@implementation AAPLRenderer
{
    dispatch_semaphore_t _inFlightSemaphore;

    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;

    id<MTLBuffer> _lightDataBuffer;
    id<MTLBuffer> _cameraDataBuffers[kMaxBuffersInFlight];
    id<MTLBuffer> _instanceTransformBuffer;

    id<MTLRenderPipelineState> _pipelineState;
    id<MTLRenderPipelineState> _pipelineStateNoRT;
    id<MTLRenderPipelineState> _pipelineStateReflOnly;
    id<MTLRenderPipelineState> _gbufferPipelineState;
    id<MTLRenderPipelineState> _skyboxPipelineState;
    id<MTLDepthStencilState> _depthState;

    MTLVertexDescriptor *_mtlVertexDescriptor;
    MTLVertexDescriptor *_mtlSkyboxVertexDescriptor;

    uint8_t _cameraBufferIndex;
    matrix_float4x4 _projectionMatrix;

    NSArray< AAPLMesh* >* _meshes;
    AAPLMesh* _skybox;
    id<MTLTexture> _skyMap;

    ModelInstance _modelInstances[kMaxInstances];
    id<MTLEvent> _accelerationStructureBuildEvent;
    id<MTLAccelerationStructure> _instanceAccelerationStructure;
    NSArray< id<MTLAccelerationStructure> >* _primitiveAccelerationStructures;
    id< MTLHeap > _accelerationStructureHeap;

    // Reflection
    id<MTLTexture> _rtReflectionMap;
    id<MTLFunction> _rtReflectionFunction;
    id<MTLComputePipelineState> _rtReflectionPipeline;
    id<MTLHeap> _rtMipmappingHeap;
    id<MTLRenderPipelineState> _rtMipmapPipeline;
    
    // First Bounce, Screen Idependent
    id<MTLTexture> _rtShadingMap;
    id<MTLTexture> _rtIrradianceMap;
    id<MTLFunction> _rtShadingFunction;
    id<MTLComputePipelineState> _rtShadingPipeline;
    
    // 2nd Bounce, Screen Relative
    id<MTLTexture> _rtBounceMap;
    id<MTLFunction> _rtBounceFunction;
    id<MTLComputePipelineState> _rtBouncePipeline;
    
    // GroundTruth
    id<MTLTexture> _rtGroundTruthMap;
    id<MTLFunction> _rtGroundTruthFunction;
    id<MTLComputePipelineState> _rtGroundTruthPipeline;
    
    // Postprocessing pipelines.
    id<MTLRenderPipelineState> _bloomThresholdPipeline;
    id<MTLRenderPipelineState> _postMergePipeline;
    id<MTLTexture> _rawColorMap;
    id<MTLTexture> _bloomThresholdMap;
    id<MTLTexture> _bloomBlurMap;
    
    // Denoiser
    MPSSVGFDefaultTextureAllocator *_textureAllocator;
    MPSSVGFDenoiser *_denoiser;
    MPSSVGFDenoiser *_denoiserIrr;
    MPSSVGFDenoiser *_denoiserRefl;
    MPSTemporalAA *_TAA;
    
    ThinGBuffer _thinGBuffer;

    // Argument buffers.
    NSSet< id<MTLResource> >* _sceneResources;
    id<MTLBuffer> _sceneArgumentBuffer;

    float _cameraAngle;
    float _cameraPanSpeedFactor;
    float _metallicBias;
    float _roughnessBias;
    float _exposure;
    RenderMode _renderMode;
    
    int _frameCount;
    CGSize _size;
}

- (void)loadMPSSVGF {
    // Create an object which allocates and caches intermediate textures
    // throughout and across frames
    _textureAllocator = [[MPSSVGFDefaultTextureAllocator alloc] initWithDevice:_device];
    
    MPSSVGF *svgf = [[MPSSVGF alloc] initWithDevice:_device];
    svgf.channelCount = 3;
    svgf.temporalWeighting = MPSTemporalWeightingExponentialMovingAverage;
    svgf.temporalReprojectionBlendFactor = 0.04f;
    svgf.minimumFramesForVarianceEstimation = 16;
    _denoiser = [[MPSSVGFDenoiser alloc] initWithSVGF:svgf textureAllocator:_textureAllocator];
    _denoiser.bilateralFilterIterations = 5;
    _denoiserIrr = [[MPSSVGFDenoiser alloc] initWithSVGF:svgf textureAllocator:_textureAllocator];
    _denoiserIrr.bilateralFilterIterations = 5;
    _denoiserRefl = [[MPSSVGFDenoiser alloc] initWithSVGF:svgf textureAllocator:_textureAllocator];
    _denoiserRefl.bilateralFilterIterations = 5;

    // Create the temporal antialiasing object
    _TAA = [[MPSTemporalAA alloc] initWithDevice:_device];
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view size:(CGSize)size
{
    self = [super init];
    if(self)
    {
        _device = view.device;
        _inFlightSemaphore = dispatch_semaphore_create(kMaxBuffersInFlight);
        _accelerationStructureBuildEvent = [_device newEvent];
        [self initializeModelInstances];
        _projectionMatrix = [self projectionMatrixWithAspect:size.width / size.height];
        [self loadMetalWithView:view];
        [self loadAssets];
        
        BOOL createdArgumentBuffers = NO;
        
        char* opts = getenv("DISABLE_METAL3_FEATURES");
        if ( (opts == NULL) || (strstr(opts, "1") != opts))
        {
            if ( @available( macOS 13, iOS 16, *) )
            {
                if( [_device supportsFamily:MTLGPUFamilyMetal3] )
                {
                    // metal3的argumentBuffer，aka gpuscene
                    createdArgumentBuffers = YES;
                    [self buildSceneArgumentBufferMetal3];
                }
            }
        }

        // Call this last to ensure everything else builds.
        [self resizeRTReflectionMapTo:size];
        // ray-trace加速结构
        [self buildRTAccelerationStructures];
        _cameraAngle = 0.5 * M_PI;
        _cameraPanSpeedFactor = 0.5f;
        _metallicBias = 0.0f;
        _roughnessBias = 0.0f;
        _exposure = 1.5f;
        
        [self loadMPSSVGF];

    }

    return self;
}

- (void)initializeModelInstances
{
    NSAssert(kMaxInstances == 2, @"Expected 3 Model Instances");

    _modelInstances[0].meshIndex = 0;
    _modelInstances[0].position = (vector_float3){0, 0, 0.0f};
    _modelInstances[0].rotationRad = 0 * M_PI / 180.0f;

//    _modelInstances[1].meshIndex = 0;
//    _modelInstances[1].position = (vector_float3){40.0f, -5.0f, -80.0f};
//    _modelInstances[1].rotationRad = -60 * M_PI / 180.0f;
//    
//    _modelInstances[2].meshIndex = 0;
//    _modelInstances[2].position = (vector_float3){40.0f, 10.0f, -80.0f};
//    _modelInstances[2].rotationRad = -60 * M_PI / 180.0f;
    
    _modelInstances[1].meshIndex = 2;
    _modelInstances[1].position = (vector_float3){0.0f, -1000.01f, 0.0f};
    _modelInstances[1].rotationRad = 0 * M_PI / 180.0f;
    
//    for( int i = 4; i < kMaxInstances; i += 2)
//    {
//        _modelInstances[i].meshIndex = 0;
//        _modelInstances[i].position = (vector_float3){-80.0f + 5.0 * (i - 4), -5.0f, -60.0f * (i - 4) -40.0f};
//        _modelInstances[i].rotationRad = -60 * M_PI / 180.0f;
//        
//        _modelInstances[i+1].meshIndex = 0;
//        _modelInstances[i+1].position = (vector_float3){-80.0f + 5.0 * (i - 4), 10.0f, -60.0f * (i - 4) -40.0f};
//        _modelInstances[i+1].rotationRad = -60 * M_PI / 180.0f;
//    }
}

- (void)resizeRTReflectionMapTo:(CGSize)size
{
    _size = size;
    
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG11B10Float
                                                                                    width:size.width
                                                                                   height:size.height
                                                                                mipmapped:YES];
    
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    _rtReflectionMap = [_device newTextureWithDescriptor:desc];
    _rtShadingMap = [_device newTextureWithDescriptor:desc];
    _rtIrradianceMap = [_device newTextureWithDescriptor:desc];
    _rtBounceMap = [_device newTextureWithDescriptor:desc];
    _rtGroundTruthMap = [_device newTextureWithDescriptor:desc];
    
    desc.mipmapLevelCount = 1;
    //_rawColorMap = [_device newTextureWithDescriptor:desc];
    _bloomThresholdMap = [_device newTextureWithDescriptor:desc];
    _bloomBlurMap = [_device newTextureWithDescriptor:desc];

    desc.pixelFormat = MTLPixelFormatRGBA16Float;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    _thinGBuffer.positionTexture = [_device newTextureWithDescriptor:desc];
    _thinGBuffer.depthNormalTexture = [_textureAllocator textureWithPixelFormat:MTLPixelFormatRGBA16Float width:_size.width height:_size.height];
    _thinGBuffer.motionVectorTexture = [_textureAllocator textureWithPixelFormat:MTLPixelFormatRG16Float width:_size.width height:_size.height];
    _thinGBuffer.albedoTexture = [_textureAllocator textureWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB width:_size.width height:_size.height];
    MTLHeapDescriptor* hd = [[MTLHeapDescriptor alloc] init];
    hd.size = size.width * size.height * 4 * 2 * 3;
    hd.storageMode = MTLStorageModePrivate;
    _rtMipmappingHeap = [_device newHeapWithDescriptor:hd];
    
    
}

#pragma mark - Build Pipeline States

/// Load the Metal state objects and initialize the renderer-dependent view properties.
- (void)loadMetalWithView:(nonnull MTKView *)view;
{
    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;

    _mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];

    // Positions.
    _mtlVertexDescriptor.attributes[AAPLVertexAttributePosition].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[AAPLVertexAttributePosition].offset = 0;
    _mtlVertexDescriptor.attributes[AAPLVertexAttributePosition].bufferIndex = AAPLBufferIndexMeshPositions;

    // Texture coordinates.
    _mtlVertexDescriptor.attributes[AAPLVertexAttributeTexcoord].format = MTLVertexFormatFloat2;
    _mtlVertexDescriptor.attributes[AAPLVertexAttributeTexcoord].offset = 0;
    _mtlVertexDescriptor.attributes[AAPLVertexAttributeTexcoord].bufferIndex = AAPLBufferIndexMeshGenerics;

    // Normals.
    _mtlVertexDescriptor.attributes[AAPLVertexAttributeNormal].format = MTLVertexFormatHalf4;
    _mtlVertexDescriptor.attributes[AAPLVertexAttributeNormal].offset = 8;
    _mtlVertexDescriptor.attributes[AAPLVertexAttributeNormal].bufferIndex = AAPLBufferIndexMeshGenerics;

    // Position Buffer Layout
    _mtlVertexDescriptor.layouts[AAPLBufferIndexMeshPositions].stride = 12;
    _mtlVertexDescriptor.layouts[AAPLBufferIndexMeshPositions].stepRate = 1;
    _mtlVertexDescriptor.layouts[AAPLBufferIndexMeshPositions].stepFunction = MTLVertexStepFunctionPerVertex;

    // Generic Attribute Buffer Layout
    _mtlVertexDescriptor.layouts[AAPLBufferIndexMeshGenerics].stride = 32;
    _mtlVertexDescriptor.layouts[AAPLBufferIndexMeshGenerics].stepRate = 1;
    _mtlVertexDescriptor.layouts[AAPLBufferIndexMeshGenerics].stepFunction = MTLVertexStepFunctionPerVertex;
    
    _mtlSkyboxVertexDescriptor = [[MTLVertexDescriptor alloc] init];
    _mtlSkyboxVertexDescriptor.attributes[VertexAttributePosition].format = MTLVertexFormatFloat3;
    _mtlSkyboxVertexDescriptor.attributes[VertexAttributePosition].offset = 0;
    _mtlSkyboxVertexDescriptor.attributes[VertexAttributePosition].bufferIndex = BufferIndexMeshPositions;
    _mtlSkyboxVertexDescriptor.attributes[VertexAttributeTexcoord].format = MTLVertexFormatFloat2;
    _mtlSkyboxVertexDescriptor.attributes[VertexAttributeTexcoord].offset = 0;
    _mtlSkyboxVertexDescriptor.attributes[VertexAttributeTexcoord].bufferIndex = BufferIndexMeshGenerics;
    _mtlSkyboxVertexDescriptor.layouts[BufferIndexMeshPositions].stride = 12;
    _mtlSkyboxVertexDescriptor.layouts[BufferIndexMeshPositions].stepRate = 1;
    _mtlSkyboxVertexDescriptor.layouts[BufferIndexMeshPositions].stepFunction = MTLVertexStepFunctionPerVertex;
    _mtlSkyboxVertexDescriptor.layouts[BufferIndexMeshGenerics].stride = sizeof(simd_float2);
    _mtlSkyboxVertexDescriptor.layouts[BufferIndexMeshGenerics].stepRate = 1;
    _mtlSkyboxVertexDescriptor.layouts[BufferIndexMeshGenerics].stepRate = MTLVertexStepFunctionPerVertex;

    NSError* error;
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];

    {
        id <MTLFunction> vertexFunction = [defaultLibrary newFunctionWithName:@"vertexShader"];

        MTLFunctionConstantValues* functionConstants = [MTLFunctionConstantValues new];

        MTLRenderPipelineDescriptor *pipelineStateDescriptor = [MTLRenderPipelineDescriptor new];

        {
            BOOL enableRaytracing = YES;
            [functionConstants setConstantValue:&enableRaytracing type:MTLDataTypeBool atIndex:AAPLConstantIndexRayTracingEnabled];
            id <MTLFunction> fragmentFunction = [defaultLibrary newFunctionWithName:@"fragmentShader" constantValues:functionConstants error:nil];

            pipelineStateDescriptor.label = @"RT Pipeline";
            pipelineStateDescriptor.rasterSampleCount = view.sampleCount;
            pipelineStateDescriptor.vertexFunction = vertexFunction;
            pipelineStateDescriptor.fragmentFunction = fragmentFunction;
            pipelineStateDescriptor.vertexDescriptor = _mtlVertexDescriptor;
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRG11B10Float; //view.colorPixelFormat;
            pipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
            pipelineStateDescriptor.stencilAttachmentPixelFormat = view.depthStencilPixelFormat;

            _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            NSAssert(_pipelineState, @"Failed to create pipeline state: %@", error);
        }

        {
            BOOL enableRaytracing = NO;
            [functionConstants setConstantValue:&enableRaytracing type:MTLDataTypeBool atIndex:AAPLConstantIndexRayTracingEnabled];
            id<MTLFunction> fragmentFunctionNoRT = [defaultLibrary newFunctionWithName:@"fragmentShader" constantValues:functionConstants error:nil];

            pipelineStateDescriptor.label = @"No RT Pipeline";
            pipelineStateDescriptor.fragmentFunction = fragmentFunctionNoRT;

            _pipelineStateNoRT = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            NSAssert(_pipelineStateNoRT, @"Failed to create No RT pipeline state: %@", error);
        }

        {
            pipelineStateDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"irradianceShader"];
            pipelineStateDescriptor.label = @"Reflection Viewer Pipeline";

            _pipelineStateReflOnly = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            NSAssert(_pipelineStateNoRT, @"Failed to create Reflection Viewer pipeline state: %@", error);
        }

        {
            id<MTLFunction> gBufferFragmentFunction = [defaultLibrary newFunctionWithName:@"gBufferFragmentShader"];
            pipelineStateDescriptor.label = @"ThinGBufferPipeline";
            pipelineStateDescriptor.fragmentFunction = gBufferFragmentFunction;
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
            pipelineStateDescriptor.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
            pipelineStateDescriptor.colorAttachments[2].pixelFormat = MTLPixelFormatRG16Float;
            pipelineStateDescriptor.colorAttachments[3].pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;

            _gbufferPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            NSAssert(_gbufferPipelineState, @"Failed to create GBuffer pipeline state: %@", error);
        }

        {
            id<MTLFunction> skyboxVertexFunction = [defaultLibrary newFunctionWithName:@"skyboxVertex"];
            id<MTLFunction> skyboxFragmentFunction = [defaultLibrary newFunctionWithName:@"skyboxFragment"];
            pipelineStateDescriptor.label = @"SkyboxPipeline";
            pipelineStateDescriptor.vertexDescriptor = _mtlSkyboxVertexDescriptor;
            pipelineStateDescriptor.vertexFunction = skyboxVertexFunction;
            pipelineStateDescriptor.fragmentFunction = skyboxFragmentFunction;
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRG11B10Float;
            pipelineStateDescriptor.colorAttachments[1].pixelFormat = MTLPixelFormatInvalid;
            pipelineStateDescriptor.colorAttachments[2].pixelFormat = MTLPixelFormatInvalid;
            pipelineStateDescriptor.colorAttachments[3].pixelFormat = MTLPixelFormatInvalid;

             _skyboxPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            NSAssert(_skyboxPipelineState, @"Failed to create Skybox Render Pipeline State: %@", error );
        }
    }

    if(_device.supportsRaytracing)
    {
        _rtShadingFunction = [defaultLibrary newFunctionWithName:@"rtShading"];
        _rtShadingPipeline = [_device newComputePipelineStateWithFunction:_rtShadingFunction error:&error];
        NSAssert(_rtShadingPipeline, @"Failed to create RT shading compute pipeline state: %@", error);
        
        _rtBounceFunction = [defaultLibrary newFunctionWithName:@"rtBounce"];
        _rtBouncePipeline = [_device newComputePipelineStateWithFunction:_rtBounceFunction error:&error];
        NSAssert(_rtBouncePipeline, @"Failed to create RT shading compute pipeline state: %@", error);
        
        _rtGroundTruthFunction = [defaultLibrary newFunctionWithName:@"rtGroundTruth"];
        _rtGroundTruthPipeline = [_device newComputePipelineStateWithFunction:_rtGroundTruthFunction error:&error];
        NSAssert(_rtGroundTruthPipeline, @"Failed to create RT shading compute pipeline state: %@", error);
        
        _renderMode = RMMetalRaytracing2;
    }
    else
    {
        _renderMode = RMNoRaytracing;
    }
    
    {
        id< MTLFunction > passthroughVert = [defaultLibrary newFunctionWithName:@"vertexPassthrough" ];
        id< MTLFunction > fragmentFn = [defaultLibrary newFunctionWithName:@"fragmentPassthrough"];
        MTLRenderPipelineDescriptor* passthroughDesc = [[MTLRenderPipelineDescriptor alloc] init];
        passthroughDesc.vertexFunction = passthroughVert;
        passthroughDesc.fragmentFunction = fragmentFn;
        passthroughDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRG11B10Float;
        
        NSError* __autoreleasing error = nil;
        _rtMipmapPipeline = [_device newRenderPipelineStateWithDescriptor:passthroughDesc error:&error];
        NSAssert( _rtMipmapPipeline, @"Error creating passthrough pipeline: %@", error.localizedDescription );
        
        fragmentFn = [defaultLibrary newFunctionWithName:@"fragmentBloomThreshold"];
        passthroughDesc.fragmentFunction = fragmentFn;
        passthroughDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRG11B10Float;
        _bloomThresholdPipeline = [_device newRenderPipelineStateWithDescriptor:passthroughDesc error:&error];
        NSAssert( _bloomThresholdPipeline, @"Error creating bloom threshold pipeline: %@", error.localizedDescription );
        
        fragmentFn = [defaultLibrary newFunctionWithName:@"fragmentPostprocessMerge"];
        passthroughDesc.fragmentFunction = fragmentFn;
        passthroughDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
        _postMergePipeline = [_device newRenderPipelineStateWithDescriptor:passthroughDesc error:&error];
        NSAssert( _postMergePipeline, @"Error creating postprocessing merge pass: %@", error.localizedDescription);
    }

    {
        MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
        depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
        depthStateDesc.depthWriteEnabled = YES;

        _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];
    }

    for(int i = 0; i < kMaxBuffersInFlight; i++)
    {
        _cameraDataBuffers[i] = [_device newBufferWithLength:sizeof(AAPLCameraData)
                                                 options:MTLResourceStorageModeShared];

        _cameraDataBuffers[i].label = [NSString stringWithFormat:@"CameraDataBuffer %d", i];
    }

    NSUInteger instanceBufferSize = kAlignedInstanceTransformsStructSize * kMaxInstances;
    _instanceTransformBuffer = [_device newBufferWithLength:instanceBufferSize
                                                    options:MTLResourceStorageModeShared];
    _instanceTransformBuffer.label = @"InstanceTransformBuffer";


    _lightDataBuffer = [_device newBufferWithLength:sizeof(AAPLLightData) options:MTLResourceStorageModeShared];
    _lightDataBuffer.label = @"LightDataBuffer";

    _commandQueue = [_device newCommandQueue];

    [self setStaticState];
}

#pragma mark - Asset Loading

/// Create and load assets into Metal objects, including meshes and textures.
- (void)loadAssets
{
    NSError *error;

    // Create a Model I/O vertexDescriptor to format the Model I/O mesh vertices to
    // fit the Metal render pipeline's vertex descriptor layout.
    MDLVertexDescriptor *modelIOVertexDescriptor =
        MTKModelIOVertexDescriptorFromMetal(_mtlVertexDescriptor);

    // Indicate the Metal vertex descriptor attribute mapping for each Model I/O attribute.
    modelIOVertexDescriptor.attributes[AAPLVertexAttributePosition].name  = MDLVertexAttributePosition;
    modelIOVertexDescriptor.attributes[AAPLVertexAttributeTexcoord].name  = MDLVertexAttributeTextureCoordinate;
    modelIOVertexDescriptor.attributes[AAPLVertexAttributeNormal].name    = MDLVertexAttributeNormal;

    NSURL *modelFileURL = [[NSBundle mainBundle] URLForResource:@"Models/livingroom.obj" withExtension:nil];

    NSAssert(modelFileURL, @"Could not find model (%@) file in bundle creating specular texture", modelFileURL.absoluteString);

    NSMutableArray< AAPLMesh* >* scene = [[NSMutableArray alloc] init];
    [scene addObjectsFromArray:[AAPLMesh newMeshesFromURL:modelFileURL
                                           modelIOVertexDescriptor:modelIOVertexDescriptor
                                                       metalDevice:_device
                                                             error:&error]];

    [scene addObject:[AAPLMesh newSphereWithRadius:8.0f onDevice:_device vertexDescriptor:modelIOVertexDescriptor]];
    
    [scene addObject:[AAPLMesh newPlaneWithDimensions:(vector_float2){80.0f, 80.0f} onDevice:_device vertexDescriptor:modelIOVertexDescriptor]];
    
    _meshes = scene;
    
    NSAssert(_meshes, @"Could not create meshes from model file %@: %@", modelFileURL.absoluteString, error);

    _skyMap = texture_from_radiance_file( @"kloppenheim_06_4k.hdr", _device, &error );
    NSAssert( _skyMap, @"Could not load sky texture: %@", error );

    MDLVertexDescriptor *skyboxModelIOVertexDescriptor =
        MTKModelIOVertexDescriptorFromMetal(_mtlSkyboxVertexDescriptor);
    skyboxModelIOVertexDescriptor.attributes[VertexAttributePosition].name = MDLVertexAttributePosition;
    skyboxModelIOVertexDescriptor.attributes[VertexAttributeTexcoord].name = MDLVertexAttributeTextureCoordinate;

    
    _skybox = [AAPLMesh newSkyboxMeshOnDevice:_device vertexDescriptor:skyboxModelIOVertexDescriptor];
    NSAssert( _skybox, @"Could not create skybox model" );
}

#pragma mark - Encode Argument Buffers

/// A convenience method to create `MTLArgumentDescriptor` objects for read-only access.
- (MTLArgumentDescriptor *)argumentDescriptorWithIndex:(NSUInteger)index dataType:(MTLDataType)dataType
{
    MTLArgumentDescriptor* argumentDescriptor = [MTLArgumentDescriptor argumentDescriptor];
    argumentDescriptor.index = index;
    argumentDescriptor.dataType = dataType;
    argumentDescriptor.access = MTLBindingAccessReadOnly;
    return argumentDescriptor;
}

/// Bindless核心，将场景压入一个Argument Buffer，这样在raytrace的cs中，可以访问整个场景
/// 原始demo是用raytrace的返回来做反射的完整光照返回
/// 改进一下，可以多trace几根光线，来做GI等更高级的效果
/// 此为METAL3前的build函数
/// Build an argument buffer with all the  resources for the scene.   The ray-tracing shaders access meshes, submeshes, and materials
/// through this argument buffer to apply the correct lighting to the calculated reflections.

- (id<MTLBuffer>)newBufferWithLabel:(NSString *)label length:(NSUInteger)length options:(MTLResourceOptions)options trackedIn:(nonnull NSMutableSet<id<MTLResource>> *)set
{
    id< MTLBuffer > buffer = [_device newBufferWithLength:length options:options];
    buffer.label = label;
    [set addObject:buffer];
    return buffer;
}

/// METAL3的build函数
///
/// Build an argument buffer with all resources for the scene.   The ray-tracing shaders access meshes, submeshes,
/// and materials through this argument buffer to apply the correct lighting to the calculated reflections.
- (void)buildSceneArgumentBufferMetal3 NS_AVAILABLE(13, 16)
{
    MTLResourceOptions storageMode;
#if TARGET_MACOS
    storageMode = MTLResourceStorageModeManaged;
#else
    storageMode = MTLResourceStorageModeShared;
#endif

    // The renderer builds this structure to match the ray-traced scene structure so the
    // ray-tracing shader navigates it. In particular, Metal represents each submesh as a
    // geometry in the primitive acceleration structure.

    NSMutableSet< id<MTLResource> >* sceneResources = [NSMutableSet new];

    NSUInteger instanceArgumentSize = sizeof( struct Instance ) * kMaxInstances;
    id<MTLBuffer> instanceArgumentBuffer = [self newBufferWithLabel:@"instanceArgumentBuffer"
                                                             length:instanceArgumentSize
                                                             options:storageMode
                                                           trackedIn:sceneResources];
    
    // Encode the instances array in `Scene` (`Scene::instances`).
    for ( NSUInteger i = 0; i < kMaxInstances; ++i )
    {
        struct Instance* pInstance = ((struct Instance *)instanceArgumentBuffer.contents) + i;
        pInstance->meshIndex = _modelInstances[i].meshIndex;
        pInstance->transform = calculateTransform(_modelInstances[i]);
    }
    
#if TARGET_MACOS
    [instanceArgumentBuffer didModifyRange:NSMakeRange(0, instanceArgumentBuffer.length)];
#endif

    NSUInteger meshArgumentSize = sizeof( struct Mesh ) * _meshes.count;
    id<MTLBuffer> meshArgumentBuffer = [self newBufferWithLabel:@"meshArgumentBuffer"
                                                         length:meshArgumentSize
                                                        options:storageMode
                                                      trackedIn:sceneResources];
    
    // Encode the meshes array in Scene (Scene::meshes).
    for ( NSUInteger i = 0; i < _meshes.count; ++i )
    {
        AAPLMesh* mesh = _meshes[i];
        
        struct Mesh* pMesh = ((struct Mesh *)meshArgumentBuffer.contents) + i;

        MTKMesh* metalKitMesh = mesh.metalKitMesh;

        // Set `Mesh::positions`.
        pMesh->positions = metalKitMesh.vertexBuffers[0].buffer.gpuAddress + metalKitMesh.vertexBuffers[0].offset;
        
        // Set `Mesh::generics`.
        pMesh->generics = metalKitMesh.vertexBuffers[1].buffer.gpuAddress + metalKitMesh.vertexBuffers[1].offset;

        NSAssert( metalKitMesh.vertexBuffers.count == 2, @"unknown number of buffers!" );
        [sceneResources addObject:metalKitMesh.vertexBuffers[0].buffer];
        [sceneResources addObject:metalKitMesh.vertexBuffers[1].buffer];
        
        // Build submeshes into a buffer and reference it through a pointer in the mesh.

        NSUInteger submeshArgumentSize = sizeof( struct Submesh ) * mesh.submeshes.count;
        id<MTLBuffer> submeshArgumentBuffer = [self newBufferWithLabel:[NSString stringWithFormat:@"submeshArgumentBuffer %lu", (unsigned long)i]
                                                                length:submeshArgumentSize
                                                                options:storageMode
                                                              trackedIn:sceneResources];
        
        for ( NSUInteger j = 0; j < mesh.submeshes.count; ++j )
        {
            AAPLSubmesh* submesh = mesh.submeshes[j];
            struct Submesh* pSubmesh = ((struct Submesh *)submeshArgumentBuffer.contents) + j;

            // Set `Submesh::indices`.
            MTKMeshBuffer* indexBuffer = submesh.metalKitSubmmesh.indexBuffer;
            pSubmesh->shortIndexType = submesh.metalKitSubmmesh.indexType == MTLIndexTypeUInt32 ? 0 : 1;
            pSubmesh->indices = indexBuffer.buffer.gpuAddress + indexBuffer.offset;

            // material parameters
            pSubmesh->baseColor = submesh.baseColor;
            pSubmesh->emissionColor = submesh.emissionColor;
            pSubmesh->roughness = submesh.roughness;
            pSubmesh->metallic = submesh.metallic;
            
            // material textures
            for (NSUInteger m = 0; m < submesh.textures.count; ++m)
            {
                pSubmesh->materials[m] = submesh.textures[m].gpuResourceID;
            }
            [sceneResources addObject:submesh.metalKitSubmmesh.indexBuffer.buffer];
            [sceneResources addObjectsFromArray:submesh.textures];

        }

#if TARGET_MACOS
        [submeshArgumentBuffer didModifyRange:NSMakeRange(0, submeshArgumentBuffer.length)];
#endif

        // Set `Mesh::submeshes`.
        pMesh->submeshes = submeshArgumentBuffer.gpuAddress;
    }

    [sceneResources addObject:meshArgumentBuffer];


    id<MTLBuffer> sceneArgumentBuffer = [self newBufferWithLabel:@"sceneArgumentBuffer"
                                                          length:sizeof( struct Scene )
                                                          options:storageMode
                                                        trackedIn:sceneResources];

    // Set `Scene::instances`.
    ((struct Scene *)sceneArgumentBuffer.contents)->instances = instanceArgumentBuffer.gpuAddress;
    
    // Set `Scene::meshes`.
    ((struct Scene *)sceneArgumentBuffer.contents)->meshes = meshArgumentBuffer.gpuAddress;


#if TARGET_MACOS
    [meshArgumentBuffer didModifyRange:NSMakeRange(0, meshArgumentBuffer.length)];
    [sceneArgumentBuffer didModifyRange:NSMakeRange(0, sceneArgumentBuffer.length)];
#endif

    _sceneResources = sceneResources;
    _sceneArgumentBuffer = sceneArgumentBuffer;
    
    
}

#pragma mark - Build Acceleration Structures

- (id<MTLAccelerationStructure>)allocateAndBuildAccelerationStructureWithDescriptor:(MTLAccelerationStructureDescriptor *)descriptor commandBuffer:(id<MTLCommandBuffer>)cmd
{
    MTLAccelerationStructureSizes sizes = [_device accelerationStructureSizesWithDescriptor:descriptor];
    id<MTLBuffer> scratch = [_device newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];
    id<MTLAccelerationStructure> accelStructure = [_device newAccelerationStructureWithSize:sizes.accelerationStructureSize];

    id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:accelStructure descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
    [enc endEncoding];

    return accelStructure;
}

/// Calculate the minimum size needed to allocate a heap that contains all acceleration structures for the passed-in descriptors.
/// The size is the sum of the needed sizes, and the scratch and refit buffer sizes are the maximum needed.
- (MTLAccelerationStructureSizes)calculateSizeForPrimitiveAccelerationStructures:(NSArray<MTLPrimitiveAccelerationStructureDescriptor*>*)primitiveAccelerationDescriptors NS_AVAILABLE(13,16)
{
    MTLAccelerationStructureSizes totalSizes = (MTLAccelerationStructureSizes){0, 0, 0};
    for ( MTLPrimitiveAccelerationStructureDescriptor* desc in primitiveAccelerationDescriptors )
    {
        MTLSizeAndAlign sizeAndAlign = [_device heapAccelerationStructureSizeAndAlignWithDescriptor:desc];
        MTLAccelerationStructureSizes sizes = [_device accelerationStructureSizesWithDescriptor:desc];
        totalSizes.accelerationStructureSize += (sizeAndAlign.size + sizeAndAlign.align);
        totalSizes.buildScratchBufferSize = MAX( sizes.buildScratchBufferSize, totalSizes.buildScratchBufferSize );
        totalSizes.refitScratchBufferSize = MAX( sizes.refitScratchBufferSize, totalSizes.refitScratchBufferSize);
    }
    return totalSizes;
}

- (NSArray<id<MTLAccelerationStructure>> *)allocateAndBuildAccelerationStructuresWithDescriptors:(NSArray<MTLAccelerationStructureDescriptor *>*)descriptors
                                                                                            heap:(id<MTLHeap>)heap
                                                                            maxScratchBufferSize:(size_t)maxScratchSize
                                                                                     signalEvent:(id<MTLEvent>)event NS_AVAILABLE(13,16)
{
    NSAssert( heap, @"Heap argument is required" );
    
    NSMutableArray< id<MTLAccelerationStructure> >* accelStructures = [NSMutableArray arrayWithCapacity:descriptors.count];
    
    id<MTLBuffer> scratch = [_device newBufferWithLength:maxScratchSize options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> cmd = [_commandQueue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];

    for ( MTLPrimitiveAccelerationStructureDescriptor* descriptor in descriptors )
    {
        MTLSizeAndAlign sizes = [_device heapAccelerationStructureSizeAndAlignWithDescriptor:descriptor];
        id<MTLAccelerationStructure> accelStructure = [heap newAccelerationStructureWithSize:sizes.size];
        [enc buildAccelerationStructure:accelStructure descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
        [accelStructures addObject:accelStructure];
    }
    
    [enc endEncoding];
    [cmd encodeSignalEvent:event value:kPrimitiveAccelerationStructureBuild];
    [cmd commit];

    return accelStructures;
}

/// Build the ray-tracing acceleration structures.
- (void)buildRTAccelerationStructures
{
    // Each mesh is an individual primitive acceleration structure, with each submesh being one
    // geometry within that acceleration structure.

    // Instance Acceleration Structure references n instances.
    // 1 Instance references 1 Primitive Acceleration Structure
    // 1 Primitive Acceleration Structure = 1 Mesh in _meshes
    // 1 Primitive Acceleration Structure -> n geometries == n submeshes

    NSMutableArray< MTLPrimitiveAccelerationStructureDescriptor* > *primitiveAccelerationDescriptors = [NSMutableArray arrayWithCapacity:_meshes.count];
    for ( AAPLMesh* mesh in _meshes )
    {
        NSMutableArray< MTLAccelerationStructureTriangleGeometryDescriptor* >* geometries = [NSMutableArray arrayWithCapacity:mesh.submeshes.count];
        for ( AAPLSubmesh* submesh in mesh.submeshes )
        {
            MTLAccelerationStructureTriangleGeometryDescriptor* g = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            g.vertexBuffer = mesh.metalKitMesh.vertexBuffers.firstObject.buffer;
            g.vertexBufferOffset = mesh.metalKitMesh.vertexBuffers.firstObject.offset;
            g.vertexStride = 12; // The buffer must be packed XYZ XYZ XYZ ...

            g.indexBuffer = submesh.metalKitSubmmesh.indexBuffer.buffer;
            g.indexBufferOffset = submesh.metalKitSubmmesh.indexBuffer.offset;
            g.indexType = submesh.metalKitSubmmesh.indexType;

            NSUInteger indexElementSize = (g.indexType == MTLIndexTypeUInt16) ? sizeof(uint16_t) : sizeof(uint32_t);
            g.triangleCount = submesh.metalKitSubmmesh.indexBuffer.length / indexElementSize / 3;
            [geometries addObject:g];
        }
        MTLPrimitiveAccelerationStructureDescriptor* primDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        primDesc.geometryDescriptors = geometries;
        [primitiveAccelerationDescriptors addObject:primDesc];
    }
    
    // Allocate all primitive acceleration structures.
    // On Metal 3, allocate directly from a MTLHeap.
    BOOL heapBasedAllocation = NO;
    
    char* opts = getenv("DISABLE_METAL3_FEATURES");
    if ( (opts == NULL) || (strstr(opts, "1") != opts) )
    {
        if ( @available( macOS 13, iOS 16, *) )
        {
            if ( [_device supportsFamily:MTLGPUFamilyMetal3] )
            {
                heapBasedAllocation = YES;
                MTLAccelerationStructureSizes storageSizes = [self calculateSizeForPrimitiveAccelerationStructures:primitiveAccelerationDescriptors];
                MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
                heapDesc.size = storageSizes.accelerationStructureSize;
                _accelerationStructureHeap = [_device newHeapWithDescriptor:heapDesc];
                _primitiveAccelerationStructures = [self allocateAndBuildAccelerationStructuresWithDescriptors:primitiveAccelerationDescriptors
                                                                                                          heap:_accelerationStructureHeap
                                                                                          maxScratchBufferSize:storageSizes.buildScratchBufferSize
                                                                                                   signalEvent:_accelerationStructureBuildEvent];
            }
        }
    }
    
    // Non-Metal 3 devices, allocate each acceleration structure individually.
    if ( !heapBasedAllocation )
    {
        NSMutableArray< id<MTLAccelerationStructure> >* primitiveAccelerationStructures = [NSMutableArray arrayWithCapacity:_meshes.count];
        id< MTLCommandBuffer > cmd = [_commandQueue commandBuffer];
        for ( MTLPrimitiveAccelerationStructureDescriptor* desc in primitiveAccelerationDescriptors )
        {
            [primitiveAccelerationStructures addObject:[self allocateAndBuildAccelerationStructureWithDescriptor:desc commandBuffer:(id<MTLCommandBuffer>)cmd]];
        }
        [cmd encodeSignalEvent:_accelerationStructureBuildEvent value:kPrimitiveAccelerationStructureBuild];
        [cmd commit];
        _primitiveAccelerationStructures = primitiveAccelerationStructures;
    }
    

    MTLInstanceAccelerationStructureDescriptor* instanceAccelStructureDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
    instanceAccelStructureDesc.instancedAccelerationStructures = _primitiveAccelerationStructures;

    instanceAccelStructureDesc.instanceCount = kMaxInstances;

    // Load instance data (two fire trucks + one sphere + floor):

    id<MTLBuffer> instanceDescriptorBuffer = [_device newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor) * kMaxInstances options:MTLResourceStorageModeShared];
    MTLAccelerationStructureInstanceDescriptor* instanceDescriptors = (MTLAccelerationStructureInstanceDescriptor *)instanceDescriptorBuffer.contents;
    for (NSUInteger i = 0; i < kMaxInstances; ++i)
    {
        instanceDescriptors[i].accelerationStructureIndex = _modelInstances[i].meshIndex;
        instanceDescriptors[i].intersectionFunctionTableOffset = 0;
        instanceDescriptors[i].mask = 0xFF;
        instanceDescriptors[i].options = MTLAccelerationStructureInstanceOptionNone;

        AAPLInstanceTransform* transforms = (AAPLInstanceTransform *)(((uint8_t *)_instanceTransformBuffer.contents) + i * kAlignedInstanceTransformsStructSize);
        instanceDescriptors[i].transformationMatrix = matrix4x4_drop_last_row( transforms->modelViewMatrix );
    }
    instanceAccelStructureDesc.instanceDescriptorBuffer = instanceDescriptorBuffer;

    id< MTLCommandBuffer > cmd = [_commandQueue commandBuffer];
    [cmd encodeWaitForEvent:_accelerationStructureBuildEvent value:kPrimitiveAccelerationStructureBuild];
    _instanceAccelerationStructure = [self allocateAndBuildAccelerationStructureWithDescriptor:instanceAccelStructureDesc commandBuffer:cmd];
    [cmd encodeSignalEvent:_accelerationStructureBuildEvent value:kInstanceAccelerationStructureBuild];
    [cmd commit];
}

#pragma mark - Update State

matrix_float4x4 calculateTransform( ModelInstance instance )
{
    vector_float3 rotationAxis = {0, 1, 0};
    matrix_float4x4 rotationMatrix = matrix4x4_rotation( instance.rotationRad, rotationAxis );
    matrix_float4x4 translationMatrix = matrix4x4_translation( instance.position );

    return matrix_multiply(translationMatrix, rotationMatrix);
}

- (void)setStaticState
{
    for (NSUInteger i = 0; i < kMaxInstances; ++i)
    {
        AAPLInstanceTransform* transforms = (AAPLInstanceTransform *)(((uint8_t *)_instanceTransformBuffer.contents) + (i * kAlignedInstanceTransformsStructSize));
        transforms->modelViewMatrix = calculateTransform( _modelInstances[i] );
    }

    [self updateCameraState];
}

- (void)updateCameraState
{
    // cornellbox
//    _projectionMatrix = matrix_perspective_right_hand(30.0f * (M_PI / 180.0f), (_size.width / _size.height), 0.5f, 1000.0f);
//    vector_float3 camPos = (vector_float3){0,1.0,6.5};
//    vector_float3 camTarget = (vector_float3){0,1.0,6.5 - 1};
    
    // living room
    _projectionMatrix = matrix_perspective_right_hand(60.0f * (M_PI / 180.0f), (_size.width / _size.height), 0.5f, 1000.0f);
    vector_float3 camPos = (vector_float3){0,1.5,6.5};
    vector_float3 camTarget = (vector_float3){0,1.5,6.5 - 1};
    
    // refltest
//    _projectionMatrix = matrix_perspective_right_hand(60.0f * (M_PI / 180.0f), (_size.width / _size.height), 0.5f, 1000.0f);
//    vector_float3 camPos = (vector_float3){-45,5,9};
//    vector_float3 camTarget = (vector_float3){-45 + 20,4.9,0};
    
    
//    vector_float3 camPos = (vector_float3){2.4,0.8,4.1};
//    vector_float3 camTarget = (vector_float3){2.4 - 1,0.9,4.1 - 0.7};
    vector_float3 camUp = (vector_float3){0,1,0};
    
    AAPLLightData* pLightData = (AAPLLightData *)(_lightDataBuffer.contents);
    pLightData->directionalLightInvDirection = -vector_normalize((vector_float3){ cosf(_cameraAngle) * 6.0 , -3, sinf(_cameraAngle) * 6.0 });
    pLightData->lightIntensity = 50.0f;
    
    // Determine next safe slot:
    
    
    AAPLCameraData* pPrevCameraData = (AAPLCameraData *)_cameraDataBuffers[_cameraBufferIndex].contents;
    _cameraBufferIndex = ( _cameraBufferIndex + 1 ) % kMaxBuffersInFlight;
    
//    vector_float2 haltonSamples[] = {
//        vector2(0.5f, 0.333333333333f),
//        vector2(0.25f, 0.666666666667f),
//        vector2(0.75f, 0.111111111111f),
//        vector2(0.125f, 0.444444444444f),
//        vector2(0.625f, 0.777777777778f),
//        vector2(0.375f, 0.222222222222f),
//        vector2(0.875f, 0.555555555556f),
//        vector2(0.0625f, 0.888888888889f),
//        vector2(0.5625f, 0.037037037037f),
//        vector2(0.3125f, 0.37037037037f),
//        vector2(0.8125f, 0.703703703704f),
//        vector2(0.1875f, 0.148148148148f),
//        vector2(0.6875f, 0.481481481481f),
//        vector2(0.4375f, 0.814814814815f),
//        vector2(0.9375f, 0.259259259259f),
//        vector2(0.03125f, 0.592592592593f),
//    };

    // Update Projection Matrix
    simd_float2 jitter = vector2(0.0f, 0.0f);
    //simd_float2 jitter = (haltonSamples[_frameCount % 16] * 2.0f - 1.0f) / vector2((float)_size.width, (float)_size.height);
    AAPLCameraData* pCameraData = (AAPLCameraData *)_cameraDataBuffers[_cameraBufferIndex].contents;
    
    matrix_float4x4 projectionMatrix = _projectionMatrix;
    projectionMatrix.columns[2][0] += jitter.x;
    projectionMatrix.columns[2][1] += jitter.y;
    
    pCameraData->prevProjectionMatrix = pPrevCameraData->projectionMatrix;
    pCameraData->projectionMatrix = projectionMatrix;
    pCameraData->invProjectionMatrix = simd_inverse(projectionMatrix);
    pCameraData->prev_jitter = pPrevCameraData->jitter;
    pCameraData->jitter = jitter * vector2(0.5f, -0.5f);
    pCameraData->width = _size.width;
    pCameraData->height = _size.height;
    pCameraData->frameIndex = _frameCount;
    // Update Camera Position (and View Matrix):

    //vector_float3 camPos = (vector_float3){ cosf( _cameraAngle ) * 10.0f, 5, sinf(_cameraAngle) * 22.5f };

    _cameraAngle += (0.02 * _cameraPanSpeedFactor);
    if ( _cameraAngle >= 2 * M_PI )
    {
        _cameraAngle -= (2 * M_PI);
    }

    pCameraData->prevViewMatrix = pPrevCameraData->viewMatrix;
    pCameraData->viewMatrix = matrix_look_at_right_hand(camPos, camTarget, camUp);
    pCameraData->invViewMatrix = simd_inverse(pCameraData->viewMatrix);
    pCameraData->prevCameraPosition = pPrevCameraData->cameraPosition;
    pCameraData->cameraPosition = camPos;
    pCameraData->metallicBias = _metallicBias;
    pCameraData->roughnessBias = _roughnessBias;
}

#pragma mark - Rendering

- (void)encodeSceneRendering:(id<MTLRenderCommandEncoder >)renderEncoder
{
    // Flag the residency of indirect resources in the scene.
    for ( id<MTLResource> res in _sceneResources)
    {
        [renderEncoder useResource:res usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    }
    
    //for (AAPLMesh *mesh in _meshes)
    for ( NSUInteger i = 0; i < kMaxInstances; ++i )
    {
        AAPLMesh* mesh = _meshes[ _modelInstances[ i ].meshIndex ];
        MTKMesh *metalKitMesh = mesh.metalKitMesh;

        // Set the mesh's vertex buffers.
        for (NSUInteger bufferIndex = 0; bufferIndex < metalKitMesh.vertexBuffers.count; bufferIndex++)
        {
            MTKMeshBuffer *vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex];
            if((NSNull *)vertexBuffer != [NSNull null])
            {
                [renderEncoder setVertexBuffer:vertexBuffer.buffer
                                        offset:vertexBuffer.offset
                                       atIndex:bufferIndex];
            }
        }

        // Draw each submesh of the mesh.
        for ( NSUInteger submeshIndex = 0; submeshIndex < mesh.submeshes.count; ++submeshIndex )
        {
            AAPLSubmesh* submesh = mesh.submeshes[ submeshIndex ];
            
            // Access textures directly from the argument buffer and avoid rebinding them individually.
            // `SubmeshKeypath` provides the path to the argument buffer containing the texture data
            // for this submesh. The shader navigates the scene argument buffer using this key
            // to find the textures.
            AAPLSubmeshKeypath submeshKeypath = {
                .instanceID = (uint32_t)i,
                .submeshID = (uint32_t)submeshIndex
            };
            
            MTKSubmesh *metalKitSubmesh = submesh.metalKitSubmmesh;
            
            [renderEncoder setVertexBuffer:_instanceTransformBuffer
                                    offset:kAlignedInstanceTransformsStructSize * i
                                   atIndex:BufferIndexInstanceTransforms];
            
            [renderEncoder setVertexBuffer:_cameraDataBuffers[_cameraBufferIndex] offset:0 atIndex:BufferIndexCameraData];
            [renderEncoder setFragmentBuffer:_cameraDataBuffers[_cameraBufferIndex] offset:0 atIndex:BufferIndexCameraData];
            [renderEncoder setFragmentBuffer:_lightDataBuffer offset:0 atIndex:BufferIndexLightData];
            
            // Bind the scene and provide the keypath to retrieve this submesh's data.
            [renderEncoder setFragmentBuffer:_sceneArgumentBuffer offset:0 atIndex:SceneIndex];
            [renderEncoder setFragmentBytes:&submeshKeypath length:sizeof(AAPLSubmeshKeypath) atIndex:BufferIndexSubmeshKeypath];
            
            [renderEncoder drawIndexedPrimitives:metalKitSubmesh.primitiveType
                                      indexCount:metalKitSubmesh.indexCount
                                       indexType:metalKitSubmesh.indexType
                                     indexBuffer:metalKitSubmesh.indexBuffer.buffer
                               indexBufferOffset:metalKitSubmesh.indexBuffer.offset];

        }

    }

}

- (void)copyDepthStencilConfigurationFrom:(MTLRenderPassDescriptor *)src to:(MTLRenderPassDescriptor *)dest
{
    dest.depthAttachment.loadAction     = src.depthAttachment.loadAction;
    dest.depthAttachment.clearDepth     = src.depthAttachment.clearDepth;
    dest.depthAttachment.texture        = src.depthAttachment.texture;
    dest.stencilAttachment.loadAction   = src.stencilAttachment.loadAction;
    dest.stencilAttachment.clearStencil = src.stencilAttachment.clearStencil;
    dest.stencilAttachment.texture      = src.stencilAttachment.texture;
}

- (void)generateGaussMipmapsForTexture:(id<MTLTexture>)texture commandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    MPSImageGaussianBlur* gauss = [[MPSImageGaussianBlur alloc] initWithDevice:_device
                                                                         sigma:5.0f];
    MTLTextureDescriptor* tmpDesc = [[MTLTextureDescriptor alloc] init];
    tmpDesc.textureType = MTLTextureType2D;
    tmpDesc.pixelFormat = MTLPixelFormatRG11B10Float;
    tmpDesc.mipmapLevelCount = 1;
    tmpDesc.usage = MTLResourceUsageRead | MTLResourceUsageWrite;
    tmpDesc.resourceOptions = MTLResourceStorageModePrivate;
    
    id< MTLTexture > src = _rtReflectionMap;
    
    uint32_t newW = (uint32_t)_rtReflectionMap.width;
    uint32_t newH = (uint32_t)_rtReflectionMap.height;
    
    id< MTLEvent > event = [_device newEvent];
    uint64_t count = 0u;
    [commandBuffer encodeSignalEvent:event value:count];
    
    while ( count+1 < _rtReflectionMap.mipmapLevelCount )
    {
        [commandBuffer pushDebugGroup:[NSString stringWithFormat:@"Mip level: %llu", count]];
        
        tmpDesc.width = newW;
        tmpDesc.height = newH;
        
        id< MTLTexture > dst = [_rtMipmappingHeap newTextureWithDescriptor:tmpDesc];
        
        
        [gauss encodeToCommandBuffer:commandBuffer
                       sourceTexture:src
                  destinationTexture:dst];
        
        ++count;
        [commandBuffer encodeSignalEvent:event value:count];
        
        [commandBuffer encodeWaitForEvent:event value:count];
        id<MTLTexture> targetMip = [_rtReflectionMap newTextureViewWithPixelFormat:MTLPixelFormatRG11B10Float
                                                                       textureType:MTLTextureType2D
                                                                            levels:NSMakeRange(count, 1)
                                                                            slices:NSMakeRange(0, 1)];
        
        MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
        rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].texture = targetMip;
        
        id< MTLRenderCommandEncoder > blit = [commandBuffer renderCommandEncoderWithDescriptor:rpd];
        [blit setCullMode:MTLCullModeNone];
        [blit setRenderPipelineState:_rtMipmapPipeline];
        [blit setFragmentTexture:dst atIndex:0];
        [blit drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        [blit endEncoding];
        
        src = targetMip;
        
        newW = newW / 2;
        newH = newH / 2;
        
        [commandBuffer popDebugGroup];
    }
}

- (void)executeCSProcess:(id<MTLCommandBuffer>)commandBuffer inPSO:(id<MTLComputePipelineState>)inPSO outTexture:(id<MTLTexture>)outTexture label:(NSString*)label {
    id<MTLComputeCommandEncoder> compEnc = [commandBuffer computeCommandEncoder];
    compEnc.label = label;
    [compEnc setTexture:outTexture atIndex:OutImageIndex];
    [compEnc setTexture:_thinGBuffer.positionTexture atIndex:ThinGBufferPositionIndex];
    [compEnc setTexture:_thinGBuffer.depthNormalTexture atIndex:ThinGBufferDirectionIndex];
    [compEnc setTexture:_rtIrradianceMap atIndex:IrradianceMapIndex];
    [compEnc setTexture:_rtReflectionMap atIndex:RefectionMapIndex];
    [compEnc setTexture:_skyMap atIndex:AAPLSkyDomeTexture];
    
    // Bind the root of the argument buffer for the scene.
    [compEnc setBuffer:_sceneArgumentBuffer offset:0 atIndex:SceneIndex];
    
    // Bind the prebuilt acceleration structure.
    [compEnc setAccelerationStructure:_instanceAccelerationStructure atBufferIndex:AccelerationStructureIndex];
    
    [compEnc setBuffer:_instanceTransformBuffer offset:0 atIndex:BufferIndexInstanceTransforms];
    [compEnc setBuffer:_cameraDataBuffers[_cameraBufferIndex] offset:0 atIndex:BufferIndexCameraData];
    [compEnc setBuffer:_lightDataBuffer offset:0 atIndex:BufferIndexLightData];
    
    // Set the ray tracing reflection kernel.
    [compEnc setComputePipelineState:inPSO];
    
    // Flag residency for indirectly referenced resources.
    // These are:
    // 1. All primitive acceleration structures.
    // 2. Buffers and textures referenced through argument buffers.
    
    if ( _accelerationStructureHeap )
    {
        // Heap backs the acceleration structures. Mark the entire heap resident.
        [compEnc useHeap:_accelerationStructureHeap];
    }
    else
    {
        // Acceleration structures are independent. Mark each one resident.
        for ( id<MTLAccelerationStructure> primAccelStructure in _primitiveAccelerationStructures )
        {
            [compEnc useResource:primAccelStructure usage:MTLResourceUsageRead];
        }
    }
    
    for ( id<MTLResource> resource in _sceneResources )
    {
        [compEnc useResource:resource usage:MTLResourceUsageRead];
    }
    
    // Determine the dispatch grid size and dispatch compute.
    
    NSUInteger w = inPSO.threadExecutionWidth;
    NSUInteger h = inPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake( w, h, 1 );
    MTLSize threadsPerGrid = MTLSizeMake(outTexture.width, outTexture.height, 1);
    
    [compEnc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    
    [compEnc endEncoding];
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    // Per-frame updates here.

    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"Render Commands";

    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(block_sema);
     }];

    [self updateCameraState];

    /// Delay getting the currentRenderPassDescriptor until the renderer absolutely needs it to avoid
    ///   holding onto the drawable and blocking the display pipeline any longer than necessary.
    MTLRenderPassDescriptor* renderPassDescriptor = view.currentRenderPassDescriptor;

    if(renderPassDescriptor != nil)
    {
        if( _frameCount == 0 )
        {
            [_denoiser clearTemporalHistory];
        }
        // When ray tracing is in an enabled state, first render a thin G-Buffer
        // that contains position and reflection direction data. Then, dispatch a
        // compute kernel that ray traces mirror-like reflections from this data.
        id <MTLTexture> denoisedTexture;
        id <MTLTexture> denoisedIrr;
        id <MTLTexture> denoisedRefl;
        
        if ( _renderMode > 0  )
        {
            /// Step1. 全局Thin GBuffer，给CS使用
            ///
            
            id<MTLTexture> depthNormalTexture = [_textureAllocator textureWithPixelFormat:MTLPixelFormatRGBA16Float width:_size.width height:_size.height];
            
            MTLRenderPassDescriptor* gbufferPass = [MTLRenderPassDescriptor new];
            gbufferPass.colorAttachments[0].loadAction = MTLLoadActionClear;
            gbufferPass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            gbufferPass.colorAttachments[0].storeAction = MTLStoreActionStore;
            gbufferPass.colorAttachments[0].texture = _thinGBuffer.positionTexture;

            gbufferPass.colorAttachments[1].loadAction = MTLLoadActionClear;
            gbufferPass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
            gbufferPass.colorAttachments[1].storeAction = MTLStoreActionStore;
            gbufferPass.colorAttachments[1].texture = depthNormalTexture;
            
            gbufferPass.colorAttachments[2].loadAction = MTLLoadActionClear;
            gbufferPass.colorAttachments[2].clearColor = MTLClearColorMake(0, 0, 0, 0);
            gbufferPass.colorAttachments[2].storeAction = MTLStoreActionStore;
            gbufferPass.colorAttachments[2].texture = _thinGBuffer.motionVectorTexture;
            
            gbufferPass.colorAttachments[3].loadAction = MTLLoadActionClear;
            gbufferPass.colorAttachments[3].clearColor = MTLClearColorMake(0, 0, 0, 0);
            gbufferPass.colorAttachments[3].storeAction = MTLStoreActionStore;
            gbufferPass.colorAttachments[3].texture = _thinGBuffer.albedoTexture;
            
            // swap
            _thinGBuffer.PrevDepthNormalTexture = _thinGBuffer.depthNormalTexture;
            _thinGBuffer.depthNormalTexture = depthNormalTexture;
            

            [self copyDepthStencilConfigurationFrom:renderPassDescriptor to:gbufferPass];
            gbufferPass.depthAttachment.storeAction = MTLStoreActionStore;

            // Create a render command encoder.
            [commandBuffer pushDebugGroup:@"全局GBuffer"];
            id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:gbufferPass];

            renderEncoder.label = @"ThinGBufferRenderEncoder";

            // Set the render command encoder state.
            [renderEncoder setCullMode:MTLCullModeFront];
            [renderEncoder setFrontFacingWinding:MTLWindingClockwise];
            [renderEncoder setRenderPipelineState:_gbufferPipelineState];
            [renderEncoder setDepthStencilState:_depthState];

            // Encode all draw calls for the scene.
            [self encodeSceneRendering:renderEncoder];

            // Finish encoding commands.
            [renderEncoder endEncoding];
            [commandBuffer popDebugGroup];

            
            
            
            /// Step2. CS阶段，使用Thin GBuffer作为源头，利用CS进行全局的光照计算（间接光照）
            /// DEMO内是计算精确反射，我们还可以用它来作GI等
            /// 比如，我们可以trace多个位置，然后accumulate一个真正的反弹出来
            
            // The ray-traced reflections.
            [commandBuffer pushDebugGroup:@"CS处理"];
            [commandBuffer encodeWaitForEvent:_accelerationStructureBuildEvent value:kInstanceAccelerationStructureBuild];
            
            // shading
            if( _renderMode == RMMetalRaytracing || _renderMode == RMMetalRaytracing2 )
            {
                [self executeCSProcess:commandBuffer inPSO:_rtShadingPipeline outTexture:_rtShadingMap label:@"光追一次反弹"];
                [self executeCSProcess:commandBuffer inPSO:_rtBouncePipeline outTexture:_rtBounceMap label:@"光追二次反弹"];
            }
            else if (_renderMode == RMReflectionsOnly )
            {
                [self executeCSProcess:commandBuffer inPSO:_rtGroundTruthPipeline outTexture:_rtGroundTruthMap label:@"GT光追"];
            }

            
            [commandBuffer popDebugGroup];
            
            [commandBuffer pushDebugGroup:@"MPS降噪"];
            

            
            if( _renderMode == RMMetalRaytracing2 )
            {
                denoisedTexture = [_denoiser encodeToCommandBuffer:commandBuffer
                                                                     sourceTexture:_rtShadingMap
                                                               motionVectorTexture:_thinGBuffer.motionVectorTexture
                                                                depthNormalTexture:_thinGBuffer.depthNormalTexture
                                                        previousDepthNormalTexture:_thinGBuffer.PrevDepthNormalTexture];
                
                denoisedIrr = [_denoiserIrr encodeToCommandBuffer:commandBuffer
                                                                     sourceTexture:_rtBounceMap
                                                               motionVectorTexture:_thinGBuffer.motionVectorTexture
                                                                depthNormalTexture:_thinGBuffer.depthNormalTexture
                                                        previousDepthNormalTexture:_thinGBuffer.PrevDepthNormalTexture];
                
                denoisedRefl = [_denoiserRefl encodeToCommandBuffer:commandBuffer
                                                                     sourceTexture:_rtReflectionMap
                                                               motionVectorTexture:_thinGBuffer.motionVectorTexture
                                                                depthNormalTexture:_thinGBuffer.depthNormalTexture
                                                        previousDepthNormalTexture:_thinGBuffer.PrevDepthNormalTexture];
            }
            else if(_renderMode == RMMetalRaytracing )
            {
                denoisedTexture = [_denoiser encodeToCommandBuffer:commandBuffer
                                                                     sourceTexture:_rtShadingMap
                                                               motionVectorTexture:_thinGBuffer.motionVectorTexture
                                                                depthNormalTexture:_thinGBuffer.depthNormalTexture
                                                        previousDepthNormalTexture:_thinGBuffer.PrevDepthNormalTexture];
                
                denoisedIrr = [_denoiserIrr encodeToCommandBuffer:commandBuffer
                                                                     sourceTexture:_rtIrradianceMap
                                                               motionVectorTexture:_thinGBuffer.motionVectorTexture
                                                                depthNormalTexture:_thinGBuffer.depthNormalTexture
                                                        previousDepthNormalTexture:_thinGBuffer.PrevDepthNormalTexture];

            }
            else if(_renderMode == RMReflectionsOnly )
            {
                denoisedIrr = [_denoiserIrr encodeToCommandBuffer:commandBuffer
                                                                     sourceTexture:_rtGroundTruthMap
                                                               motionVectorTexture:_thinGBuffer.motionVectorTexture
                                                                depthNormalTexture:_thinGBuffer.depthNormalTexture
                                                        previousDepthNormalTexture:_thinGBuffer.PrevDepthNormalTexture];

            }

            

            
            [commandBuffer popDebugGroup];
            
        }

        /// Step3. 常规的渲染pass
        /// 如果是传统渲染，这是第一步，就是简单的前向渲染
        /// 不过如果已经有了ThinGBuffer，这里其实可以直接再开一个CS，来作真正的DeferredShading
        
        // Encode the forward pass.
        
        id <MTLTexture> compositeTexture = [_textureAllocator textureWithPixelFormat:MTLPixelFormatRG11B10Float width:_size.width height:_size.height];
        
        MTLRenderPassDescriptor* rpd = view.currentRenderPassDescriptor;
        id<MTLTexture> drawableTexture = rpd.colorAttachments[0].texture;
        rpd.colorAttachments[0].texture = compositeTexture;
        
        [commandBuffer pushDebugGroup:@"标准渲染"];
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:rpd];
        renderEncoder.label = @"ForwardPassRenderEncoder";

        if ( _renderMode == RMMetalRaytracing || _renderMode == RMMetalRaytracing2 )
        {
            [renderEncoder setRenderPipelineState:_pipelineState];
        }
        else if ( _renderMode == RMNoRaytracing )
        {
            [renderEncoder setRenderPipelineState:_pipelineStateNoRT];
        }
        else if ( _renderMode == RMReflectionsOnly )
        {
            [renderEncoder setRenderPipelineState:_pipelineStateReflOnly];
        }

        [renderEncoder setCullMode:MTLCullModeFront];
        [renderEncoder setFrontFacingWinding:MTLWindingClockwise];
        [renderEncoder setDepthStencilState:_depthState];
        
        if ( _renderMode == RMMetalRaytracing || _renderMode == RMMetalRaytracing2 )
        {
            [renderEncoder setFragmentTexture:denoisedRefl atIndex:AAPLTextureIndexReflections];
            [renderEncoder setFragmentTexture:denoisedTexture atIndex:AAPLTextureIndexGI];
            [renderEncoder setFragmentTexture:denoisedIrr atIndex:AAPLTextureIndexIrrGI];
        }
        if( _renderMode == RMReflectionsOnly )
        {
            [renderEncoder setFragmentTexture:denoisedIrr atIndex:AAPLTextureIndexIrrGI];
            [renderEncoder setFragmentTexture:_thinGBuffer.albedoTexture atIndex:AAPLTextureIndexGBufferAlbedo];
        }
        
        [renderEncoder setFragmentTexture:_skyMap atIndex:AAPLSkyDomeTexture];
        
        [self encodeSceneRendering:renderEncoder];
        
        if ( _renderMode == RMMetalRaytracing || _renderMode == RMMetalRaytracing2 )
        {
            [_textureAllocator returnTexture:denoisedTexture];
            [_textureAllocator returnTexture:denoisedIrr];
            [_textureAllocator returnTexture:_thinGBuffer.PrevDepthNormalTexture];
        }
        if( _renderMode == RMReflectionsOnly )
        {
            [_textureAllocator returnTexture:denoisedIrr];
        }
        
        /// Step4. 传统的背景渲染
        // Encode the skybox rendering.
        if(true)
        {
            [renderEncoder setCullMode:MTLCullModeBack];
            [renderEncoder setRenderPipelineState:_skyboxPipelineState];
            
            [renderEncoder setVertexBuffer:_cameraDataBuffers[_cameraBufferIndex]
                                    offset:0
                                   atIndex:BufferIndexCameraData];
            
            [renderEncoder setFragmentTexture:_skyMap atIndex:0];
            
            MTKMesh* metalKitMesh = _skybox.metalKitMesh;
            for (NSUInteger bufferIndex = 0; bufferIndex < metalKitMesh.vertexBuffers.count; bufferIndex++)
            {
                MTKMeshBuffer *vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex];
                if((NSNull *)vertexBuffer != [NSNull null])
                {
                    [renderEncoder setVertexBuffer:vertexBuffer.buffer
                                            offset:vertexBuffer.offset
                                           atIndex:bufferIndex];
                }
            }
            
            for(MTKSubmesh *submesh in metalKitMesh.submeshes)
            {
                [renderEncoder drawIndexedPrimitives:submesh.primitiveType
                                          indexCount:submesh.indexCount
                                           indexType:submesh.indexType
                                         indexBuffer:submesh.indexBuffer.buffer
                                   indexBufferOffset:submesh.indexBuffer.offset];
            }
        }

        
        [renderEncoder endEncoding];
        [commandBuffer popDebugGroup];
        
        id <MTLTexture> AATexture = compositeTexture;
        if(_rawColorMap)
            [_textureAllocator returnTexture:_rawColorMap];
        _rawColorMap = AATexture;
 
        // Merge the postprocessing results with the scene rendering.
        {
            [commandBuffer pushDebugGroup:@"合成阶段"];
            MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
            rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            rpd.colorAttachments[0].texture = drawableTexture;
            
            id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:rpd];
            [renderEncoder pushDebugGroup:@"后处理曝光"];
            [renderEncoder setRenderPipelineState:_postMergePipeline];
            [renderEncoder setFragmentBytes:&_exposure length:sizeof(float) atIndex:0];
//            if(_renderMode == RMReflectionsOnly)
//            {
//                [renderEncoder setFragmentTexture:_rtGroundTruthMap atIndex:0];
//            }
//            else
            {
                [renderEncoder setFragmentTexture:_rawColorMap atIndex:0];
            }
            
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                              vertexStart:0
                              vertexCount:6];
            
            [renderEncoder popDebugGroup];
            [renderEncoder endEncoding];
            [commandBuffer popDebugGroup];
        }
        
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    [commandBuffer commit];
    
    _frameCount++;
}

- (matrix_float4x4)projectionMatrixWithAspect:(float)aspect
{
    return matrix_perspective_right_hand(60.0f * (M_PI / 180.0f), aspect, 0.5f, 1000.0f);
}

#pragma mark - Event Handling

/// Respond to drawable size or orientation changes.
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    [_denoiser releaseTemporaryTextures];
    [_denoiserIrr releaseTemporaryTextures];
    [_denoiserRefl releaseTemporaryTextures];
    [_textureAllocator reset];
    
    float aspect = size.width / (float)size.height;
    _projectionMatrix = [self projectionMatrixWithAspect:aspect];

    // The passed-in size is already in backing coordinates.
    [self resizeRTReflectionMapTo:CGSizeMake(size.width, size.height)];
    //[self resizeRTReflectionMapTo:size];
    
    _frameCount = 0;
}

- (void)setRenderMode:(RenderMode)renderMode
{
    _renderMode = renderMode;
}

- (void)setCameraPanSpeedFactor:(float)speedFactor
{
    _cameraPanSpeedFactor = speedFactor;
}

- (void)setMetallicBias:(float)metallicBias
{
    _metallicBias = metallicBias;
}

- (void)setRoughnessBias:(float)roughnessBias
{
    _roughnessBias = roughnessBias;
}

- (void)setExposure:(float)exposure
{
    _exposure = exposure;
}

@end
