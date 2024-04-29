/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The Metal shaders and kernels.
*/

#include <metal_stdlib>
#include <simd/simd.h>
#include "Loki/loki_header.metal"
// Include the header that this Metal shader code shares with the Swift/C code that executes Metal API commands.
#include "AAPLShaderTypes.h"

#include "AAPLArgumentBufferTypes.h"

using namespace metal;

constant float PI = 3.1415926535897932384626433832795;
constant float kMaxHDRValue = 500.0f;

typedef struct
{
    float4 position [[position]];
    float4 currPosition;
    float4 prevPosition;
    float3 viewPosition;
    float3 ndcpos;
    float3 worldPosition;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float3 r;
    float2 texCoord;
} ColorInOut;

#pragma mark - Lighting

struct LightingParameters
{
    float3  lightDir;
    float3  viewDir;
    float3  halfVector;
    float3  reflectedVector;
    float3  normal;
    float3  reflectedColor;
    float3  irradiatedColor;
    float4  baseColor;
    float   nDoth;
    float   nDotv;
    float   nDotl;
    float   hDotl;
    float   metalness;
    float   roughness;
    float   ambientOcclusion;
};

constexpr sampler linearSampler (address::repeat,
                                 mip_filter::linear,
                                 mag_filter::linear,
                                 min_filter::linear);

constexpr sampler nearestSampler(address::repeat,
                                 min_filter::nearest,
                                 mag_filter::nearest,
                                 mip_filter::none);

inline float Fresnel(float dotProduct);
inline float sqr(float a);
float3 computeSpecular(LightingParameters parameters);
float Geometry(float Ndotv, float alphaG);
float3 computeNormalMap(ColorInOut in, texture2d<float> normalMapTexture);
float3 computeDiffuse(LightingParameters parameters);
float Distribution(float NdotH, float roughness);

inline float Fresnel(float dotProduct) {
    return pow(clamp(1.0 - dotProduct, 0.0, 1.0), 5.0);
}

inline float sqr(float a) {
    return a * a;
}

float Geometry(float Ndotv, float alphaG) {
    float a = alphaG * alphaG;
    float b = Ndotv * Ndotv;
    return (float)(1.0 / (Ndotv + sqrt(a + b - a*b)));
}

float3 computeNormalMap(ColorInOut in, texture2d<float> normalMapTexture) {
    float4 encodedNormal = normalMapTexture.sample(nearestSampler, float2(in.texCoord));
    float4 normalMap = float4(normalize(encodedNormal.xyz * 2.0 - float3(1,1,1)), 0.0);
    return float3(normalize(in.normal * normalMap.z + in.tangent * normalMap.x + in.bitangent * normalMap.y));
}

float3 computeDiffuse(LightingParameters parameters)
{
    float3 diffuseRawValue = float3(((1.0/PI) * parameters.baseColor) * (1.0 - parameters.metalness));
    return diffuseRawValue * (parameters.nDotl);
}

float Distribution(float NdotH, float roughness)
{
    if (roughness >= 1.0)
        return 1.0 / PI;

    float roughnessSqr = saturate( roughness * roughness );

    float d = (NdotH * roughnessSqr - NdotH) * NdotH + 1;
    return roughnessSqr / (PI * d * d);
}

float3 computeSpecular(LightingParameters parameters)
{
    float specularRoughness = saturate( parameters.roughness * (1.0 - parameters.metalness) + parameters.metalness );

    float Ds = Distribution(parameters.nDoth, specularRoughness);

    float3 Cspec0 = parameters.baseColor.rgb;
    float3 Fs = float3(mix(float3(Cspec0), float3(1), Fresnel(parameters.hDotl)));
    float alphaG = sqr(specularRoughness * 0.5 + 0.5);
    float Gs = Geometry(parameters.nDotl, alphaG) * Geometry(parameters.nDotv, alphaG);

    float3 specularOutput = (Ds * Gs * Fs * parameters.irradiatedColor) * (1.0 + parameters.metalness * float3(parameters.baseColor))
    + float3(parameters.metalness) * parameters.irradiatedColor * float3(parameters.baseColor);

    return specularOutput * parameters.ambientOcclusion;
}

// The helper for the equirectangular textures.
float4 equirectangularSample(float3 direction, sampler s, texture2d<float> image)
{
    float3 d = normalize(direction);

    float2 t = float2((atan2(d.z, d.x) + M_PI_F) / (2.f * M_PI_F), acos(d.y) / M_PI_F);

    return image.sample(s, t);
}

LightingParameters calculateParameters(ColorInOut in,
                                       AAPLCameraData cameraData,
                                       constant AAPLLightData& lightData,
                                       texture2d<float>   baseColorMap,
                                       texture2d<float>   normalMap,
                                       texture2d<float>   metallicMap,
                                       texture2d<float>   roughnessMap,
                                       texture2d<float>   ambientOcclusionMap,
                                       texture2d<float>   skydomeMap)
{
    LightingParameters parameters;

    parameters.baseColor = baseColorMap.sample(linearSampler, in.texCoord.xy);

    parameters.normal = computeNormalMap(in, normalMap);

    parameters.viewDir = normalize(cameraData.cameraPosition - float3(in.worldPosition));

    parameters.roughness = mix(0.01,1.0,roughnessMap.sample(linearSampler, in.texCoord.xy).x);

    parameters.metalness = max(metallicMap.sample(linearSampler, in.texCoord.xy).x, 0.1);

    parameters.ambientOcclusion = 1.0;//ambientOcclusionMap.sample(linearSampler, in.texCoord.xy).x;

    parameters.reflectedVector = reflect(-parameters.viewDir, parameters.normal);
    
    constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 c = equirectangularSample(parameters.reflectedVector, linearFilterSampler, skydomeMap).rgb;
    parameters.irradiatedColor = clamp(c, 0.f, kMaxHDRValue);

    parameters.lightDir = lightData.directionalLightInvDirection;
    parameters.nDotl = max(0.001f,saturate(dot(parameters.normal, parameters.lightDir)));

    parameters.halfVector = normalize(parameters.lightDir + parameters.viewDir);
    parameters.nDoth = max(0.001f,saturate(dot(parameters.normal, parameters.halfVector)));
    parameters.nDotv = max(0.001f,saturate(dot(parameters.normal, parameters.viewDir)));
    parameters.hDotl = max(0.001f,saturate(dot(parameters.lightDir, parameters.halfVector)));

    return parameters;
}

#pragma mark - Skybox

struct SkyboxVertex
{
    float3 position [[ attribute(AAPLVertexAttributePosition) ]];
    float2 texcoord [[ attribute(AAPLVertexAttributeTexcoord)]];
};

struct SkyboxV2F
{
    float4 position [[position]];
    float4 cameraToPointV;
    float2 texcoord;
    float y;
};

vertex SkyboxV2F skyboxVertex(SkyboxVertex in [[stage_in]],
                                 constant AAPLCameraData& cameraData [[buffer(BufferIndexCameraData)]])
{
    SkyboxV2F v;
    v.cameraToPointV = cameraData.viewMatrix * float4( in.position, 1.0f );
    v.position = cameraData.projectionMatrix * v.cameraToPointV;
    v.texcoord = in.texcoord;
    v.y = v.cameraToPointV.y / v.cameraToPointV.w;
    return v;
}

fragment float4 skyboxFragment(SkyboxV2F v [[stage_in]], texture2d<float> skytexture [[texture(0)]])
{
    constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 c = equirectangularSample(v.cameraToPointV.xyz/v.cameraToPointV.w, linearFilterSampler, skytexture).rgb;
    return float4(clamp(c, 0.f, kMaxHDRValue), 1.f);
}

#pragma mark - Rasterization

typedef struct
{
    float3 position  [[ attribute(AAPLVertexAttributePosition) ]];
    float2 texCoord  [[ attribute(AAPLVertexAttributeTexcoord) ]];
    float3 normal    [[ attribute(AAPLVertexAttributeNormal) ]];
    float3 tangent   [[ attribute(AAPLVertexAttributeTangent) ]];
    float3 bitangent [[ attribute(AAPLVertexAttributeBitangent) ]];
} Vertex;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant AAPLInstanceTransform& instanceTransform [[ buffer(BufferIndexInstanceTransforms) ]],
                               constant AAPLCameraData& cameraData [[ buffer(BufferIndexCameraData) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = cameraData.projectionMatrix * cameraData.viewMatrix * instanceTransform.modelViewMatrix * position;
    out.currPosition = out.position;
    out.prevPosition = cameraData.prevProjectionMatrix * cameraData.prevViewMatrix * instanceTransform.modelViewMatrix * position;
    out.viewPosition = (cameraData.viewMatrix * instanceTransform.modelViewMatrix * position).xyz;
    out.ndcpos = out.position.xyz/out.position.w;

    // Reflections and lighting that occur in the world space, so
    // `camera.viewMatrix` isn’t taken into consideration here.
    float4x4 objToWorld = instanceTransform.modelViewMatrix;
    out.worldPosition = (objToWorld * position).xyz;

    float3x3 normalMx = float3x3(objToWorld.columns[0].xyz,
                                 objToWorld.columns[1].xyz,
                                 objToWorld.columns[2].xyz);
    out.normal = normalMx * normalize(in.normal);
    out.tangent = normalMx * normalize(in.tangent);
    out.bitangent = normalMx * normalize(in.bitangent);

    float3 v = out.worldPosition - cameraData.cameraPosition;
    out.r = reflect( v, out.normal );

    out.texCoord = in.texCoord;

    return out;
}

float2 calculateScreenCoord( float3 ndcpos )
{
    //float2 screenTexcoord = (ndcpos.xy) * 0.5 + float2(0.5);
    //screenTexcoord.y = 1.0 - screenTexcoord.y;
    float2 screenTexcoord = (ndcpos.xy) * float2(0.5f, -0.5f) + float2(0.5f,0.5f);
    return screenTexcoord;
}

constant bool is_raytracing_enabled [[function_constant(AAPLConstantIndexRayTracingEnabled)]];

fragment float4 fragmentShader(
                    ColorInOut                  in                    [[stage_in]],
                    constant AAPLCameraData&    cameraData            [[ buffer(BufferIndexCameraData) ]],
                    constant AAPLLightData&     lightData             [[ buffer(BufferIndexLightData) ]],
                    constant AAPLSubmeshKeypath&submeshKeypath        [[ buffer(BufferIndexSubmeshKeypath)]],
                    constant Scene*             pScene                [[ buffer(SceneIndex)]],
                    texture2d<float>            skydomeMap            [[ texture(AAPLSkyDomeTexture) ]],
                    texture2d<float>            rtReflections         [[ texture(AAPLTextureIndexReflections), function_constant(is_raytracing_enabled)]],
                    texture2d<float>            rtShadings            [[ texture(AAPLTextureIndexGI), function_constant(is_raytracing_enabled)]],
                    texture2d<float>            rtIrrandiance         [[ texture(AAPLTextureIndexIrrGI), function_constant(is_raytracing_enabled)]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    float2 screenTexcoord = calculateScreenCoord( in.ndcpos );
    
    constant Mesh* pMesh = &(pScene->meshes[ pScene->instances[submeshKeypath.instanceID].meshIndex ]);
    constant Submesh* pSubmesh = &(pMesh->submeshes[submeshKeypath.submeshID]);

    LightingParameters params = calculateParameters(in,
                                                    cameraData,
                                                    lightData,
                                                    pSubmesh->materials[AAPLTextureIndexBaseColor],        //colorMap
                                                    pSubmesh->materials[AAPLTextureIndexNormal],           //normalMap
                                                    pSubmesh->materials[AAPLTextureIndexMetallic],         //metallicMap
                                                    pSubmesh->materials[AAPLTextureIndexRoughness],        //roughnessMap
                                                    pSubmesh->materials[AAPLTextureIndexAmbientOcclusion], //ambientOcclusionMap
                                                    skydomeMap);
    float3 skylight = params.ambientOcclusion * 0.1;
    float li = lightData.lightIntensity;
    params.roughness += cameraData.roughnessBias;
    clamp( params.roughness, 0.f, 1.0f );

    if ( is_raytracing_enabled )
    {
        uint8_t mipLevel = params.roughness * rtReflections.get_num_mip_levels();
        float3 reflectedColor = rtReflections.sample(colorSampler, screenTexcoord, level(mipLevel)).xyz;

        //float hasReflection = (dot( reflectedColor.rgb, float3(1,1,1) ) > 0.0);
        //params.irradiatedColor = mix(params.irradiatedColor, reflectedColor.rgb, hasReflection);
        
        
        float4 gi = rtShadings.sample(colorSampler, screenTexcoord, level(0)).xyzw;
        li *= gi.y;
        
        params.irradiatedColor = reflectedColor.rgb * reflectedColor.rgb * gi.x;
        
        float4 irr = rtIrrandiance.sample(colorSampler, screenTexcoord, level(0)).xyzw;
        skylight *= irr.xyz * 2.0;
        
    }
    params.metalness += cameraData.metallicBias;
    float4 final_color = float4(skylight * float3((params.baseColor) * (1.0 - params.metalness)) + computeSpecular(params) + li * computeDiffuse(params), 1.0f);
    //float4 final_color = float4((skylight + li * params.nDotl) * mix(params.baseColor.xyz, dot(params.baseColor.xyz, float3(.33)), 0.8), 1.0f);
    //float4 final_color = float4(skylight, 1.0f);
    return final_color;
}

fragment float4 reflectionShader(ColorInOut in [[stage_in]],
                                 texture2d<float> rtReflections [[texture(AAPLTextureIndexReflections)]])
{
    float2 screenTexcoord = calculateScreenCoord( in.ndcpos );
    float4 reflectedColor = rtReflections.sample(linearSampler, screenTexcoord, level(0));
    reflectedColor.a = 1.0;
    return reflectedColor;
}

struct ThinGBufferOut
{
    float4 position [[color(0)]];
    float4 direction [[color(1)]];
    float2 motionVector [[color(2)]];
};

fragment ThinGBufferOut gBufferFragmentShader(ColorInOut in [[stage_in]],
                                              constant AAPLCameraData&    cameraData            [[ buffer(BufferIndexCameraData) ]])
{
    ThinGBufferOut out;

    out.position = float4(in.worldPosition, 1.0);
  
    float2 motionVector = 0.0f;
    if (cameraData.frameIndex > 0) {
        //float2 uv = in.position.xy / float2(cameraData.width, cameraData.height);
        float2 uv = in.currPosition.xy / in.currPosition.w * float2(0.5f, -0.5f) + float2(0.5f,0.5f);
        
        // Unproject the position from the previous frame then transform it from
        // NDC space to 0..1
        float2 prevUV = in.prevPosition.xy / in.prevPosition.w * float2(0.5f, -0.5f) + float2(0.5f,0.5f);
        
        uv -= cameraData.jitter;
        prevUV -= cameraData.prev_jitter;
        
        motionVector = (uv - prevUV);
    }
    
    // Then the motion vector is simply the difference between the two
    out.direction = float4(length(in.viewPosition), in.normal);
    out.motionVector = motionVector;
    
    return out;
}

#if __METAL_VERSION__ >= 230

#pragma mark - Ray tracing
using raytracing::instance_acceleration_structure;

kernel void rtShading(
             texture2d< float, access::write >      outImage                [[texture(OutImageIndex)]],
             texture2d< float, access::write >      outIrradiance                [[texture(IrradianceMapIndex)]],
             texture2d< float >                     positions               [[texture(ThinGBufferPositionIndex)]],
             texture2d< float >                     directions              [[texture(ThinGBufferDirectionIndex)]],
             texture2d< float >                     skydomeMap              [[texture(AAPLSkyDomeTexture)]],
             constant AAPLInstanceTransform*        instanceTransforms      [[buffer(BufferIndexInstanceTransforms)]],
             constant AAPLCameraData&               cameraData              [[buffer(BufferIndexCameraData)]],
             constant AAPLLightData&                lightData               [[buffer(BufferIndexLightData)]],
             constant Scene*                        pScene                  [[buffer(SceneIndex)]],
             instance_acceleration_structure        accelerationStructure   [[buffer(AccelerationStructureIndex)]],
             uint2 tid [[thread_position_in_grid]])
{

    uint w = outImage.get_width();
    uint h = outImage.get_height();
    if ( tid.x < w&& tid.y < h )
    {
        float4 finalColor = float4( 0.0, 0.0, 0.0, 1.0 );
        float4 finalIrradiance = float4( 0.0, 0.0, 0.0, 1.0 );
        if (is_null_instance_acceleration_structure(accelerationStructure))
        {
            finalColor = float4( 1.0, 0.0, 1.0, 1.0 );
        }
        else
        {
            auto position = positions.read(tid).xyz;
            auto normal = directions.read(tid).yzw;
            Loki rng = Loki(tid.x + 1, tid.y + 1, cameraData.frameIndex);
            
            // 构造一个在normal半球内的ray
            uint skyRayCount = 4;
            float hit = 0.0;
            
            raytracing::intersector<raytracing::instancing, raytracing::triangle_data> inter;
            inter.assume_geometry_type( raytracing::geometry_type::triangle );
            
            for( uint i = 0; i < skyRayCount; ++i)
            {
                raytracing::ray r;

                // 这里需要构造一个基于法线的hemisphere来采样，并且引入重要性分布，使用hottonPattern
                r.origin = position;
                r.direction = normalize(float3(rng.rand() - 0.5,rng.rand() - 0.5,rng.rand() - 0.5));
                r.min_distance = 0.1;
                r.max_distance = FLT_MAX;
                
                // 在半球内发射射线
                auto intersection = inter.intersect( r, accelerationStructure, 0xFF );
                if ( intersection.type == raytracing::intersection_type::triangle )
                {
                    // 打到了, 取一次反弹
                    hit += 1.0;
                    
                    float2 bary2 = intersection.triangle_barycentric_coord;
                    float3 bary3 = float3( 1.0 - (bary2.x + bary2.y), bary2.x, bary2.y );

                    constant Instance& instance = pScene->instances[ intersection.instance_id ];
                    constant Mesh* pMesh = &(pScene->meshes[instance.meshIndex]);
                    constant Submesh & submesh = pMesh->submeshes[ intersection.geometry_id ];
                    
                    uint32_t i0, i1, i2;
                    
                    if ( submesh.shortIndexType )
                    {
                        constant uint16_t* pIndices = (constant uint16_t *)submesh.indices;
                        i0 = pIndices[ intersection.primitive_id * 3 + 0];
                        i1 = pIndices[ intersection.primitive_id * 3 + 1];
                        i2 = pIndices[ intersection.primitive_id * 3 + 2];
                    }
                    else
                    {
                        constant uint32_t* pIndices = (constant uint32_t *)submesh.indices;
                        i0 = pIndices[ intersection.primitive_id * 3 + 0];
                        i1 = pIndices[ intersection.primitive_id * 3 + 1];
                        i2 = pIndices[ intersection.primitive_id * 3 + 2];
                    }

                    float4x4 mv = instance.transform;
                    half3x3 normalMx = half3x3(half3(mv.columns[0].xyz), half3(mv.columns[1].xyz), half3(mv.columns[2].xyz));

                    // Normal

                    half3 n0 = pMesh->generics[i0].normal.xyz;
                    half3 n1 = pMesh->generics[i1].normal.xyz;
                    half3 n2 = pMesh->generics[i2].normal.xyz;

                    half3 n = (n0 * bary3.x) + (n1 * bary3.y) + (n2 * bary3.z);
                    n = normalize(normalMx * n);

                    // Texcoords

                    float2 tc0 = pMesh->generics[i0].texcoord.xy;
                    float2 tc1 = pMesh->generics[i1].texcoord.xy;
                    float2 tc2 = pMesh->generics[i2].texcoord.xy;

                    float2 texcoord = (tc0 * bary3.x) + (tc1 * bary3.y) + (tc2 * bary3.z);

                    // Tangent

                    half3 t0 = pMesh->generics[i0].tangent.xyz;
                    half3 t1 = pMesh->generics[i1].tangent.xyz;
                    half3 t2 = pMesh->generics[i2].tangent.xyz;

                    half3 tangent = (t0 * bary3.x) + (t1 * bary3.y) + (t2 * bary3.z);
                    tangent = normalMx * tangent;

                    // Bitangent

                    half3 bt0 = pMesh->generics[i0].bitangent.xyz;
                    half3 bt1 = pMesh->generics[i1].bitangent.xyz;
                    half3 bt2 = pMesh->generics[i2].bitangent.xyz;

                    half3 bitangent = (bt0 * bary3.x) + (bt1 * bary3.y) + (bt2 * bary3.z);
                    bitangent = normalMx * bitangent;

                    // World Position:

                    packed_float3 wp0 = pMesh->positions[i0].xyz;
                    packed_float3 wp1 = pMesh->positions[i1].xyz;
                    packed_float3 wp2 = pMesh->positions[i2].xyz;

                    packed_float3 worldPosition = (wp0 * bary3.x) + (wp1 * bary3.y) + (wp2 * bary3.z);

                    // Prepare structures for shading:
                    
                    ColorInOut colorIn = {};
                    colorIn.worldPosition = worldPosition;
                    colorIn.normal = float3(n);
                    colorIn.tangent = float3(tangent);
                    colorIn.bitangent = float3(bitangent);
                    colorIn.texCoord = texcoord;

                    texture2d< float > baseColorMap        = submesh.materials[AAPLTextureIndexBaseColor];        //colorMap
                    texture2d< float > normalMap           = submesh.materials[AAPLTextureIndexNormal];           //normalMap
                    texture2d< float > metallicMap         = submesh.materials[AAPLTextureIndexMetallic];         //metallicMap
                    texture2d< float > roughnessMap        = submesh.materials[AAPLTextureIndexRoughness];        //roughnessMap
                    texture2d< float > ambientOcclusionMap = submesh.materials[AAPLTextureIndexAmbientOcclusion]; //ambientOcclusionMap

                    // For shading, adjust the camera position and the world position to
                    // correctly account for reflections in reflections (noticeable on the
                    // sphere's reflection environment map). This is because to correctly
                    // sample the environment map, the shader needs to take into account that
                    // the ray starts from the thin G-Buffer, not from the camera.
                    AAPLCameraData cd( cameraData );
                    cd.cameraPosition = r.origin;
                    colorIn.worldPosition = r.origin + r.direction * intersection.distance;
                    
                    LightingParameters params = calculateParameters(colorIn,
                                                                    cd,
                                                                    lightData,
                                                                    baseColorMap,
                                                                    normalMap,
                                                                    metallicMap,
                                                                    roughnessMap,
                                                                    ambientOcclusionMap,
                                                                    skydomeMap);
                    
                    // check if in shadow
                    raytracing::ray rb;

                    rb.origin = colorIn.worldPosition;
                    rb.direction = normalize(lightData.directionalLightInvDirection);
                    rb.min_distance = 0.1;
                    rb.max_distance = FLT_MAX;
                    
                    float ndotl_bounce = saturate( dot(normal, r.direction) );
                    
                    auto intersectionb = inter.intersect( rb, accelerationStructure, 0xFF );
                    if ( intersectionb.type == raytracing::intersection_type::none )
                    {
                        // if not in shadow, accumlate the direct light as bounce, consider light atten
                        finalIrradiance += lightData.lightIntensity * 0.5 * ndotl_bounce * params.nDotl / (float)skyRayCount;
                    }
                }
                else if ( intersection.type == raytracing::intersection_type::none )
                {
                    // 没打到, 取天光
                    constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
                    float3 c = equirectangularSample( r.direction, linearFilterSampler, skydomeMap ).rgb;
                    finalIrradiance += float4( clamp( c, 0.0f, kMaxHDRValue ), 1.0f) / (float)skyRayCount;
                }
            }
            
            finalColor.x = hit / (float)skyRayCount;
            
            // lightcasting
            uint sunRayCount = 4;
            float shadowHit = 0;
            for( uint i = 0; i < sunRayCount; ++i)
            {
                raytracing::ray r;
                r.origin = position;
                r.direction = normalize(lightData.directionalLightInvDirection + float3(rng.rand() - 0.5, 0.0, rng.rand() - 0.5) * 0.4);
                r.min_distance = 0.1;
                r.max_distance = FLT_MAX;
                
                auto intersection = inter.intersect( r, accelerationStructure, 0xFF );
                if ( intersection.type == raytracing::intersection_type::triangle )
                {
                    // 打到了
                    shadowHit += 1.0;
                }
                else if ( intersection.type == raytracing::intersection_type::none )
                {
                    // 没打到
                }
            }
            
            shadowHit = 1.0 - shadowHit / (float)sunRayCount;
            finalColor.y = shadowHit;
        }
        outImage.write( finalColor, tid );
        outIrradiance.write( finalIrradiance, tid );
    }
}


kernel void rtReflection(
             texture2d< float, access::write >      outImage                [[texture(OutImageIndex)]],
             texture2d< float >                     positions               [[texture(ThinGBufferPositionIndex)]],
             texture2d< float >                     directions              [[texture(ThinGBufferDirectionIndex)]],
             texture2d< float >                     skydomeMap              [[texture(AAPLSkyDomeTexture)]],
             constant AAPLInstanceTransform*        instanceTransforms      [[buffer(BufferIndexInstanceTransforms)]],
             constant AAPLCameraData&               cameraData              [[buffer(BufferIndexCameraData)]],
             constant AAPLLightData&                lightData               [[buffer(BufferIndexLightData)]],
             constant Scene*                        pScene                  [[buffer(SceneIndex)]],
             instance_acceleration_structure        accelerationStructure   [[buffer(AccelerationStructureIndex)]],
             uint2 tid [[thread_position_in_grid]])
{

    uint w = outImage.get_width();
    uint h = outImage.get_height();
    if ( tid.x < w&& tid.y < h )
    {
        float4 finalColor = float4( 0.0, 0.0, 0.0, 1.0 );
        if (is_null_instance_acceleration_structure(accelerationStructure))
        {
            finalColor = float4( 1.0, 0.0, 1.0, 1.0 );
        }
        else
        {
            raytracing::ray r;
            auto position = positions.read(tid).xyz;
            auto normal = directions.read(tid).yzw;
            float3 v = normalize(position - cameraData.cameraPosition);
            auto refl = reflect( v, normal );
            r.origin = position;
            r.direction = refl;
            r.min_distance = 0.1;
            r.max_distance = FLT_MAX;
            
            // 这里造了一根反射ray，找到反射的tri，然后算出光照结果
            // 非常完整，相当于是单像素的scene rendering了
            // 这里可以作一个简单的raytraced ao，半球内随机打ray，有结果的贡献遮蔽
            
            raytracing::intersector<raytracing::instancing, raytracing::triangle_data> inter;
            inter.assume_geometry_type( raytracing::geometry_type::triangle );
            auto intersection = inter.intersect( r, accelerationStructure, 0xFF );
            if ( intersection.type == raytracing::intersection_type::triangle )
            {
                float2 bary2 = intersection.triangle_barycentric_coord;
                float3 bary3 = float3( 1.0 - (bary2.x + bary2.y), bary2.x, bary2.y );

                constant Instance& instance = pScene->instances[ intersection.instance_id ];
                constant Mesh* pMesh = &(pScene->meshes[instance.meshIndex]);
                constant Submesh & submesh = pMesh->submeshes[ intersection.geometry_id ];
                
                uint32_t i0, i1, i2;
                
                if ( submesh.shortIndexType )
                {
                    constant uint16_t* pIndices = (constant uint16_t *)submesh.indices;
                    i0 = pIndices[ intersection.primitive_id * 3 + 0];
                    i1 = pIndices[ intersection.primitive_id * 3 + 1];
                    i2 = pIndices[ intersection.primitive_id * 3 + 2];
                }
                else
                {
                    constant uint32_t* pIndices = (constant uint32_t *)submesh.indices;
                    i0 = pIndices[ intersection.primitive_id * 3 + 0];
                    i1 = pIndices[ intersection.primitive_id * 3 + 1];
                    i2 = pIndices[ intersection.primitive_id * 3 + 2];
                }

                float4x4 mv = instance.transform;
                half3x3 normalMx = half3x3(half3(mv.columns[0].xyz), half3(mv.columns[1].xyz), half3(mv.columns[2].xyz));

                // Normal

                half3 n0 = pMesh->generics[i0].normal.xyz;
                half3 n1 = pMesh->generics[i1].normal.xyz;
                half3 n2 = pMesh->generics[i2].normal.xyz;

                half3 n = (n0 * bary3.x) + (n1 * bary3.y) + (n2 * bary3.z);
                n = normalize(normalMx * n);

                // Texcoords

                float2 tc0 = pMesh->generics[i0].texcoord.xy;
                float2 tc1 = pMesh->generics[i1].texcoord.xy;
                float2 tc2 = pMesh->generics[i2].texcoord.xy;

                float2 texcoord = (tc0 * bary3.x) + (tc1 * bary3.y) + (tc2 * bary3.z);

                // Tangent

                half3 t0 = pMesh->generics[i0].tangent.xyz;
                half3 t1 = pMesh->generics[i1].tangent.xyz;
                half3 t2 = pMesh->generics[i2].tangent.xyz;

                half3 tangent = (t0 * bary3.x) + (t1 * bary3.y) + (t2 * bary3.z);
                tangent = normalMx * tangent;

                // Bitangent

                half3 bt0 = pMesh->generics[i0].bitangent.xyz;
                half3 bt1 = pMesh->generics[i1].bitangent.xyz;
                half3 bt2 = pMesh->generics[i2].bitangent.xyz;

                half3 bitangent = (bt0 * bary3.x) + (bt1 * bary3.y) + (bt2 * bary3.z);
                bitangent = normalMx * bitangent;

                // World Position:

                packed_float3 wp0 = pMesh->positions[i0].xyz;
                packed_float3 wp1 = pMesh->positions[i1].xyz;
                packed_float3 wp2 = pMesh->positions[i2].xyz;

                packed_float3 worldPosition = (wp0 * bary3.x) + (wp1 * bary3.y) + (wp2 * bary3.z);

                // Prepare structures for shading:
                
                ColorInOut colorIn = {};
                colorIn.worldPosition = worldPosition;
                colorIn.normal = float3(n);
                colorIn.tangent = float3(tangent);
                colorIn.bitangent = float3(bitangent);
                colorIn.texCoord = texcoord;

                texture2d< float > baseColorMap        = submesh.materials[AAPLTextureIndexBaseColor];        //colorMap
                texture2d< float > normalMap           = submesh.materials[AAPLTextureIndexNormal];           //normalMap
                texture2d< float > metallicMap         = submesh.materials[AAPLTextureIndexMetallic];         //metallicMap
                texture2d< float > roughnessMap        = submesh.materials[AAPLTextureIndexRoughness];        //roughnessMap
                texture2d< float > ambientOcclusionMap = submesh.materials[AAPLTextureIndexAmbientOcclusion]; //ambientOcclusionMap

                // For shading, adjust the camera position and the world position to
                // correctly account for reflections in reflections (noticeable on the
                // sphere's reflection environment map). This is because to correctly
                // sample the environment map, the shader needs to take into account that
                // the ray starts from the thin G-Buffer, not from the camera.
                AAPLCameraData cd( cameraData );
                cd.cameraPosition = r.origin;
                colorIn.worldPosition = r.origin + r.direction * intersection.distance;
                
                LightingParameters params = calculateParameters(colorIn,
                                                                cd,
                                                                lightData,
                                                                baseColorMap,
                                                                normalMap,
                                                                metallicMap,
                                                                roughnessMap,
                                                                ambientOcclusionMap,
                                                                skydomeMap);
                
                finalColor = float4( computeSpecular( params ) + lightData.lightIntensity * computeDiffuse( params ), 1.0 );
                finalColor.rgb *= lightData.lightIntensity;

            }
            else if ( intersection.type == raytracing::intersection_type::none )
            {
                constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
                float3 c = equirectangularSample( r.direction, linearFilterSampler, skydomeMap ).rgb;
                finalColor = float4( clamp( c, 0.0f, kMaxHDRValue ), 1.0f);
            }
        }
        outImage.write( finalColor, tid );
    }
}

struct VertexInOut
{
    float4 position [[position]];
    float2 uv;
};

constant float4 s_quad[] = {
    float4( -1.0f, +1.0f, 0.0f, 1.0f ),
    float4( -1.0f, -1.0f, 0.0f, 1.0f ),
    float4( +1.0f, -1.0f, 0.0f, 1.0f ),
    float4( +1.0f, -1.0f, 0.0f, 1.0f ),
    float4( +1.0f, +1.0f, 0.0f, 1.0f ),
    float4( -1.0f, +1.0f, 0.0f, 1.0f )
};

constant float2 s_quadtc[] = {
    float2( 0.0f, 0.0f ),
    float2( 0.0f, 1.0f ),
    float2( 1.0f, 1.0f ),
    float2( 1.0f, 1.0f ),
    float2( 1.0f, 0.0f ),
    float2( 0.0f, 0.0f )
};

vertex VertexInOut vertexPassthrough( uint vid [[vertex_id]] )
{
    VertexInOut o;
    o.position = s_quad[vid];
    o.uv = s_quadtc[vid];
    return o;
}

fragment float4 fragmentPassthrough( VertexInOut in [[stage_in]], texture2d< float > tin )
{
    constexpr sampler s( address::repeat, min_filter::linear, mag_filter::linear );
    return tin.sample( s, in.uv );
}

fragment float4 fragmentBloomThreshold( VertexInOut in [[stage_in]],
                                       texture2d< float > tin [[texture(0)]],
                                       constant float* threshold [[buffer(0)]] )
{
    constexpr sampler s( address::repeat, min_filter::linear, mag_filter::linear );
    float4 c = tin.sample( s, in.uv );
    if ( dot( c.rgb, float3( 0.299f, 0.587f, 0.144f ) ) > (*threshold) )
    {
        return c;
    }
    return float4(0.f, 0.f, 0.f, 1.f );
}

// The standard ACES tonemap function from "Modern Rendering" sample.
static float3 ToneMapACES(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

fragment float4 fragmentPostprocessMerge( VertexInOut in [[stage_in]],
                                         constant float& exposure [[buffer(0)]],
                                         texture2d< float > texture0 [[texture(0)]])
{
    constexpr sampler s( address::repeat, min_filter::linear, mag_filter::linear );
    float4 t0 = texture0.sample( s, in.uv );
    float3 c = t0.rgb;
    c = ToneMapACES( c * exposure );
    return float4( c, 1.0f );
}

#endif
