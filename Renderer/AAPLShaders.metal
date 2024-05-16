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

constant unsigned int primes[] = {
    2,   3,  5,  7,
    11, 13, 17, 19,
    23, 29, 31, 37,
    41, 43, 47, 53,
    59, 61, 67, 71,
    73, 79, 83, 89
};

// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base. The Halton sequence is a low discrepency sequence: the values appear
// random, but are more evenly distributed than a purely random sequence. Each random
// value used to render the image uses a different independent dimension, `d`,
// and each sample (frame) uses a different index `i`. To decorrelate each pixel,
// you can apply a random offset to `i`.
float halton(unsigned int i, unsigned int d) {
    unsigned int b = primes[d];

    float f = 1.0f;
    float invB = 1.0f / b;

    float r = 0;

    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }

    return r;
}

constant float PI = 3.1415926535897932384626433832795;
constant float kMaxHDRValue = 500.0f;
constant float rayGap = 0.005;

typedef struct
{
    float4 position [[position]];
    float4 currPosition;
    float4 prevPosition;
    float3 viewPosition;
    float3 worldPosition;
    float3 normal;
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
                                 min_filter::linear,
                                 max_anisotropy(16));

constexpr sampler nearestSampler(address::repeat,
                                 min_filter::nearest,
                                 mag_filter::nearest,
                                 mip_filter::none);


// Polynomial approximation by Christophe Schlick
float schlick(const float cosine, const float refractionIndex)
{
    float r0 = (1.0 - refractionIndex) / (1.0 + refractionIndex);
    r0 *= r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

float3 scatterGBuffer( float3 direction, float3 surfNormal, float ior, float roughness, float3 albedo)
{
    const float dotVaule = dot(direction, surfNormal);
    const float3 outwardNormal = dotVaule > 0 ? -surfNormal : surfNormal;
    const float cosine = dotVaule > 0 ? ior * dotVaule : -dotVaule;
    const float reflectProb = schlick(cosine, ior);
        
    return mix( albedo, float3(1,1,1), reflectProb);
}

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

float2 EnvBRDFApproxLazarov(float Roughness, float NoV)
{
    // [ Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II" ]
    // Adaptation to fit our G term.
    const float4 c0 = { -1, -0.0275, -0.572, 0.022 };
    const float4 c1 = { 1, 0.0425, 1.04, -0.04 };
    float4 r = Roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
    return AB;
}

float3 EnvBRDFApprox( float3 SpecularColor, float Roughness, float NoV )
{
    float2 AB = EnvBRDFApproxLazarov(Roughness, NoV);

    // Anything less than 2% is physically impossible and is instead considered to be shadowing
    // Note: this is needed for the 'specular' show flag to work, since it uses a SpecularColor of 0
    float F90 = saturate( 50.0 * SpecularColor.g );

    return SpecularColor * AB.x + F90 * AB.y;
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
                                       float3 baseColor,
                                       float roughness,
                                       float metallic,
                                       texture2d<float>   baseColorMap,
                                       texture2d<float>   metallicMap,
                                       texture2d<float>   roughnessMap
                                       )
{
    LightingParameters parameters;
    parameters.baseColor = baseColorMap.sample(linearSampler, in.texCoord.xy * float2(1.f, -1.f) + float2(1.f,1.f)) * float4(baseColor, 1.0);
    // the tangent space in not correct, normalmap ignore
    parameters.normal = in.normal;//computeNormalMap(in, normalMap);
    parameters.viewDir = normalize(cameraData.cameraPosition - float3(in.worldPosition));
    parameters.roughness = roughness;//mix(0.01,1.0,roughnessMap.sample(linearSampler, in.texCoord.xy).x);
    parameters.metalness = metallic;//max(metallicMap.sample(linearSampler, in.texCoord.xy).x, 0.1);
    parameters.ambientOcclusion = 1.0;//ambientOcclusionMap.sample(linearSampler, in.texCoord.xy).x;
    parameters.reflectedVector = reflect(-parameters.viewDir, parameters.normal);
    
//    constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
//    float3 c = equirectangularSample(parameters.reflectedVector, linearFilterSampler, skydomeMap).rgb;
//    parameters.irradiatedColor = clamp(c, 0.f, kMaxHDRValue);

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

    // Reflections and lighting that occur in the world space, so
    // `camera.viewMatrix` isn’t taken into consideration here.
    float4x4 objToWorld = instanceTransform.modelViewMatrix;
    out.worldPosition = (objToWorld * position).xyz;

    float3x3 normalMx = float3x3(objToWorld.columns[0].xyz,
                                 objToWorld.columns[1].xyz,
                                 objToWorld.columns[2].xyz);
    out.normal = normalMx * normalize(in.normal);

    float3 v = out.worldPosition - cameraData.cameraPosition;
    out.r = reflect( v, out.normal );

    out.texCoord = in.texCoord;

    return out;
}

float2 calculateScreenCoord( float4 rawposition )
{
    float2 screenTexcoord = rawposition.xy / rawposition.w * float2(0.5f, -0.5f) + float2(0.5f,0.5f);
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

    float2 screenTexcoord = calculateScreenCoord( in.currPosition );
    
    constant Mesh* pMesh = &(pScene->meshes[ pScene->instances[submeshKeypath.instanceID].meshIndex ]);
    constant Submesh* pSubmesh = &(pMesh->submeshes[submeshKeypath.submeshID]);

    LightingParameters params = calculateParameters(in,
                                                    cameraData,
                                                    lightData,
                                                    pSubmesh->baseColor,
                                                    pSubmesh->roughness,
                                                    pSubmesh->metallic,
                                                    pSubmesh->materials[AAPLTextureIndexBaseColor],        //colorMap
                                                    //pSubmesh->materials[AAPLTextureIndexNormal],           //normalMap
                                                    pSubmesh->materials[AAPLTextureIndexMetallic],         //metallicMap
                                                    pSubmesh->materials[AAPLTextureIndexRoughness]        //roughnessMap
                                                    //pSubmesh->materials[AAPLTextureIndexAmbientOcclusion], //ambientOcclusionMap
                                                    );
    float3 skylight = params.ambientOcclusion * 0.1;
    float li = lightData.lightIntensity;
    params.roughness += cameraData.roughnessBias;
    clamp( params.roughness, 0.f, 1.0f );

    if ( is_raytracing_enabled )
    {
        float4 gi = rtShadings.sample(colorSampler, screenTexcoord).xyzw;
        li *= gi.y;
        
        float3 reflectedColor = rtReflections.sample(colorSampler, screenTexcoord, level(0)).xyz;
        params.reflectedColor = reflectedColor * EnvBRDFApprox(float3(0.04), params.roughness, params.nDotv);
        params.irradiatedColor = 0;//gi.y;//reflectedColor * gi.x;
        
        float4 irr = rtIrrandiance.sample(colorSampler, screenTexcoord).xyzw;
        skylight *= irr.xyz * 5.0;
        
    }
    params.metalness += cameraData.metallicBias;
    float4 final_color = float4(skylight * float3((params.baseColor) * (1.0 - params.metalness)) + computeSpecular(params) + li * computeDiffuse(params), 1.0f) + float4(pSubmesh->emissionColor + params.reflectedColor, 1.0f);

    return final_color;
}



fragment float4 irradianceShader(ColorInOut in [[stage_in]],
                                 texture2d<float>            albedoBuffer         [[ texture(AAPLTextureIndexGBufferAlbedo)]],
                                 texture2d<float>            rtIrrandiance         [[ texture(AAPLTextureIndexIrrGI)]])
{
    float2 screenTexcoord = calculateScreenCoord( in.currPosition );
    float4 reflectedColor = rtIrrandiance.sample(nearestSampler, screenTexcoord, level(0));
    float4 albedoColor = albedoBuffer.sample(nearestSampler, screenTexcoord, level(0));
    reflectedColor = reflectedColor * albedoColor;
    reflectedColor.a = 1.0;
    return reflectedColor;
}


struct ThinGBufferOut
{
    float4 position [[color(0)]];
    float4 direction [[color(1)]];
    float2 motionVector [[color(2)]];
    float4 albedo [[color(3)]];
};

fragment ThinGBufferOut gBufferFragmentShader(ColorInOut in [[stage_in]],
                                              constant AAPLCameraData&    cameraData            [[ buffer(BufferIndexCameraData) ]],
                                              constant AAPLLightData&     lightData             [[ buffer(BufferIndexLightData) ]],
                                              constant AAPLSubmeshKeypath&submeshKeypath        [[ buffer(BufferIndexSubmeshKeypath)]],
                                              constant Scene*             pScene                [[ buffer(SceneIndex)]])
{
    constant Mesh* pMesh = &(pScene->meshes[ pScene->instances[submeshKeypath.instanceID].meshIndex ]);
    constant Submesh* pSubmesh = &(pMesh->submeshes[submeshKeypath.submeshID]);
    LightingParameters params = calculateParameters(in,
                                                    cameraData,
                                                    lightData,
                                                    pSubmesh->baseColor,
                                                    pSubmesh->roughness,
                                                    pSubmesh->metallic,
                                                    pSubmesh->materials[AAPLTextureIndexBaseColor],        //colorMap
                                                    //pSubmesh->materials[AAPLTextureIndexNormal],           //normalMap
                                                    pSubmesh->materials[AAPLTextureIndexMetallic],         //metallicMap
                                                    pSubmesh->materials[AAPLTextureIndexRoughness]        //roughnessMap
                                                    //pSubmesh->materials[AAPLTextureIndexAmbientOcclusion], //ambientOcclusionMap
                                                    );
    ThinGBufferOut out;

    out.position = float4(in.worldPosition, params.roughness);
  
    float2 motionVector = 0.0f;
    if (cameraData.frameIndex > 0) {
        float2 uv = in.currPosition.xy / in.currPosition.w * float2(0.5f, -0.5f) + float2(0.5f,0.5f);
        float2 prevUV = in.prevPosition.xy / in.prevPosition.w * float2(0.5f, -0.5f) + float2(0.5f,0.5f);
        
        uv -= cameraData.jitter;
        prevUV -= cameraData.prev_jitter;
        
        motionVector = (uv - prevUV);
    }
    
    float3 eyeRay = normalize(in.worldPosition - cameraData.cameraPosition);
    
    // Then the motion vector is simply the difference between the two
    out.direction = float4(length(in.viewPosition), normalize(in.normal));
    out.motionVector = motionVector;
    // this should consider the schilick
    
    out.albedo.rgb = scatterGBuffer( eyeRay, in.normal, 1.45, params.roughness, params.baseColor.rgb);
    
    return out;
}

#if __METAL_VERSION__ >= 230

#pragma mark - Ray tracing
using raytracing::instance_acceleration_structure;

// Maps two uniformly random numbers to a uniform distribution within a cone
float3 sampleCone(float2 u, float cosAngle) {
    float phi = 2.0f * M_PI_F * u.x;
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    float cos_theta = 1.0f - u.y + u.y * cosAngle;
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

// Aligns a direction such that the "up" direction (0, 1, 0) maps to the given
// surface normal direction
float3 alignWithNormal(float3 sample, float3 normal) {
    float3 up = normal;
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));
    float3 forward = cross(right, up);
    return sample.x * right + sample.y * up + sample.z * forward;
}

// Uses the inversion method to map two uniformly random numbers to a 3D
// unit hemisphere, where the probability of a given sample is proportional to the cosine
// of the angle between the sample direction and the "up" direction (0, 1, 0).
inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

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
            uint skyRayCount = 1;
            uint sunRayCount = 1;
            
            float hit = 0.0;
            raytracing::intersector<raytracing::instancing, raytracing::triangle_data> inter;
            inter.assume_geometry_type( raytracing::geometry_type::triangle );
            
            for( uint i = 0; i < skyRayCount; ++i)
            {
                raytracing::ray r;

                // 这里需要构造一个基于法线的hemisphere来采样，并且引入重要性分布，使用hottonPattern
                r.origin = position + normal * rayGap;
                float3 traceNormal = normal;
                
                //r.direction = normalize(float3(rng.rand() - 0.5,rng.rand() - 0.5,rng.rand() - 0.5));

                float2 uv = float2(rng.rand(), rng.rand());
                float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(uv);
                worldSpaceSampleDirection = alignWithNormal(worldSpaceSampleDirection, traceNormal);
                
                r.direction = worldSpaceSampleDirection;
                
                r.min_distance = 0;
                r.max_distance = FLT_MAX;
                
                // 在半球内发射射线
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

                    // World Position:

                    packed_float3 wp0 = pMesh->positions[i0].xyz;
                    packed_float3 wp1 = pMesh->positions[i1].xyz;
                    packed_float3 wp2 = pMesh->positions[i2].xyz;

                    packed_float3 worldPosition = (wp0 * bary3.x) + (wp1 * bary3.y) + (wp2 * bary3.z);

                    // Prepare structures for shading:
                    
                    ColorInOut colorIn = {};
                    colorIn.worldPosition = worldPosition;
                    colorIn.normal = float3(n);
                    colorIn.texCoord = texcoord;

                    texture2d< float > baseColorMap        = submesh.materials[AAPLTextureIndexBaseColor];        //colorMap
                    texture2d< float > metallicMap         = submesh.materials[AAPLTextureIndexMetallic];      //metallicMap
                    texture2d< float > roughnessMap        = submesh.materials[AAPLTextureIndexRoughness];        //roughnessMap

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
                                                                    submesh.baseColor,
                                                                    submesh.roughness,
                                                                    submesh.metallic,
                                                                    baseColorMap,
                                                                    //normalMap,
                                                                    metallicMap,
                                                                    roughnessMap
                                                                    //ambientOcclusionMap,
                                                                    );
                    
                    // check if in shadow
                    raytracing::ray rb;

                    rb.origin = colorIn.worldPosition;
                    rb.direction = normalize(lightData.directionalLightInvDirection);
                    rb.min_distance = rayGap;
                    rb.max_distance = FLT_MAX;
                    
                    float ndotl_bounce = saturate( dot(normal, r.direction) );
                    
                    // emission, with attenuion
                    float mappedDist = intersection.distance;
                    float atten = max(1.0, mappedDist * 1.0);
                    float atten2 = atten * atten;
                    
                    auto intersectionb = inter.intersect( rb, accelerationStructure, 0xFF );
                    if ( intersectionb.type == raytracing::intersection_type::none )
                    {
                        // if not in shadow, accumlate the direct light as bounce, consider light atten
                        // should consider the light result
                        finalIrradiance.xyz += params.baseColor.xyz * (1.0 - params.metalness) * lightData.lightIntensity * params.nDotl / atten / (float)skyRayCount;
                    }
                    
                    finalIrradiance.xyz += submesh.emissionColor * ndotl_bounce / atten2 / (float)skyRayCount;
                }
                else if ( intersection.type == raytracing::intersection_type::none )
                {
                    // 打到了, 取一次反弹
                    hit += 1.0;
                    
                    // 没打到, 取天光
                    constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
                    float3 c = equirectangularSample( r.direction, linearFilterSampler, skydomeMap ).rgb;
                    finalIrradiance += float4( clamp( c, 0.0f, kMaxHDRValue ), 1.0f) / (float)skyRayCount;
                }
            }
            
            finalColor.x = hit / (float)skyRayCount;
            
            // lightcasting
            float shadowHit = 0;
            for( uint i = 0; i < sunRayCount; ++i)
            {
                raytracing::ray r;
                r.origin = position + normal * rayGap;
                //r.direction = normalize(lightData.directionalLightInvDirection + float3(rng.rand() - 0.5, 0.0, rng.rand() - 0.5) * 0.4);
                float2 uv = float2(rng.rand(), rng.rand());
                float3 sample = sampleCone(uv, cos(1.f / 180.0f * M_PI_F));
                r.direction = alignWithNormal(sample, lightData.directionalLightInvDirection);
                
                r.min_distance = 0.0;
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

kernel void rtBounce(
             texture2d< float, access::write >      outImage                [[texture(OutImageIndex)]],
             texture2d< float, access::write >      outReflcetion           [[texture(RefectionMapIndex)]],
             texture2d< float >                     irradiance              [[texture(IrradianceMapIndex)]],
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
            auto rawdata = positions.read(tid).xyzw;
            auto position = rawdata.xyz;
            float roughness = rawdata.w;
            auto normal = directions.read(tid).yzw;
            Loki rng = Loki(tid.x + 1, tid.y + 1, cameraData.frameIndex);
            
            // 构造一个在normal半球内的ray
            uint skyRayCount = 1;

            raytracing::intersector<raytracing::instancing, raytracing::triangle_data> inter;
            inter.assume_geometry_type( raytracing::geometry_type::triangle );
            
            for( uint i = 0; i < skyRayCount; ++i)
            {
                raytracing::ray r;

                // 这里需要构造一个基于法线的hemisphere来采样，并且引入重要性分布，使用hottonPattern
                r.origin = position;
                float2 uv = float2(rng.rand(), rng.rand());
                float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(uv);
                worldSpaceSampleDirection = alignWithNormal(worldSpaceSampleDirection, normal);
                
                r.direction = worldSpaceSampleDirection;
                
                r.min_distance = rayGap;
                r.max_distance = FLT_MAX;
                
                // 在半球内发射射线
                auto intersection = inter.intersect( r, accelerationStructure, 0xFF );
                if ( intersection.type == raytracing::intersection_type::triangle )
                {
                    // 打到了, 从上一次的反弹结果取颜色
                    auto worldPosition = r.origin + r.direction * intersection.distance;
                    
                    constant Instance& instance = pScene->instances[ intersection.instance_id ];
                    constant Mesh* pMesh = &(pScene->meshes[instance.meshIndex]);
                    constant Submesh & submesh = pMesh->submeshes[ intersection.geometry_id ];
                    
                    // 从worldPosition计算出采样坐标
                    auto hpos = cameraData.projectionMatrix * cameraData.viewMatrix * float4(worldPosition, 1.0);
                    auto screenTexcoord = calculateScreenCoord(hpos);
                    
                    constexpr sampler colorSampler(mip_filter::linear,
                                                   mag_filter::linear,
                                                   min_filter::linear);
                    float3 bounceColor = irradiance.sample(colorSampler, screenTexcoord).xyz * submesh.baseColor.xyz * 0.5;// 衰减为0.5
                    // 这里hpos有可能是在实际像素的后方，应该在用depth检查一下
                    finalIrradiance.xyz += bounceColor / (float)skyRayCount;
                }
            }
            
            uint specularRayCount = 1;
            for( uint i = 0; i < specularRayCount; ++i)
            {
                raytracing::ray r;
                r.origin = position;
                float2 uv = float2(rng.rand(), rng.rand());
                // this cone radius is roughness tageted
                float3 sample = sampleCone(uv, cos(max(0.001f, roughness * 30.f) / 180.0f * M_PI_F));
                
                float3 v = normalize(position - cameraData.cameraPosition);
                auto refl = reflect( v, normal );

                r.direction = alignWithNormal(sample, refl);
                
                r.min_distance = 0.0;
                r.max_distance = FLT_MAX;
                
                auto intersection = inter.intersect( r, accelerationStructure, 0xFF );
                if ( intersection.type == raytracing::intersection_type::triangle )
                {
                    // 打到了, 从上一次的反弹结果取颜色
                    auto worldPosition = r.origin + r.direction * intersection.distance;
                    
                    constant Instance& instance = pScene->instances[ intersection.instance_id ];
                    constant Mesh* pMesh = &(pScene->meshes[instance.meshIndex]);
                    constant Submesh & submesh = pMesh->submeshes[ intersection.geometry_id ];
                    
                    // 从worldPosition计算出采样坐标
                    auto hpos = cameraData.projectionMatrix * cameraData.viewMatrix * float4(worldPosition, 1.0);
                    auto screenTexcoord = calculateScreenCoord(hpos);
                    
                    if(screenTexcoord.x < 1.0 && screenTexcoord.y < 1.0)
                    {
                        float3 targetpos = positions.sample(nearestSampler, screenTexcoord).xyz;
                        if( distance(targetpos, worldPosition) < 0.1 )
                        {
                            float3 specular = irradiance.sample(linearSampler, screenTexcoord).xyz * submesh.baseColor;
                            finalColor.xyz += specular / (float)specularRayCount;
                        }
                    }
                }
                else if ( intersection.type == raytracing::intersection_type::none )
                {
                    // 没打到
                    constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
                    float3 c = equirectangularSample( r.direction, linearFilterSampler, skydomeMap ).rgb;
                    finalColor += float4( clamp( c, 0.0f, kMaxHDRValue ), 1.0f) / (float)specularRayCount;
                }
            }
        }
        // combine with first bounce here
        float4 firstbounce = irradiance.read(tid);
        outReflcetion.write( finalColor, tid );
        outImage.write( firstbounce + finalIrradiance, tid );
    }
}

struct ScatterPayload
{
    float3 direction;
    float3 color;
    bool isEnded;
};

ScatterPayload scatter(float2 randomuv, float3 direction, float3 surfNormal, float ior, float roughness, float3 albedo)
{
    ScatterPayload payload;
    const float dotVaule = dot(direction, surfNormal);
    const float3 outwardNormal = dotVaule > 0 ? -surfNormal : surfNormal;
    const float cosine = dotVaule > 0 ? ior * dotVaule : -dotVaule;
    const float reflectProb = schlick(cosine, ior);
        
    if( randomuv.x > reflectProb )
    {
        payload.direction = alignWithNormal(sampleCosineWeightedHemisphere(randomuv), outwardNormal);
        payload.isEnded = dotVaule > 0;
        payload.color = albedo;
    }
    else
    {
        float3 reflected = reflect(direction, outwardNormal);
        payload.direction = alignWithNormal(sampleCone(randomuv, cos(roughness * 45.f / 180.0f * M_PI_F)), reflected);
        payload.isEnded = dotVaule > 0;
        payload.color = float3(1.5,1.5,1.5);
    }
    
    return payload;
}

kernel void rtGroundTruth(
             texture2d< float, access::write >      outImage                [[texture(OutImageIndex)]],
             texture2d< float >                     positions               [[texture(ThinGBufferPositionIndex)]],
             texture2d< float >                     directions              [[texture(ThinGBufferDirectionIndex)]],
             texture2d< float >                     albedo                  [[texture(AAPLTextureIndexGBufferAlbedo)]],
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
        float4 finalColor = float4(0,0,0,1);

        auto rawdata = positions.read(tid).xyzw;
        auto positionBase = rawdata.xyz;
        auto normalBase = normalize( directions.read(tid).yzw );
        
        Loki rng = Loki(tid.x + 1, tid.y + 1, cameraData.frameIndex);
        float random = rng.rand() * (w*h);
        
        uint spp = 1;
        
        uint counter = 0;
        for( uint s = 0; s < spp; s += 1  )
        {
            float4 rayColor = float4( 1.0, 1.0, 1.0, 1.0 );
            // 构造一个在normal半球内的ray
            uint bounceCount = 2;

            raytracing::intersector<raytracing::instancing, raytracing::triangle_data> inter;
            inter.assume_geometry_type( raytracing::geometry_type::triangle );
            
            raytracing::ray r;
            r.min_distance = 0;
            r.max_distance = FLT_MAX;
            
            // primary hit from rasteraze result
            if(true)
            {
                float3 eyeRay = normalize(positionBase - cameraData.cameraPosition);
                float2 uv = float2( halton(cameraData.frameIndex + random, counter),
                                    halton(cameraData.frameIndex + random, counter+1));
                counter += 2;
                
                r.origin = positionBase + normalBase * rayGap;
                ScatterPayload payload = scatter(uv, eyeRay, normalBase, 1.45f, rawdata.a, float3(1,1,1));
                r.direction = payload.direction;
            }
            else
            // all physical: ray from camera
            {
                const float2 pixel = float2(tid.x + 0.0, h - tid.y + 0.0);
                const float2 lensuv = (pixel / float2(w,h)) * 2.0 - 1.0;
                
                //float2 offset = Camera.Aperture/2 * RandomInUnitDisk(Ray.RandomSeed);
                float2 offset = float2(0.0);
                float4 origin = cameraData.invViewMatrix * float4(offset, 0, 1);
                float4 target = cameraData.invProjectionMatrix * (float4(lensuv.x, lensuv.y, 1, 1));
                float4 direction = cameraData.invViewMatrix * float4(normalize(target.xyz * 5.0 - float3(offset, 0)), 0);

                r.origin = origin.xyz;
                r.direction = direction.xyz;
            }
            
            
            for( uint i = 0; i <= bounceCount; ++i)
            {
                // if reach the last bounce u still here, no hit light, exit
                if(i == bounceCount)
                {
                    rayColor = float4(0,0,0,0);
                    break;
                }
                
                // trace loop
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

                    texture2d< float > baseColorMap        = submesh.materials[AAPLTextureIndexBaseColor];        //colorMap
                    
                    float3 surfNormal = float3(n);
                    float3 worldpos = r.origin + r.direction * intersection.distance + surfNormal * rayGap;
                    float3 albedo = baseColorMap.sample(linearSampler, texcoord.xy * float2(1.f, -1.f) + float2(1.f,1.f)).rgb * submesh.baseColor;
                    
                    // if hit emitter, stop here and accumulate the light
                    float lum = dot(submesh.emissionColor, float3(.33,.33,.33));
                    if(lum > 0.1)
                    {
                        rayColor.xyz *= submesh.emissionColor.xyz;
                        break;
                    }
                    
                    // scatter
                    float2 uv = float2( halton(cameraData.frameIndex + random, counter),
                                        halton(cameraData.frameIndex + random, counter+1));
                    counter += 2;
                    
                    ScatterPayload payload = scatter(uv, r.direction, surfNormal, 1.45f, submesh.roughness, albedo);
                    rayColor.xyz *= payload.color;
                    
                    if(payload.isEnded)
                    {
                        break;
                    }
                    
                    // prepare next ray
                    r.origin = worldpos;
                    r.direction = payload.direction;
                }
                else if ( intersection.type == raytracing::intersection_type::none )
                {
                    // 没打到, 取天光
                    //constexpr sampler linearFilterSampler(coord::normalized, address::clamp_to_edge, filter::linear);
                    //float3 c = equirectangularSample( r.direction, linearFilterSampler, skydomeMap ).rgb * 0.2;
                    //finalColor.xyz += color * clamp( c, 0.0f, kMaxHDRValue );
                    //rayColor.xyz = clamp( c, 0.0f, kMaxHDRValue );
                    rayColor.xyz = float3(0.0);
                    break;
                }
            }
            finalColor += rayColor;
        }
     
        outImage.write( finalColor / (float)spp, tid );
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
