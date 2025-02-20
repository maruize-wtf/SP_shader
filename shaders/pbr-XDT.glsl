/*************************************************************************
* ADOBE CONFIDENTIAL
* ___________________
* Copyright 2014 Adobe
* All Rights Reserved.
* NOTICE:  All information contained herein is, and remains
* the property of Adobe and its suppliers, if any. The intellectual
* and technical concepts contained herein are proprietary to Adobe
* and its suppliers and are protected by all applicable intellectual
* property laws, including trade secret and copyright laws.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Adobe.
*************************************************************************/

//- Substance 3D Painter Metal/Rough and opacity PBR shader
//- ================================================
//-
//- Import from libraries.
import lib-pbr.glsl
import lib-bent-normal.glsl
//import lib-emissive.glsl
//import lib-pom.glsl
//import lib-sss.glsl
import lib-alpha.glsl




//- Channels needed for metal/rough workflow are bound here.
//: param auto channel_basecolor
uniform SamplerSparse basecolor_tex;
//: param auto channel_roughness
uniform SamplerSparse roughness_tex;
//: param auto channel_metallic
uniform SamplerSparse metallic_tex;
//: param auto channel_specularlevel
uniform SamplerSparse specularlevel_tex;

//: param auto camera_view_matrix 
uniform mat4 uniform_camera_view_matrix_it;
//: param auto camera_view_matrix 
uniform mat4 uniform_camera_view_matrix;


// 添加光照方向和光照颜色的参数接口
//: param custom { "default": false, "label": "反转z轴", "widget": "Boolean", "group": "Directional Light" }
uniform bool enable_InvertZ;
//: param custom { "default": 58, "label": "Light Rotation X", "min": 0, "max": 360, "group": "Directional Light"  } 
uniform int light_rotation_x;
//: param custom { "default": 32, "label": "Light Rotation Y", "min": 0, "max": 360, "group": "Directional Light"  } 
uniform int light_rotation_y;
//: param custom { "default": 0, "label": "Light Rotation Z", "min": 0, "max": 360, "group": "Directional Light"  } 
uniform int light_rotation_z;
//: param custom { "default": [1, 0.973, 0.564], "label": "Light Color", "widget": "color", "group": "Directional Light" } 
uniform vec3 light_color;

//: param custom { "default": [0.5, 0.624, 0.834], "label": "Sky Color", "widget": "color", "group": "Environment Lighting" } 
uniform vec3 gradient_skyColor;
//: param custom { "default": [0.627, 0.584, 0.902], "label": "Equatorent Color", "widget": "color", "group": "Environment Lighting" } 
uniform vec3 gradient_equatorColor;
//: param custom { "default": [0.537, 0.510, 0.479], "label": "Ground Color", "widget": "color", "group": "Environment Lighting" } 
uniform vec3 gradient_groundColor;

//: param custom { "default": true, "label": "Metallic", "widget": "Boolean" }
uniform bool enable_metallic;

//: param custom { "default": 1, "label": "Matcap Strength", "min": 0.0, "max": 10.0, "group": "Matcap Settings" }
uniform float matcap_strength;

//: param custom { "default": 1, "label": "Matcap Roughness", "min": 0.0, "max": 5.0, "group": "Matcap Settings" }
uniform float matcap_roughness;

//: param custom { "default": true, "label": "Post ColorAdjustments", "widget": "Boolean", "group": "Post Settings" }
uniform bool post_colorAdjustments;
//: param custom { "default": 0.4, "label": "Post Exposure", "min": -5.0, "max": 5.0, "group": "Post Settings" }
uniform float post_exposure;
//: param custom { "default": 20, "label": "Post Contrast", "min": -100, "max":100, "group": "Post Settings" }
uniform int post_contrast;
//: param custom { "default": -5, "label": "Post Saturatin", "min": -100, "max":100, "group": "Post Settings" }
uniform int post_saturation;
//: param custom { "default": true, "label": "Post Tonemapping", "widget": "Boolean", "group": "Post Settings" }
uniform bool post_tonemapping;


const float LogC_a = 5.555556;
const float LogC_b = 0.047996;
const float LogC_c = 0.244161;
const float LogC_d = 0.386036;
const vec3 AP1_RGB2Y = vec3(0.272229, 0.674082, 0.0536895);
const float ACEScc_MIDGRAY = 0.4135884;
const float HALF_MAX = 65504.0;
const float HALF_MIN = 6.103515625e-5;
const float HALF_MIN_SQRT = 0.0078125;
const float RRT_GLOW_GAIN = 0.05;
const float RRT_GLOW_MID = 0.08;
const float RRT_RED_SCALE = 0.82;

const float RRT_RED_PIVOT = 0.03;

const float RRT_RED_HUE = 0.0;

const float RRT_RED_WIDTH = 135.0;

const float RRT_SAT_FACTOR = 0.96;

const vec4 kDielectricSpec = vec4(0.04, 0.04, 0.04, 1.0 - 0.04);

const float PI = 3.1415926;

/***************postprocess*********************/
const mat3 sRGB_2_AP0 = transpose(mat3(

    0.4397010, 0.3829780, 0.1773350,

    0.0897923, 0.8134230, 0.0967616,

    0.0175440, 0.1115440, 0.8707040

));

const mat3 AP0_2_AP1_MAT = transpose(mat3(

     1.4514393161, -0.2365107469, -0.2149285693,

    -0.0765537734,  1.1762296998, -0.0996759264,

     0.0083161484, -0.0060324498,  0.9977163014

));

const mat3 AP1_2_AP0_MAT = transpose(mat3(

     0.6954522414, 0.1406786965, 0.1638690622,

     0.0447945634, 0.8596711185, 0.0955343182,

    -0.0055258826, 0.0040252103, 1.0015006723

));

const mat3 XYZ_2_AP1_MAT = transpose(mat3(

     1.6410233797, -0.3248032942, -0.2364246952,

    -0.6636628587,  1.6153315917,  0.0167563477,

     0.0117218943, -0.0082844420,  0.9883948585

));

const mat3 AP1_2_XYZ_MAT = transpose(mat3(

     0.6624541811, 0.1340042065, 0.1561876870,

     0.2722287168, 0.6740817658, 0.0536895174,

    -0.0055746495, 0.0040607335, 1.0103391003

));

const mat3 D60_2_D65_CAT = transpose(mat3(

     0.98722400, -0.00611327, 0.0159533,

    -0.00759836,  1.00186000, 0.0053302,

     0.00307257, -0.00509595, 1.0816800

));

const mat3 XYZ_2_REC709_MAT = transpose(mat3(

     3.2409699419, -1.5373831776, -0.4986107603,

    -0.9692436363,  1.8759675015,  0.0415550574,

     0.0556300797, -0.2039769589,  1.0569715142

));

vec3 unity_to_ACES(vec3 x)

{

    x = sRGB_2_AP0 * x;

    return x;

}


vec3 ACES_to_ACEScg(vec3 x)
{

    return AP0_2_AP1_MAT * x;

}

vec3 ACEScg_to_ACES(vec3 x)

{

    return AP1_2_AP0_MAT * x;

}

vec3 ACES_to_ACEScc(vec3 x)

{

    x = clamp(x, 0.0, HALF_MAX);

    // x is clamped to [0, HALF_MAX], skip the <= 0 check

    vec3 result = vec3(0);
    if (x.x < 0.00003051757||x.y < 0.00003051757||x.z < 0.00003051757) {
        result = (log2(0.00001525878 + x * 0.5) + 9.72) / 17.52;
    } else {
        result = (log2(x) + 9.72) / 17.52;
    }

    return result;

}

float ACEScc_to_ACES(float x)
{

    // TODO: Optimize me

    if (x < -0.3013698630) // (9.72 - 15) / 17.52

        return (pow(2.0, x * 17.52 - 9.72) - pow(2.0, -16.0)) * 2.0;

    else if (x < (log2(HALF_MAX) + 9.72) / 17.52)

        return pow(2.0, x * 17.52 - 9.72);

    else if (x==0.0)
        return 0.0;

    else // (x >= (log2(HALF_MAX) + 9.72) / 17.52)

        return HALF_MAX;

}

vec3 ACEScc_to_ACES(vec3 x)
{

    return vec3(

        ACEScc_to_ACES(x.r),

        ACEScc_to_ACES(x.g),

        ACEScc_to_ACES(x.b)

    );

}

vec3 log10(vec3 x) {
    return vec3(
        log(x.r) / log(10.0),
        log(x.g) / log(10.0),
        log(x.b) / log(10.0)
    );
}

// 从 LogC 转线性空间
vec3 LogCToLinear(vec3 x) {
    return (vec3(
        pow(10.0, (x.r - LogC_d) / LogC_c),
        pow(10.0, (x.g - LogC_d) / LogC_c),
        pow(10.0, (x.b - LogC_d) / LogC_c)
    ) - vec3(LogC_b,LogC_b,LogC_b)) / vec3(LogC_a,LogC_a,LogC_a);
}

// 从线性空间转 LogC
vec3 LinearToLogC(vec3 x) {
    return LogC_c * log10(LogC_a * x + LogC_b) + vec3(LogC_d,LogC_d,LogC_d);
}

float Luminance(vec3 linearRgb)

{

    return clamp(dot(linearRgb, vec3(0.2126729, 0.7151522, 0.0721750)), 0, 1);

}

float AcesLuminance(vec3 linearRgb)

{

    return dot(linearRgb, AP1_RGB2Y);

}

float GetLuminance(vec3 colorLinear)

{

if(post_tonemapping)

    return AcesLuminance(colorLinear);

else

    return Luminance(colorLinear);

}

float Min3(float a, float b, float c)
{
    return min(a, min(b, c));
}

float Max3(float a, float b, float c)
{
    return max(a, max(b, c));
}

int FastSign(float value)
{
    return int(value > 0.0) - int(value < 0.0);
}

float rgb_2_saturation(vec3 rgb)

{

    const float TINY = 1e-4;

    float mi = Min3(rgb.r, rgb.g, rgb.b);

    float ma = Max3(rgb.r, rgb.g, rgb.b);

    return (max(ma, TINY) - max(mi, TINY)) / max(ma, 1e-2);

}

float rgb_2_yc(vec3 rgb)

{

    const float ycRadiusWeight = 1.75;

    float r = rgb.x;

    float g = rgb.y;

    float b = rgb.z;

    float k = b * (b - g) + g * (g - r) + r * (r - b);

    k = max(k, 0.0); // Clamp to avoid precision issue causing k < 0, making sqrt(k) undefined

    float chroma = sqrt(k);

    return (b + g + r + ycRadiusWeight * chroma) / 3.0;

}

float sigmoid_shaper(float x)

{

    // Sigmoid function in the range 0 to 1 spanning -2 to +2.



    float t = max(1.0 - abs(x / 2.0), 0.0);

    float y = 1.0 + FastSign(x) * (1.0 - t * t);



    return y / 2.0;

}

float glow_fwd(float ycIn, float glowGainIn, float glowMid)

{

    float glowGainOut;



    if (ycIn <= 2.0 / 3.0 * glowMid)

        glowGainOut = glowGainIn;

    else if (ycIn >= 2.0 * glowMid)

        glowGainOut = 0.0;

    else

        glowGainOut = glowGainIn * (glowMid / ycIn - 1.0 / 2.0);



    return glowGainOut;

}

float atan2(float y, float x)
{
    if (x > 0.0) {
        return atan(y / x);  // 第一象限或第四象限
    } else if (x < 0.0) {
        return atan(y / x) + 3.14159265358979323846;  // 第二象限或第三象限
    } else {
        return (y > 0.0) ? 1.57079632679489661923 : (y < 0.0) ? -1.57079632679489661923 : 0.0;  // x = 0
    }
}

float rgb_2_hue(vec3 rgb)

{

    // Returns a geometric hue angle in degrees (0-360) based on RGB values.

    // For neutral colors, hue is undefined and the function will return a quiet NaN value.

    float hue;

    if (rgb.x == rgb.y && rgb.y == rgb.z)

        hue = 0.0; // RGB triplets where RGB are equal have an undefined hue

    else

        hue = (180.0 / PI) * atan2(sqrt(3.0) * (rgb.y - rgb.z), 2.0 * rgb.x - rgb.y - rgb.z);



    if (hue < 0.0) hue = hue + 360.0;



    return hue;

}

float center_hue(float hue, float centerH)

{

    float hueCentered = hue - centerH;

    if (hueCentered < -180.0) hueCentered = hueCentered + 360.0;

    else if (hueCentered > 180.0) hueCentered = hueCentered - 360.0;

    return hueCentered;

}

vec3 XYZ_2_xyY(vec3 XYZ)

{

    float divisor = max(dot(XYZ, vec3(1.0)), 1e-4);

    return vec3(XYZ.xy / divisor, XYZ.y);

}



vec3 xyY_2_XYZ(vec3 xyY)

{

    float m = xyY.z / max(xyY.y, 1e-4);

    vec3 XYZ = vec3(xyY.xz, (1.0 - xyY.x - xyY.y));

    XYZ.xz *= m;

    return XYZ;

}

const float DIM_SURROUND_GAMMA = 0.9811;

const float ODT_SAT_FACTOR = 0.93;

vec3 darkSurround_to_dimSurround(vec3 linearCV)

{

    vec3 XYZ = AP1_2_XYZ_MAT * linearCV;



    vec3 xyY = XYZ_2_xyY(XYZ);

    xyY.z = clamp(xyY.z, 0.0, HALF_MAX);

    xyY.z = pow(xyY.z, DIM_SURROUND_GAMMA);

    XYZ = xyY_2_XYZ(xyY);



    return XYZ_2_AP1_MAT * XYZ;

}

vec3 ColorGrade(vec3 color)
{
    float exposure = pow(2,post_exposure);
    color *= exposure;

    float contrast = float(float(post_contrast/100.0) + 1.0);

    float saturation = float(float(post_saturation/100.0) + 1.0);

    vec3 colorLinear = LogCToLinear(color);

    vec3 a = colorLinear;

    vec3 colorLog = vec3(0.0);

    if(post_tonemapping)
    {
        colorLog = ACES_to_ACEScc(unity_to_ACES(color));
    }
    else
    {
        colorLog = LinearToLogC(color);
    }

    if(post_colorAdjustments)
    {
        colorLog = (colorLog - ACEScc_MIDGRAY) * contrast + ACEScc_MIDGRAY;
    }

    if(post_tonemapping)
    {
        colorLinear = ACES_to_ACEScg(ACEScc_to_ACES(colorLog));
    }
    else
    {
        colorLinear = LogCToLinear(colorLog);
    }

    if(post_colorAdjustments)
    {
        colorLinear = max(vec3(0.0), colorLinear);

        vec3 colorGamma = vec3(pow(colorLinear.r, 1.0 / 2.2), pow(colorLinear.g, 1.0 / 2.2), pow(colorLinear.b, 1.0 / 2.2));

        float luma = clamp(GetLuminance(clamp(colorGamma, 0, 1)), 0, 1);

        colorLinear = vec3(pow(colorGamma.r, 2.2), pow(colorGamma.g, 2.2), pow(colorGamma.b, 2.2));

        // Global saturation

        luma = GetLuminance(colorLinear);

        colorLinear = vec3(luma) + vec3(saturation) * (colorLinear - vec3(luma));
    }

    colorLinear = max(vec3(0.0), colorLinear);

    return colorLinear;

}

vec3 AcesTonemap(vec3 aces)
{
    // --- Glow module --- //
    float saturation = rgb_2_saturation(aces);
    float ycIn = rgb_2_yc(aces);
    float s = sigmoid_shaper((saturation - 0.4) / 0.2);
    float addedGlow = 1.0 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
    aces *= addedGlow; 

    // --- Red modifier --- //
    float hue = rgb_2_hue(aces);
    float centeredHue = center_hue(hue, RRT_RED_HUE);
    float hueWeight;

    hueWeight = smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centeredHue / RRT_RED_WIDTH));
    hueWeight *= hueWeight;
    //return vec3(hueWeight);
    aces.r += hueWeight * saturation * (RRT_RED_PIVOT - aces.r) * (1.0 - RRT_RED_SCALE);


    // --- ACES to RGB rendering space --- //
    vec3 acescg = vec3(max(0.0, ACES_to_ACEScg(aces).r),max(0.0, ACES_to_ACEScg(aces).g),max(0.0, ACES_to_ACEScg(aces).b));

    // --- Global desaturation --- //
    acescg = mix(vec3(dot(acescg, AP1_RGB2Y)), acescg, RRT_SAT_FACTOR);

    // Luminance fitting of *RRT.a1.0.3 + ODT.Academy.RGBmonitor_100nits_dim.a1.0.3*.
    const float a = 2.785085;
    const float b = 0.107772;
    const float c = 2.936045;
    const float d = 0.887122;
    const float e = 0.806889;

    vec3 x = acescg;
    vec3 rgbPost = (x * (a * x + b)) / (x * (c * x + d) + e);

    // Apply gamma adjustment to compensate for dim surround
    vec3 linearCV = darkSurround_to_dimSurround(rgbPost);

    // Apply desaturation to compensate for luminance difference
    linearCV = mix(vec3(dot(linearCV, AP1_RGB2Y)), linearCV, ODT_SAT_FACTOR);

    // Convert to display primary encoding
    vec3 XYZ = AP1_2_XYZ_MAT * linearCV;

    // Apply CAT from ACES white point to assumed observer adapted white point
    XYZ = D60_2_D65_CAT * XYZ;

    // CIE XYZ to display primaries
    linearCV = XYZ_2_REC709_MAT * XYZ;

    return linearCV;
}

vec3 Tonemap(vec3 colorLinear)
{
    vec3 aces = ACEScg_to_ACES(colorLinear);

    colorLinear = AcesTonemap(aces);

    return colorLinear;
}

vec3 XDTPostprocess(vec3 color)
{
    color = ColorGrade(color);
    color = Tonemap(color);
    
    return color;
}

/*************Non PBR*********/
vec3 LightingSpecular(vec3 lightColor, vec3 lightDirection, vec3 normal, vec3 viewDirection, float specularGloss, float smoothness)
{
  vec3 halfVec = normalize(viewDirection + lightDirection);
  float NdotH = clamp(dot(normal, halfVec), 0.0, 1.0);
  float smoothDelta = pow(smoothness, 6) * (1 - step(smoothness, 0.0));
  float modifier = pow(NdotH, max(1.0, smoothDelta * 2122.0)) * (1 - step(smoothness, 0.0));
  vec3 specularColor = lightColor * specularGloss * modifier * 116.0 * smoothDelta;

  return specularColor;
}

vec3 LightingLambert(vec3 lightColor, vec3 lightDir, vec3 normal)
{

    float NdotL = clamp(dot(normal, lightDir),0,1);

    return lightColor * NdotL;

}

vec3 UniversalFragmentBlinnPhong_XDTCustom(vec3 diffColor, vec3 lightDir, vec3 normal, vec3 viewDirection, float smoothness, float occlusion, float shadowFactor, vec3 ambient)
{
    float specularGloss = pow(smoothness, 3);
    vec3 cubeColor = vec3(0.4352941,0.5607843,0.6039216);//BlinnPhong_Color

    vec3 lightColor = ambient * occlusion + LightingLambert(light_color, lightDir, normal);

    // 计算 Blinn-Phong 镜面反射
    vec3 specularColor = LightingSpecular(light_color, lightDir, normal, viewDirection, specularGloss, smoothness);

    // 计算菲涅耳反射
    float NoV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    float fresnelTerm = pow((1.0 - NoV),4);
    vec3 grayScale = vec3(0.299, 0.587, 0.114);
    vec3 fresnelColor = clamp(mix(cubeColor * clamp(dot(cubeColor, grayScale) - dot(diffColor, grayScale), 0.0, 1.0) * 0.1, cubeColor, fresnelTerm) * smoothness, 0.0, 1.0);

    vec3 color = (occlusion * shadowFactor * lightColor) * diffColor  + occlusion * (specularColor + fresnelColor * 0.48);

    return color;
}

vec3 getNormal(SamplerSparse sampler, SparseCoord coord, vec3 T, vec3 B, vec3 N) 
{
    vec4 normalTex = textureSparse(normal_texture, coord);
    vec3 normal = normalTex.xyz;
    if (normalTex.a == 0.0 || normalTex.xyz == vec3(0.0)) { 
        return N; 
    } 

    normal =  normal * 2.0 - vec3(1.0); 
    normal.y *= base_normal_y_coeff; 
    normal.z = max(1.0e-16, sqrt(1.0 - clamp(dot(normal.xy, normal.xy),0.0,1.0)));
    normal = normalize(normal); 

    return normalize( 
        normal.x * T + 
        normal.y * B + 
        normal.z * N 
    ); 
}

vec3 powVec3(vec3 color, float factor)
{
    color.r = pow(color.r,factor);
    color.g = pow(color.g,factor);
    color.b = pow(color.b,factor);

    return color;
}

vec3 caculateAmbient(vec3 ogNormal)
{
    vec3 skyColor = powVec3(gradient_skyColor, 2.2);
    vec3 equatorGradientColor = powVec3(gradient_equatorColor, 2.2);
    vec3 groundColor = powVec3(gradient_groundColor, 2.2);

    equatorGradientColor = mix(0.78,0.45,abs(dot(ogNormal,vec3(0,1,0)))) * equatorGradientColor;
    skyColor = mix(0.11,0.55,clamp(dot(ogNormal,vec3(0,1,0)),0,1)) * mix(1,0,clamp(dot(ogNormal,vec3(0,-1,0)),0,1)) * skyColor;
    groundColor = mix(0.11,0.55,clamp(dot(ogNormal,vec3(0,-1,0)),0,1)) * mix(1,0,clamp(dot(ogNormal,vec3(0,1,0)),0,1)) * groundColor;

    vec3 ambient = equatorGradientColor + skyColor + groundColor; 
    return ambient;
}


/*****************PBR***********************/
struct BRDF {
    float reflectivity;        
    float grazingTerm;          
    float perceptualRoughness;
    float roughness;           
    float roughness2;           
    vec3 brdfDiffuse;        
    vec3 brdfSpecular;
    float roughness2MinusOne;        
    float normalizationTerm;   
};

float PerceptualSmoothnessToPerceptualRoughness(float perceptualRoughness)
{
    return 1.0 - perceptualRoughness;
}

float OneMinusReflectivityMetallic(float metallic)
{
    float oneMinusDielectricSpec = kDielectricSpec.a;
    return oneMinusDielectricSpec - metallic * oneMinusDielectricSpec;
}

float PerceptualRoughnessToRoughness(float perceptualRoughness)
{

    return perceptualRoughness * perceptualRoughness;

}

BRDF InitializeBRDF(float metallic, float smoothness, vec3 baseColor, float occlusion) {

    BRDF brdf;

    // 计算所需的BRDF参数
    float oneMinusReflectivity = OneMinusReflectivityMetallic(metallic);
    brdf.reflectivity = 1.0 - oneMinusReflectivity;
    brdf.grazingTerm = clamp(smoothness + brdf.reflectivity, 0.0, 1.0);

    brdf.perceptualRoughness = PerceptualSmoothnessToPerceptualRoughness(smoothness);
    brdf.roughness = max(PerceptualRoughnessToRoughness(brdf.perceptualRoughness), HALF_MIN_SQRT);
    brdf.roughness2 = max(brdf.roughness * brdf.roughness, HALF_MIN);

    brdf.brdfDiffuse = baseColor * oneMinusReflectivity;
    brdf.brdfSpecular = mix(kDielectricSpec.rgb, baseColor, metallic);

    brdf.roughness2MinusOne = brdf.roughness2 - 1.0;

    // 归一化项
    brdf.normalizationTerm = brdf.roughness * 4.0 + 2.0;

    return brdf;
}

float DirectBRDFSpecular(
    float roughness2MinusOne, 
    float roughness2,
    float normalizationTerm,
    vec3 normalWS, 
    vec3 lightDirectionWS, 
    vec3 viewDirectionWS)
{
    vec3 halfDir = normalize(lightDirectionWS + viewDirectionWS);
    float NoH = clamp(dot(normalWS, halfDir), 0.0, 1.0);
    float LoH = clamp(dot(lightDirectionWS, halfDir), 0.0, 1.0);

    float d = NoH * NoH * roughness2MinusOne + 1.00001;
    float LoH2 = LoH * LoH;

    float specularTerm = min(roughness2 / ((d * d) * max(0.1, LoH2) * normalizationTerm), 65504.0);

    return specularTerm;
}

vec3 LightingPhysicallyBased(
    vec3 diffuse, 
    vec3 specular,
    vec3 lightColor, 
    vec3 lightDirectionWS, 
    vec3 normalWS, 
    vec3 viewDirectionWS,
    float roughness2MinusOne,
    float roughness2,
    float normalizationTerm)
{
    float NdotL = clamp(dot(normalWS, lightDirectionWS), 0.0, 1.0);
    vec3 radiance = lightColor * NdotL;

    vec3 brdf = diffuse;
    
    brdf += specular * DirectBRDFSpecular(roughness2MinusOne, roughness2, normalizationTerm, normalWS, lightDirectionWS, viewDirectionWS);

    return brdf * radiance;
}

vec3 GlossyEnvironmentReflection_Matcap(vec3 crossVector, float perceptualRoughness, float occlusion,
    float matcapMultiplyStrength, float matcapMultiplyRoughness )
{
    crossVector.r *= -1.0; 

    // 计算 Matcap 贴图的 UV 坐标
    vec2 matcap_uv = (vec2(crossVector.r, crossVector.g) * 0.5) + 0.5;
    vec4 matcap_sampleUV = vec4(matcap_uv.x, matcap_uv.y, 0.0, perceptualRoughness * 16.59 * 1.1 * matcapMultiplyRoughness);

    // 采样 Matcap 贴图
    vec4 matcapCol = vec4(0.0, 0.0, 0.0, 1.0);

    matcapCol = textureLod(environment_texture, matcap_sampleUV.xy, matcap_sampleUV.w);
    matcapCol.r = pow(matcapCol.r,2.2);
    matcapCol.g = pow(matcapCol.g,2.2);
    matcapCol.b = pow(matcapCol.b,2.2);
    matcapCol *= matcapMultiplyStrength;

    vec3 irradiance = matcapCol.rgb;

    return irradiance * occlusion;
}

vec3 EnvironmentBRDFSpecular(float roughness2, vec3 specular, float grazingTerm, float fresnelTerm) {
    float surfaceReduction = 1.0 / (roughness2 + 1.0);
    return surfaceReduction * mix(specular, vec3(grazingTerm), fresnelTerm);
}

vec3 EnvironmentBRDF(float roughness2, vec3 specular, float grazingTerm, vec3 diffuse, vec3 indirectDiffuse, vec3 indirectSpecular, float fresnelTerm) {
    vec3 c = indirectDiffuse * diffuse;
    c += indirectSpecular * EnvironmentBRDFSpecular(roughness2, specular, grazingTerm, fresnelTerm);

    return c;
}

vec3 GlobalIllumination(
    float roughness2,
    vec3 specular,
    float grazingTerm,
    vec3 diffuse,
    float perceptualRoughness, 
    float occlusion,
    vec3 normalWS, vec3 viewDirectionWS,
    float matcapMultiplyStrength,
    float matcapMultiplyRoughness,
    mat4 viewMatrix,
    vec3 positionWS,
    vec3 ambient
) {
    vec3 reflectVector = vec3(0.0);
    vec3 crossVector = vec3(0.0);

    vec3 normal1 = vec3(-normalWS.x,normalWS.y,normalWS.z);
    // 计算法线空间转换
    vec3 normalVS = normalize(mat3(viewMatrix) * normal1).xyz;

    // 计算相机方向
    vec3 positionVS = (viewMatrix * vec4(positionWS,1)).xyz;
    vec3 dirToCamVS = normalize(positionVS);

    // 计算 matcap 方向
    crossVector = cross(dirToCamVS, normalVS);


    // Fresnel 计算
    float NoV = clamp(dot(normalWS, viewDirectionWS), 0.0, 1.0);
    float fresnelTerm = pow(1.0 - NoV, 4.0);

    // 计算间接光照
    vec3 indirectDiffuse = ambient;
    vec3 indirectSpecular = vec3(0.0);

    indirectSpecular = GlossyEnvironmentReflection_Matcap(
        crossVector, perceptualRoughness, occlusion, 
        matcapMultiplyStrength, matcapMultiplyRoughness
    );

    // 计算最终环境光 BRDF
    vec3 color = EnvironmentBRDF(roughness2, specular, grazingTerm, diffuse, indirectDiffuse, indirectSpecular, fresnelTerm);

    return color;
}

vec3 UniversalFragmentPBR_XDTCustom(
    float roughness2,
    vec3 specular,
    float grazingTerm,
    vec3 diffuse,
    float perceptualRoughness, 
    float occlusion,
    vec3 normalWS, vec3 viewDirectionWS,
    float matcapMultiplyStrength,
    float matcapMultiplyRoughness,
    mat4 viewMatrix,
    vec3 positionWS,
    vec3 lightDirectionWS, 
    float roughness2MinusOne,
    float normalizationTerm,
    vec3 ambient
)
{
    vec3 indirectColor = GlobalIllumination(roughness2, specular, grazingTerm, diffuse, perceptualRoughness, occlusion, normalWS, viewDirectionWS, matcap_strength, matcap_roughness, uniform_camera_view_matrix_it, positionWS, ambient);

    vec3 color = indirectColor;
    
    color += LightingPhysicallyBased(diffuse, specular, light_color, lightDirectionWS, normalWS, viewDirectionWS, roughness2MinusOne, roughness2, normalizationTerm);

    return color;
}


//- Shader entry point.
void shade(V2F inputs)
{
    // Fetch material parameters, and conversion to the specular/roughness model
    float roughness = getRoughness(roughness_tex, inputs.sparse_coord);
    float smoothness = 1 - roughness;
    vec3 baseColor = getBaseColor(basecolor_tex, inputs.sparse_coord);
    float metallic = getMetallic(metallic_tex, inputs.sparse_coord);
    float specularLevel = getSpecularLevel(specularlevel_tex, inputs.sparse_coord);

    // Get detail (ambient occlusion) and global (shadow) occlusion factors
    // separately in order to blend the bent normals properly
    float shadowFactor = getShadowFactor();
    float occlusion = getAO(inputs.sparse_coord, true, false);

    vec3 viewDir = normalize(getEyeVec(inputs.position));
    viewDir.r *= -1;

    //inputs.bitangent *= -1; 
    LocalVectors vectors = computeLocalFrame(inputs);
    computeBentNormal(vectors,inputs);
    vec3 normal = getDiffuseBentNormal(vectors);
    normal.r *= -1;
    // vec3 T = inputs.tangent;
    // T.r *= -1;
    // vec3 B = inputs.bitangent;
    // B.r *= -1;
    // vec3 N = inputs.normal;
    // N.r *= -1;

    // vec4 normalTex = textureSparse(normal_texture, inputs.sparse_coord);
    // if(normalTex.r != 0 ||normalTex.g != 0)
    //     normal = getNormal(normal_texture,inputs.sparse_coord,T,B,N);


    // Feed parameters for a physically based BRDF integration
    emissiveColorOutput(pbrComputeEmissive(emissive_tex, inputs.sparse_coord));

    // Discard current fragment on the basis of the opacity channel
    // and a user defined threshold
    alphaKill(inputs.sparse_coord);

    //计算光照向量
    float pitch = float(light_rotation_x) * 0.01745329252;
    float offset = 180.0;
    if(enable_InvertZ)
        offset = 0.0;
    float yaw = (float(light_rotation_y) + offset) * 0.01745329252;
    float dir_x = sin(yaw) * cos(pitch);
    float dir_y = sin(pitch);
    float dir_z = cos(yaw) * cos(pitch);
    vec3 lightDir = vec3(dir_x,dir_y,dir_z);

    if(!enable_metallic)
    {
        metallic = 0;
    }
    vec3 diffColor = generateDiffuseColor(baseColor, metallic);
    albedoOutput(vec3(1,1,1));

    vec3 finalColor = vec3(1,1,1);

    vec3 ogNormal = inputs.normal;
    ogNormal.r *= -1;

    vec3 ambient = caculateAmbient(ogNormal); 

    if(!enable_metallic)
    {
        finalColor = UniversalFragmentBlinnPhong_XDTCustom(diffColor, lightDir, normal, viewDir, smoothness, occlusion, shadowFactor, ambient);
    }
    else
    {
        BRDF brdfData = InitializeBRDF(metallic, smoothness, baseColor, occlusion);

        finalColor = UniversalFragmentPBR_XDTCustom(brdfData.roughness2, brdfData.brdfSpecular, brdfData.grazingTerm, diffColor, brdfData.perceptualRoughness, occlusion, normal, viewDir, matcap_strength, matcap_roughness, 
                                                uniform_camera_view_matrix_it, inputs.position, lightDir, brdfData.roughness2MinusOne, brdfData.normalizationTerm, ambient); 
    }
    
    //postprocess
    finalColor = XDTPostprocess(finalColor);



    diffuseShadingOutput(vec3(smoothness));
}
