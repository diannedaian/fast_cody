#version 150
 uniform mat4 view;
 uniform mat4 proj;
 uniform vec4 fixed_color;
 in vec3 position_eye;
 in vec3 normal_eye;
 uniform vec3 light_position_eye;
 vec3 Ls = vec3 (1, 1, 1);
 vec3 Ld = vec3 (1, 1, 1);
 vec3 La = vec3 (1, 1, 1);
 in vec4 Ksi;
 in vec4 Kdi;
 in vec4 Kai;
 in float w;
 in vec2 texcoordi;
 in vec3 v_worldPos;
 uniform sampler2D tex;
 uniform sampler2D u_causticsAtlas;
uniform float u_time;
uniform int u_numFrames;
uniform float u_frameRate;
uniform float specular_exponent;
 uniform float lighting_factor;
 uniform float texture_factor;
 uniform float matcap_factor;
 uniform float double_sided;
 out vec4 outColor;
 void main()
 {
    // Forced usage of 'w' to prevent shader linker from removing it
    float _force_link_w = length(w);

//     vec3 xTangent = dFdx( position_eye );
//     vec3 yTangent = dFdy( position_eye );
//     vec3 normal_eye = normalize( cross( xTangent, yTangent ) );
     if(matcap_factor == 1.0f)
     {
         vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
         outColor = texture(tex, uv);
    }else
    {
        // Original lighting calculation
        vec3 Ia = La * vec3(Kai);    // ambient intensity

        vec3 vector_to_light_eye = light_position_eye - position_eye;
        vec3 direction_to_light_eye = normalize (vector_to_light_eye);
        float dot_prod = dot (direction_to_light_eye, normalize(normal_eye));
        float clamped_dot_prod = abs(max (dot_prod, -double_sided));
        vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

        vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
        vec3 surface_to_viewer_eye = normalize (-position_eye);
        float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
        dot_prod_specular = float(abs(dot_prod)==dot_prod) * abs(max (dot_prod_specular, -double_sided));
        float specular_factor = pow (dot_prod_specular, specular_exponent);
        vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
        vec4 color = vec4(lighting_factor * (Is + Id) + Ia + (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);
        vec4 finalColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;

        // ANIMATED CAUSTICS: Sample caustics atlas with frame-based animation
        // Atlas is 4096x256 (16 frames of 256x256 each, arranged horizontally)
        // Use mesh texture coordinates to preserve organic caustic patterns

        // Base UV from mesh texture coordinates - this preserves the organic pattern
        // Scale to control how many times the pattern repeats across the surface
        float tileScale = 2.0;  // Higher = more repeats, lower = larger patterns (1.0-4.0 range)
        vec2 baseUV = texcoordi * tileScale;

        // Add time-based drift for flowing effect
        vec2 drift = vec2(
            u_time * 0.15,  // Horizontal drift
            u_time * 0.10   // Vertical drift
        );

        // Add subtle organic wobble for realistic water surface movement
        // Use world position for wobble variation to avoid obvious tiling
        vec2 wobble = vec2(
            sin(u_time * 0.5 + v_worldPos.x * 0.2) * 0.02,  // Subtle position-dependent wobble
            cos(u_time * 0.4 + v_worldPos.z * 0.2) * 0.02
        );

        // Combine and wrap with fract to create seamless tiling
        vec2 tiledUV = fract(baseUV + drift + wobble);

        // Select current frame (0-15) based on time for animation
        int frame = int(floor(u_time * u_frameRate)) % u_numFrames;

        // Calculate final UV coordinates
        // X: frame offset (which 256-wide slice) + tiled UV within that frame
        // Y: use tiled UV directly (atlas is full height)
        float frameOffset = float(frame) / float(u_numFrames);  // 0, 1/16, 2/16, ..., 15/16
        float frameWidth = 1.0 / float(u_numFrames);  // Width of one frame in normalized coords (1/16)
        vec2 uv = vec2(frameOffset + tiledUV.x * frameWidth, tiledUV.y);
        vec4 causticsSample = texture(u_causticsAtlas, uv);
        float C = causticsSample.r;  // Use red channel for caustics intensity

        // Apply caustics: bright areas add light, dark areas cast shadows
        // C is in range [0, 1] where 0 = black (shadow), 1 = white (bright light)

        // Bright areas: add light (multiplicative + additive for realistic light)
        float brightIntensity = max(0.0, C - 0.3);  // Only bright areas (above 0.3)
        brightIntensity = smoothstep(0.0, 1.0, brightIntensity);  // Smooth transition
        finalColor.rgb += brightIntensity * 0.5;  // Additive light in bright areas

        // Dark areas: cast shadows (reduce brightness)
        float shadowIntensity = max(0.0, 0.3 - C);  // Only dark areas (below 0.3)
        shadowIntensity = smoothstep(0.0, 1.0, shadowIntensity);  // Smooth transition
        finalColor.rgb *= 1.0 - shadowIntensity * 0.2;  // Darken shadow areas (max 20% darker)

        // Depth-based attenuation: darker at lower Y (deeper), brighter at higher Y (shallower)
        float atten = clamp(exp(-v_worldPos.y * 0.3), 0.2, 1.0);
        finalColor.rgb *= atten;

        if (fixed_color != vec4(0.0)) finalColor = fixed_color;
        outColor = finalColor;
    }
 }
