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
        // Integrate caustics as brightness multiplier for underwater shimmering effect
        vec2 localUV = v_worldPos.xz * 0.15;

        // Make caustics drift slowly over time
        // TUNE THESE VALUES to adjust drift speed:
        // - Increase values (e.g., 0.5, 1.0) for faster/more visible drift
        // - Decrease values (e.g., 0.05, 0.03) for slower/subtle drift
        localUV += vec2(
            u_time * 0.30,     // horizontal flow (was 0.10 - increased for visibility)
            u_time * 0.20      // vertical flow (was 0.07 - increased for visibility)
        );

        // Add gentle wobble for more realistic underwater light (simulates water surface refraction)
        // TUNE THESE VALUES to adjust wobble:
        // - Increase amplitude (0.03 -> 0.1) for more wobble
        // - Increase frequency (0.6 -> 1.2) for faster wobble
        localUV += vec2(
            sin(u_time * 0.8) * 0.08,   // horizontal wobble (was 0.6 * 0.03 - increased)
            cos(u_time * 0.6) * 0.08   // vertical wobble (was 0.4 * 0.03 - increased)
        );

        int frame = int(floor(u_time * u_frameRate)) % u_numFrames;
        float frameU = (localUV.x + float(frame)) / float(u_numFrames);
        vec2 uv = vec2(frameU, localUV.y);
        float C = texture(u_causticsAtlas, uv).r;

        // Apply caustics to brighten surfaces (underwater shimmering patches)
        finalColor.rgb *= 1.0 + C * 0.45;

        // Depth-based attenuation: darker at lower Y (deeper), brighter at higher Y (shallower)
        float atten = clamp(exp(-v_worldPos.y * 0.3), 0.2, 1.0);
        finalColor.rgb *= atten;

        if (fixed_color != vec4(0.0)) finalColor = fixed_color;
        outColor = finalColor;
    }
 }
