# Guide: Creating a Realistic Underwater Scene

This guide explains various techniques you can use to create a more realistic underwater scene for your fish simulation, beyond just a background color.

## Overview

A realistic underwater scene typically includes:
1. **Water effects** (caustics, particles, fog/volumetric lighting)
2. **Environment geometry** (seafloor, plants, rocks, bubbles)
3. **Lighting effects** (god rays, depth-based color shifts, reflections)
4. **Post-processing** (color grading, depth of field, distortion)

## 1. Shader-Based Effects (Recommended Starting Point)

### A. Underwater Fog/Volumetric Lighting

**Concept**: Light scatters in water, creating a foggy effect that increases with depth.

**Implementation**:
- Modify the fragment shader to add distance-based fog
- Use the camera's distance to objects to calculate fog density
- Blend object colors with a blue/cyan fog color based on distance

**Shader Code Example** (for `fragment_shader.glsl`):
```glsl
uniform float fog_density;
uniform vec3 fog_color;
uniform float camera_depth;

void main() {
    // Calculate distance from camera
    float distance = length(vPosition - camera_position);
    
    // Calculate fog factor (exponential fog)
    float fog_factor = exp(-fog_density * distance);
    fog_factor = clamp(fog_factor, 0.0, 1.0);
    
    // Blend object color with fog
    vec3 final_color = mix(fog_color, object_color, fog_factor);
    
    gl_FragColor = vec4(final_color, 1.0);
}
```

**In Python** (add to `interactive_cd_dual_mode.py`):
```python
# Set fog parameters
viewer_base.set_fog_density(0.05)  # Adjust for fog density
viewer_base.set_fog_color(np.array([0.1, 0.4, 0.6]))  # Deep blue
```

### B. Caustics (Light Patterns on Surfaces)

**Concept**: Light rays refracting through water create dynamic light patterns.

**Implementation Options**:
1. **Texture-based caustics**: Use animated caustic textures projected onto surfaces
2. **Procedural caustics**: Generate caustic patterns using noise functions in shaders
3. **Pre-computed caustics**: Use baked caustic maps that animate over time

**Shader Code Example**:
```glsl
uniform sampler2D caustics_texture;
uniform float time;

void main() {
    // Project caustics texture onto surface
    vec2 caustics_uv = vWorldPos.xz * 0.1 + time * 0.1;
    vec3 caustics = texture2D(caustics_texture, caustics_uv).rgb;
    
    // Add caustics to lighting
    vec3 lit_color = object_color * (ambient + diffuse * caustics);
}
```

### C. Depth-Based Color Shifting

**Concept**: Water absorbs different wavelengths at different rates (red is absorbed first).

**Implementation**:
```glsl
void main() {
    float depth = camera_depth - vPosition.y;
    
    // Red channel decreases with depth (absorption)
    float red_factor = exp(-depth * 0.1);
    float green_factor = exp(-depth * 0.05);
    float blue_factor = exp(-depth * 0.02);
    
    vec3 depth_color = vec3(red_factor, green_factor, blue_factor);
    vec3 final_color = object_color * depth_color;
}
```

## 2. Geometry-Based Effects

### A. Seafloor

**Implementation**:
- Create a large plane mesh below the fish
- Apply a sand/rock texture
- Use displacement mapping or a height field for terrain variation
- Add normal mapping for surface detail

**Python Example**:
```python
def create_seafloor(size=50.0, resolution=100):
    """Create a seafloor plane"""
    x = np.linspace(-size/2, size/2, resolution)
    z = np.linspace(-size/2, size/2, resolution)
    X, Z = np.meshgrid(x, z)
    
    # Create height field using noise
    Y = generate_height_field(X, Z)  # Your noise function
    
    # Create vertices and faces
    V = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    F = create_quad_faces(resolution, resolution)
    
    return V, F
```

### B. Underwater Plants (Seaweed/Kelp)

**Implementation**:
- Create simple mesh instances (strips or cylinders)
- Animate with simple physics (swaying motion)
- Use instanced rendering for performance
- Add vertex shader animation for movement

**Python Example**:
```python
def create_seaweed(position, height, segments=10):
    """Create a single seaweed strand"""
    vertices = []
    faces = []
    
    for i in range(segments + 1):
        t = i / segments
        # Swaying motion (sin wave)
        offset_x = 0.2 * np.sin(t * np.pi) * np.sin(time * 2.0)
        y = t * height
        x = position[0] + offset_x
        z = position[2]
        vertices.append([x, y, z])
    
    # Create faces connecting segments
    # ... (create cylinder-like mesh)
    
    return np.array(vertices), np.array(faces)
```

### C. Bubbles

**Implementation**:
- Create small sphere instances
- Animate upward movement
- Use particle system or instanced rendering
- Add transparency and reflection

**Python Example**:
```python
def create_bubbles(num_bubbles=50):
    """Create bubble particles"""
    bubbles = []
    for i in range(num_bubbles):
        bubble = {
            'position': random_position(),
            'velocity': [0, random_float(0.5, 1.5), 0],  # Upward
            'size': random_float(0.1, 0.3),
            'lifetime': random_float(5.0, 15.0)
        }
        bubbles.append(bubble)
    return bubbles
```

## 3. Lighting Effects

### A. God Rays (Light Beams)

**Concept**: Light rays visible through water, especially near the surface.

**Implementation**:
- Use volumetric ray marching in shaders
- Create light shafts from surface to depth
- Add to post-processing or as overlay

**Shader Approach**:
```glsl
uniform vec3 light_position;
uniform vec3 light_direction;

float calculate_god_rays(vec3 ray_start, vec3 ray_dir) {
    float density = 0.0;
    float step_size = 0.1;
    
    for (int i = 0; i < 64; i++) {
        vec3 pos = ray_start + ray_dir * (i * step_size);
        density += sample_volumetric_density(pos);
    }
    
    return density;
}
```

### B. Ambient Lighting

**Implementation**:
- Use ambient occlusion for depth perception
- Add rim lighting for object definition
- Implement hemisphere lighting (brighter above, darker below)

```glsl
vec3 calculate_underwater_lighting(vec3 normal, vec3 view_dir) {
    // Hemisphere lighting (brighter from above)
    float up_factor = max(0.0, dot(normal, vec3(0, 1, 0)));
    vec3 ambient = mix(dark_blue, light_blue, up_factor);
    
    // Rim lighting
    float rim = 1.0 - max(0.0, dot(normal, -view_dir));
    rim = pow(rim, 2.0);
    vec3 rim_color = rim * vec3(0.5, 0.8, 1.0);
    
    return ambient + rim_color;
}
```

## 4. Post-Processing Effects

### A. Color Grading

**Implementation**:
- Adjust color curves (more blue/cyan, less red)
- Add color lookup tables (LUTs) for cinematic look
- Increase contrast and saturation slightly

### B. Depth of Field

**Implementation**:
- Blur objects based on distance from focal point
- Use depth buffer for blur calculations
- More blur for distant objects

### C. Screen-Space Reflections

**Implementation**:
- Reflect environment in water surfaces
- Use screen-space techniques for performance
- Add distortion for water surface

## 5. Integration with Your Viewer

### Step 1: Modify Shaders

1. Locate your shader files:
   - `vertex_shader_16.glsl`
   - `fragment_shader.glsl`

2. Add uniforms for underwater effects:
```glsl
uniform float time;
uniform float fog_density;
uniform vec3 fog_color;
uniform float water_depth;
```

3. Modify fragment shader to apply effects

### Step 2: Update Viewer Code

1. Add uniform setters to viewer (C++ side):
```cpp
void set_fog_density(float density);
void set_fog_color(const Eigen::RowVector3d& color);
void set_time(float time);
```

2. Update Python bindings to expose these functions

3. Set parameters in `interactive_cd_dual_mode.py`:
```python
viewer_base.set_fog_density(0.05)
viewer_base.set_fog_color(np.array([0.1, 0.4, 0.6]))
viewer_base.set_time(current_time)
```

### Step 3: Add Geometry (Optional)

If you want to add seafloor or plants:
1. Create mesh data in Python
2. Combine with main mesh or render separately
3. Apply appropriate textures and materials

## 6. Recommended Implementation Order

1. **Start Simple**: Add fog effect to shader (easiest, biggest impact)
2. **Add Depth Color**: Modify color based on depth
3. **Add Seafloor**: Create ground plane with texture
4. **Add Caustics**: Animated light patterns
5. **Add Plants**: Simple animated geometry
6. **Add Bubbles**: Particle system
7. **Add God Rays**: Advanced lighting effect
8. **Polish**: Color grading, post-processing

## 7. Performance Considerations

- **Fog**: Very cheap, minimal performance impact
- **Caustics**: Moderate cost (texture lookup)
- **Geometry**: Depends on complexity (use LOD for distant objects)
- **God Rays**: Expensive (requires multiple samples)
- **Post-processing**: Moderate cost (full-screen pass)

## 8. Resources

- **Caustic Textures**: Search for "underwater caustics texture" online
- **Noise Functions**: Use Perlin noise or Simplex noise for procedural effects
- **Shader Tutorials**: Learn OpenGL shader programming
- **Marine Biology**: Study real underwater lighting and colors

## 9. Example: Minimal Fog Implementation

Here's a minimal example to get started:

**Fragment Shader Addition**:
```glsl
uniform float u_fog_density;
uniform vec3 u_fog_color;

void main() {
    // Your existing shader code...
    vec3 object_color = ...;
    
    // Calculate distance (you'll need to pass this from vertex shader)
    float dist = v_distance_from_camera;
    
    // Fog calculation
    float fog_factor = exp(-u_fog_density * dist);
    fog_factor = clamp(fog_factor, 0.0, 1.0);
    
    // Final color
    vec3 final_color = mix(u_fog_color, object_color, fog_factor);
    gl_FragColor = vec4(final_color, 1.0);
}
```

**Python Integration**:
```python
# In interactive_cd_dual_mode.py, add to pre_draw_callback:
def pre_draw_callback():
    # ... existing code ...
    
    # Update fog (if viewer supports it)
    if hasattr(viewer_base, 'set_fog_density'):
        viewer_base.set_fog_density(0.05)
        viewer_base.set_fog_color(np.array([0.1, 0.4, 0.6]))
```

## Next Steps

1. Start with fog effect (easiest)
2. Test and adjust parameters
3. Add more effects incrementally
4. Optimize for performance
5. Polish and fine-tune

Good luck creating your underwater scene! 🌊🐟

