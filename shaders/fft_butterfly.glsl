#[compute]
#version 450

layout(rgba32f, set = 0, binding = 0) uniform image2D out_image;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = vec4(0.1, 0.6, 1.0, 1.0); // light blue
    imageStore(out_image, pixel, color);
}