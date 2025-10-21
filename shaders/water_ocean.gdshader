shader_type spatial;
render_mode blend_mix, depth_prepass_alpha, cull_disabled, diffuse_burley, specular_schlick_ggx; // depth_draw_opaque

uniform vec4 albedo : source_color;
uniform sampler2D texture_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform ivec2 albedo_texture_size;
uniform float point_size : hint_range(0.1, 128.0, 0.1);

uniform float roughness : hint_range(0.0, 1.0);
uniform sampler2D texture_metallic : hint_default_white, filter_linear_mipmap, repeat_enable;
uniform vec4 metallic_texture_channel;
uniform sampler2D texture_roughness : hint_roughness_r, filter_linear_mipmap, repeat_enable;

uniform float specular : hint_range(0.0, 1.0, 0.01);
uniform float metallic : hint_range(0.0, 1.0, 0.01);

uniform vec3 uv1_scale;
uniform vec3 uv1_offset;
uniform vec3 uv2_scale;
uniform vec3 uv2_offset;

// --- Height-based color ---
uniform float min_height = -2.0; // deep water
uniform float max_height = 2.0;  // wave crests
uniform vec3 deep_color:source_color = vec3(0.0, 0.05, 0.2);
uniform vec3 high_color:source_color = vec3(0.0, 0.3, 0.7);
uniform vec3 foam_color:source_color = vec3(1.0, 1.0, 1.0);

// Optional depth-based alpha
uniform float min_alpha = 0.5;
uniform float max_alpha = 1.0;

varying float v_height;

void vertex() {
	UV = UV * uv1_scale.xy + uv1_offset.xy;

	// Store vertex world height
	v_height = VERTEX.y;
}

void fragment() {
	vec2 base_uv = UV;

	vec4 albedo_tex = texture(texture_albedo, base_uv);
	vec3 base_color = albedo.rgb * albedo_tex.rgb;

	// --- Height-based color (deep → high) ---
	float t = clamp((v_height - min_height) / (max_height - min_height), 0.0, 1.0);
	vec3 water_color = mix(deep_color, high_color, t);

	// --- Slope-based foam ---
	float dx = dFdx(v_height);
	float dz = dFdy(v_height);
	float slope = length(vec2(dx, dz));
	// Foam appears only on steep areas
	float foam_intensity = smoothstep(0.05, 0.2, slope);

	water_color = mix(water_color, foam_color, foam_intensity);
	ALBEDO = base_color * water_color;

	// Optional depth-based alpha
	float alpha_t = clamp((v_height - min_height)/(max_height - min_height), 0.0, 1.0);
	ALPHA = mix(max_alpha, min_alpha, alpha_t);

	float metallic_tex = dot(texture(texture_metallic, base_uv), metallic_texture_channel);
	METALLIC = metallic_tex * metallic;
	SPECULAR = specular;

	vec4 roughness_texture_channel = vec4(1.0, 0.0, 0.0, 0.0);
	float roughness_tex = dot(texture(texture_roughness, base_uv), roughness_texture_channel);
	ROUGHNESS = roughness_tex * roughness;
}
