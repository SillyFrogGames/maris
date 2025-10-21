class_name OceanSurface3D extends Node3D

@export_category("Simulation")
@export var N := 128 # grid size (power of two). Higher = slower, more detail
@export var L := 200.0 # patch length in world units
@export var A := 1e-2 # amplitude constant for spectrum
@export var wind_vel := Vector2(20.0,0.0)
@export var g := 9.81
@export var p := 997

@export var recompute := true
@export var enable_debug_draw: bool:
	set(value):
		_enable_debug_draw = value
		# Ensure the node is in the scene tree
		if is_inside_tree():
			get_viewport().debug_draw = Viewport.DEBUG_DRAW_WIREFRAME if value else Viewport.DEBUG_DRAW_DISABLED
	get:
		return _enable_debug_draw

# Backing variable to store the value
var _enable_debug_draw: bool = false

# Mesh / visualization params
@export var mesh_instance_path: NodePath
var mesh_instance: MeshInstance3D
var array_mesh: ArrayMesh

var _time = 0.0

# Functions
func _ready():
	add_to_group("WaterSurface")
	
	if enable_debug_draw:
		get_viewport().debug_draw = Viewport.DEBUG_DRAW_WIREFRAME
	
	if mesh_instance_path:
		mesh_instance = get_node(mesh_instance_path)
	else:
		mesh_instance = $MeshInstance3D if has_node("MeshInstance3D") else null

	assert(is_power_of_two(N), "N must be a power of two for this FFT implementation")
	
	_init_gpu_resources()
	_init_cpu_resources()

func _process(delta: float):
	if enable_debug_draw:
		get_viewport().debug_draw = Viewport.DEBUG_DRAW_WIREFRAME
	else:
		get_viewport().debug_draw = Viewport.DEBUG_DRAW_DISABLED
	
	_time += delta
	_compute_spectrum()
	simulate_frame(_time, delta)

func is_power_of_two(x: int) -> bool:
	return x > 0 and (x & (x - 1)) == 0

# Helper to read height safely
func _get_height_at(i, j):
	return _height_field[j * N + i] if _height_field else 0.0

func get_height_at_world(x: float, z: float) -> float:
	# Grid and world parameters
	var N = self.N
	var L = self.L
	var half_L = L * 0.5
	
	if _height_field == null:
		return 0.0
	
	# Convert world position -> grid space [0, N)
	# Assuming the grid is centered at the origin (-L/2 to +L/2)
	var fx = (x + half_L) / L * N
	var fz = (z + half_L) / L * N
	
	# Get integer indices for the cell
	var i0 = floor(fx)
	var j0 = floor(fz)
	var i1 = i0 + 1
	var j1 = j0 + 1
	
	# Clamp to valid range
	if i0 < 0 or j0 < 0 or i1 >= N or j1 >= N:
		return 0.0
	
	# Fractional part (for interpolation)
	var tx = fx - i0
	var tz = fz - j0
	
	# Sample four corners
	var h00 = _get_height_at(i0, j0)
	var h10 = _get_height_at(i1, j0)
	var h01 = _get_height_at(i0, j1)
	var h11 = _get_height_at(i1, j1)
	
	# Bilinear interpolation
	var h0_c = lerp(h00, h10, tx)
	var h1 = lerp(h01, h11, tx)
	return lerp(h0_c, h1, tz)

#region CPU Simulation

# Internal arrays: complex arrays are represented as arrays of complex numbers [re, im]
var h0_c: PackedVector2Array
var h_c: PackedVector2Array
var u_c: PackedVector2Array
var v_c: PackedVector2Array

# Precomputed k vectors and omegas
var kx: PackedFloat32Array
var ky: PackedFloat32Array
var k_len: PackedFloat32Array
var omega: PackedFloat32Array

var _height_field: PackedFloat32Array
var _vel_field: PackedVector2Array

func _init_cpu_resources():
	_alloc_arrays()
	_build_mesh()

func _alloc_arrays():
	var size = N * N
	h0_c = PackedVector2Array()
	h_c = PackedVector2Array()
	u_c = PackedVector2Array()
	v_c = PackedVector2Array()
	kx = PackedFloat32Array()
	ky = PackedFloat32Array()
	k_len = PackedFloat32Array()
	omega = PackedFloat32Array()
	_height_field = PackedFloat32Array()
	_vel_field = PackedVector2Array()

	h0_c.resize(size)
	h_c.resize(size)
	u_c.resize(size)
	v_c.resize(size)
	kx.resize(size)
	ky.resize(size)
	k_len.resize(size)
	omega.resize(size)
	_height_field.resize(size)
	_vel_field.resize(size)

func _k_vector(i:int, j:int) -> Vector2:
	# Map grid indices to wavevector components (centered at 0)
	var dk = (2.0 * PI) / L
	var kx_i = i * dk if (i <= N/2) else (i - N) * dk
	var ky_j = j * dk if (j <= N/2) else (j - N) * dk
	return Vector2(kx_i, ky_j)

func _dispersion(k_mag: float) -> float:
	if k_mag == 0.0:
		return 0.0
	return sqrt(g * k_mag)

func _phillips_spectrum(k_vec: Vector2, wv: Vector2) -> float:
	# Basic Phillips spectrum with wind alignment
	var k = k_vec.length()
	if k < 1e-6:
		return 0.0
	var Lw = wv.dot(wv) / g
	var k_dot_w = k_vec.normalized().dot(wv.normalized())
	var kdotw2 = k_dot_w * k_dot_w
	var ph = A * exp(-1.0 / (k * Lw * k * Lw)) / (k * k * k * k) * kdotw2
	# damp small waves
	var lambda = 0.0001
	ph *= exp(-k * k * lambda * lambda)
	return ph

func _compute_spectrum():
	if !recompute: return
	recompute = false
	
	# create initial random spectrum h0_c(k) using Phillips spectrum
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	var halfN = N / 2
	for j in range(N):
		for i in range(N):
			var idx = j * N + i
			var k_vec = _k_vector(i, j)
			kx[idx] = k_vec.x
			ky[idx] = k_vec.y
			var k_mag = k_vec.length()
			k_len[idx] = k_mag
			omega[idx] = _dispersion(k_mag)

			if k_mag < 1e-6:
				h0_c[idx] = Vector2.ZERO
			else:
				var P = _phillips_spectrum(k_vec, wind_vel)
				# gaussian random complex with variance P/2 per component
				var r1 = rng.randf_range(-1.0, 1.0)
				var r2 = rng.randf_range(-1.0, 1.0)
				var scale = sqrt(P) * 0.70710678118 # 1/sqrt(2) to distribute correctly
				h0_c[idx] = Vector2(r1, r2) * scale

func simulate_frame(time: float, delta: float) -> void:
	# Build h_c(k, t) = h0_c(k) * exp(i * omega * t) + conj(h0_c(-k)) * exp(-i * omega * t)
	var size = N * N
	for j in range(N):
		for i in range(N):
			var idx = j * N + i
			var w = omega[idx]
			var coswt = cos(w * time)
			var sinwt = sin(w * time)
			# h_c(k, t) = h0_c * e^{iwt} + h0_- * e^{-iwt}
			var re = h0_c[idx].x * coswt - h0_c[idx].y * sinwt
			var im = h0_c[idx].x * sinwt + h0_c[idx].y * coswt
			# compute conjugate partner index (-k)
			var i_neg = 0 if (i == 0) else (N - i)
			var j_neg = 0 if (j == 0) else (N - j)
			var idx_neg = j_neg * N + i_neg
			var h0m_re = h0_c[idx_neg].x
			var h0m_im = -h0_c[idx_neg].y # conjugate
			var re2 = h0m_re * coswt - h0m_im * -sinwt
			var im2 = h0m_re * -sinwt + h0m_im * coswt
			h_c[idx] = Vector2(re + re2, im + im2)

			# compute horizontal velocity spectral components from h_k
			# Using linear theory: phi_k = -i g / omega * eta_k ; u_k = i k_x * phi_k
			var kxv = kx[idx]
			var kyv = ky[idx]
			var kmag = k_len[idx]
			if w == 0.0 or kmag == 0.0:
				u_c[idx] = Vector2.ZERO
				v_c[idx] = Vector2.ZERO
			else:
				# phi_k = -i * g / w * (h_re + i h_im) = (g/w) * (h_im - i h_re)
				var phi_re = (g / w) * h_c[idx].y
				var phi_im = -(g / w) * h_c[idx].x
				# u_k = i * kx * phi_k => multiply by i: ( -phi_im, phi_re ) * kx
				u_c[idx] = Vector2(-phi_im, phi_re) * kxv
				v_c[idx] = Vector2(-phi_im, phi_re) * kyv

	# inverse FFTs to spatial domain
	_height_field = _ifft2d_real_from_complex(h_c)
	var vel_u_field = _ifft2d_real_from_complex(u_c)
	var vel_v_field = _ifft2d_real_from_complex(v_c)
	
	for i in range(size):
		var c_vel = Vector2.ZERO#current_vel#Vector2(randf_range(-1000, 1000), randf_range(-1000, 1000))#
		_vel_field[i] = Vector2(vel_u_field[i], vel_v_field[i]) + c_vel
	
	#var dx = L / float(N)
	#for j in range(N):
		#for i in range(N):
			#var idx = j * N + i
#
			## compute local divergence using finite differences
			#var u_r = _vel_field[j * N + min(i+1, N-1)].x
			#var u_l = _vel_field[j * N + max(i-1, 0)].x
			#var v_u = _vel_field[min(j+1, N-1) * N + i].y
			#var v_d = _vel_field[max(j-1, 0) * N + i].y
			#
			#var dudx = (u_r - u_l) * 0.5 / dx
			#var dvdy = (v_u - v_d) * 0.5 / dx
#
			#var divergence = dudx + dvdy
#
			## update height based on continuity
			#_height_field[idx] -= divergence * delta

	# Use height_field and velocity fields to update mesh vertices
	_update_mesh(_height_field)

# -------------------- Simple 2D inverse FFT wrapper --------------------
# For clarity we implement a straightforward 1D Cooley-Tukey FFT and use it
# row/column-wise for 2D. This is O(N^2 log N) and written in GDScript —
# slow but easy to understand. Replace with a C module or GPU FFT for real-time.

func _ifft2d_real_from_complex(c_in: PackedVector2Array) -> PackedFloat32Array:
	# perform inverse FFT: we treat arrays of size N*N
	var tmp_c = c_in.duplicate()

	# inverse FFT on rows
	for y in range(N):
		var row_c = PackedVector2Array()
		row_c.resize(N);
		for x in range(N):
			var idx = y * N + x
			row_c[x].x = tmp_c[idx].x
			row_c[x].y = tmp_c[idx].y
		var or_c = _ifft_1d_unscaled(row_c)
		for x in range(N):
			var idx = y * N + x
			tmp_c[idx] = or_c[x]

	# inverse FFT on columns
	for x in range(N):
		var col_c = PackedVector2Array()
		col_c.resize(N);
		for y in range(N):
			var idx = y * N + x
			col_c[y] = tmp_c[idx]
		var oc_c = _ifft_1d_unscaled(col_c)
		for y in range(N):
			var idx = y * N + x
			tmp_c[idx] = oc_c[y]

	# after inverse FFT, output real part scaled by 1/(N*N)
	var out = PackedFloat32Array()
	out.resize(N * N)
	var scale = 1.0 / float(N * N)
	for i in range(N * N):
		out[i] = tmp_c[i].x * scale
	return out

func _ifft_1d_unscaled(c_in: PackedVector2Array) -> PackedVector2Array:
	# naive plan: perform forward fft with conjugation trick
	# inverse FFT(x) = conj( FFT(conj(x)) ) / N
	var c = c_in.duplicate()
	
	# Conjugate
	for i in c:
		i.y = -i.y
	var fc = _fft_1d(c)
	
	# conjugate and scale by N will be done by caller
	for i in fc:
		i.y = -i.y
	return fc

func _fft_1d(c_in: PackedVector2Array) -> PackedVector2Array:
	# Iterative Cooley-Tukey radix-2 FFT
	var n = c_in.size()
	assert(is_power_of_two(n), "fft length must be power of two")
	var c = c_in.duplicate()
	
	# bit reversal
	var j = 0
	for i in range(1, n):
		var bit = n >> 1
		while j & bit:
			j = j ^ bit
			bit = bit >> 1
		j = j ^ bit
		if i < j:
			var tmp = c[i];
			c[i] = c[j];
			c[j] = tmp
	# FFT
	var len = 2
	while len <= n:
		var ang = 2.0 * PI / len
		var wlen = Vector2(cos(ang), sin(ang))
		for i in range(0, n, len):
			var w = Vector2(1.0, 0.0)
			for k in range(i, i + len/2):
				var e = c[k]
				var o = c[k + len/2]
				# v = o * w
				var v = Vector2(
					o.x * w.x - o.y * w.y,
					o.x * w.y + o.y * w.x)
				c[k] = e + v
				c[k + len/2] = e - v
				# update w = w * wlen
				w = Vector2(
					w.x * wlen.x - w.y * wlen.y,
					w.x * wlen.y + w.y * wlen.x)
		len <<= 1
	return c

# -------------------- Mesh building & updating --------------------
func _build_mesh():
	array_mesh = ArrayMesh.new()
	if mesh_instance:
		mesh_instance.mesh = array_mesh
	_create_flat_grid_mesh()

func _create_flat_grid_mesh():
	# create a flat N x N grid mesh (XZ plane). Vertices stored in vertex array
	var st = Array()
	st.resize(ArrayMesh.ARRAY_MAX)
	var verts = PackedVector3Array()
	var normals = PackedVector3Array()
	var uvs = PackedVector2Array()
	var indices = PackedInt32Array()
	verts.resize(N * N)
	normals.resize(N * N)
	uvs.resize(N * N)

	for j in range(N):
		for i in range(N):
			var idx = j * N + i
			var x = (float(i) / float(N - 1) - 0.5) * L
			var z = (float(j) / float(N - 1) - 0.5) * L
			verts[idx] = Vector3(x, 0.0, z)
			normals[idx] = Vector3(0,1,0)
			uvs[idx] = Vector2(float(i) / float(N - 1), float(j) / float(N - 1))

	# indices for triangles
	for j in range(N - 1):
		for i in range(N - 1):
			var a = j * N + i
			var b = j * N + (i + 1)
			var c = (j + 1) * N + i
			var d = (j + 1) * N + (i + 1)
			# two triangles (a,b,c) and (b,d,c)
			indices.append(a); indices.append(b); indices.append(c)
			indices.append(b); indices.append(d); indices.append(c)

	st[ArrayMesh.ARRAY_VERTEX] = verts
	st[ArrayMesh.ARRAY_NORMAL] = normals
	st[ArrayMesh.ARRAY_TEX_UV] = uvs
	st[ArrayMesh.ARRAY_INDEX] = indices
	array_mesh.clear_surfaces()
	array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, st)

func _update_mesh(height_field: PackedFloat32Array):
	if not array_mesh:
		return
	
	var surf = array_mesh.surface_get_arrays(0)
	var verts: PackedVector3Array = surf[ArrayMesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = surf[ArrayMesh.ARRAY_NORMAL]
	
	# --- Update vertex heights ---
	for j in range(N):
		for i in range(N):
			var idx = j * N + i
			var v_c = verts[idx]
			v_c.y = height_field[idx]
			verts[idx] = v_c

	# --- Recompute normals ---
	#for j in range(N):
		#for i in range(N):
			#var idx = j * N + i
			#var hL = height_field[j * N + max(i - 1, 0)]
			#var hR = height_field[j * N + min(i + 1, N - 1)]
			#var hD = height_field[max(j - 1, 0) * N + i]
			#var hU = height_field[min(j + 1, N - 1) * N + i]
#
			## finite difference normal
			#var dx = (hR - hL) * (L / float(N))
			#var dz = (hU - hD) * (L / float(N))
			#var n = Vector3(-dx, 2.0, -dz).normalized()
			#normals[idx] = n
	
	# --- Update mesh ---
	#surf[ArrayMesh.ARRAY_VERTEX] = verts
	#surf[ArrayMesh.ARRAY_NORMAL] = normals
	#array_mesh.clear_surfaces()
	#array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surf)
	
	array_mesh.surface_update_vertex_region(ArrayMesh.ARRAY_VERTEX, 0, verts.to_byte_array())

#endregion

#region GPU Simulation

@export var fft_butterfly_shader: Resource

var _rd: RenderingDevice

var _fft_butterfly_pipeline: RID
var _uniform_set: RID
var _texture: RID

func _init_gpu_resources():
	_rd = RenderingServer.create_local_rendering_device()
	
	var fft_butterfly_shader = _load_shader(fft_butterfly_shader.resource_path)
	_fft_butterfly_pipeline = _rd.compute_pipeline_create(fft_butterfly_shader)

func _load_shader(resource_path: String) -> RID:
	var shader_file := load(resource_path)
	return _rd.shader_create_from_spirv(shader_file.get_spirv())

func _create_texture(dimensions: Vector2i, format: RenderingDevice.DataFormat, usage:=0, num_layers:=1, view:=RDTextureView.new(), data: PackedByteArray=[]) -> RID:
	var texture_format := RDTextureFormat.new()
	texture_format.array_layers = num_layers
	texture_format.format = format
	texture_format.width = dimensions.x
	texture_format.height = dimensions.y
	texture_format.texture_type = RenderingDevice.TEXTURE_TYPE_2D if num_layers == 1 else RenderingDevice.TEXTURE_TYPE_2D_ARRAY
	texture_format.usage_bits = usage
	return _rd.texture_create(texture_format, view, data)

func create_storage_buffer(size: int, data: PackedByteArray=[], usage:=0) -> RID:
	if size > len(data):
		var padding := PackedByteArray(); padding.resize(size - len(data))
		data += padding
	return _rd.storage_buffer_create(max(size, len(data)), data, usage)

func create_uniform_buffer(size: int, data: PackedByteArray=[]) -> RID:
	size = max(16, size)
	if size > len(data):
		var padding := PackedByteArray(); padding.resize(size - len(data))
		data += padding
	return _rd.uniform_buffer_create(max(size, len(data)), data)

#endregion
