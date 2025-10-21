class_name BuoyantBody3D extends RigidBody3D

@export var damping: float = 5.0
@export var voxel_size: float = 0.5

class BuoyancyVoxel:
	var position: Vector3
	var height: float

var _first_frame = true
var _water_surfaces: Array[OceanSurface3D] = []
var _voxels: Array[BuoyancyVoxel] = []

func _ready():
	_find_water_surfaces()
	call_deferred("_generate_voxels")
	#_generate_voxels()

func _process(delta: float):
	if _first_frame && !_water_surfaces.is_empty():
		position.y = _water_surfaces[0].get_height_at_world(global_position.x, global_position.z)
		_first_frame = false
	
	for voxel in _voxels:
		var from = global_transform * voxel.position
		var to = global_transform * (voxel.position + Vector3(0, voxel.height, 0))
		DebugDraw.draw_line_3d(from, to, Color.GREEN)

func _find_water_surfaces():
	var surfaces_nodes = get_tree().get_nodes_in_group("WaterSurface")
	if surfaces_nodes.is_empty():
		push_warning("No OceanSurface3D found in the scene!")
		return
	
	for node in surfaces_nodes:
		if node is OceanSurface3D:
			_water_surfaces.push_back(node)

func _generate_voxels():
	_voxels.clear()
	
	var aabb = $CSGCombiner3D.get_aabb()
	var width = int(aabb.size.x / voxel_size)
	var depth = int(aabb.size.z / voxel_size)
	var total_voxel_count = width * depth
	
	for i in range(width):
		for j in range(depth):
			var x = aabb.position.x + i * voxel_size + voxel_size * 0.5
			var z = aabb.position.z + j * voxel_size + voxel_size * 0.5
			
			var rqp := PhysicsRayQueryParameters3D.new()
			rqp.from = global_transform * Vector3(x, aabb.position.y - 1.0, z)
			rqp.to = global_transform * Vector3(x, aabb.position.y + aabb.size.y, z)
			
			var space_state = get_world_3d().direct_space_state
			var result = space_state.intersect_ray(rqp)
			if result:
				var voxel = BuoyancyVoxel.new()
				voxel.position = global_transform.inverse() * result.position
				voxel.height = (aabb.position.y + aabb.size.y) - voxel.position.y
				_voxels.push_back(voxel)

func _physics_process(delta: float) -> void:
	for surface in _water_surfaces:
		_process_buoyancy_forces(delta, surface)
		pass

func _process_buoyancy_forces(delta: float, surface: OceanSurface3D):
	if surface == null:
		return
	
	for voxel in _voxels:
		var world_pos = global_transform * voxel.position
		var water_height = surface.get_height_at_world(world_pos.x, world_pos.z)
		var depth = water_height - world_pos.y
		
		if depth > 0.0:
			var submerged_height = min(depth, voxel.height)
			var displaced_volume = voxel_size * voxel_size * submerged_height
			
			# Buoyant force proportional to voxel mass and displaced volume
			var buoyant_force = surface.p * displaced_volume * -get_gravity()
			
			# Damping (drag)
			var velocity_at_v = linear_velocity + angular_velocity.cross(world_pos - global_position)
			var drag = -velocity_at_v * damping * displaced_volume
			
			apply_force(buoyant_force + drag, global_transform.inverse() * world_pos)
