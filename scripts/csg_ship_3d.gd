@tool
class_name CSGShip3D extends CSGCombiner3D

@export var deck_size := Vector3(10, 2, 5):
	set(value):
		deck_size = value
		_update_ship()
	get: return deck_size

@export var bow_subdivisions := 8:
	set(value):
		bow_subdivisions = value
		_update_ship()
	get: return bow_subdivisions

@export var bow_height := 0.0:
	set(value):
		bow_height = value
		_update_ship()
	get: return bow_height

@export var stern_subdivisions := 8:
	set(value):
		stern_subdivisions = value
		_update_ship()
	get: return stern_subdivisions

@export var stern_height := 0.0:
	set(value):
		stern_height = value
		_update_ship()
	get: return stern_height

@export var hull_subdivisions := 5:
	set(value):
		hull_subdivisions = value
		_update_ship()
	get: return hull_subdivisions

@export var hull_height := 1.0:
	set(value):
		hull_height = value
		_update_ship()
	get: return hull_height
	
@export var hull_param := Vector2(5.0, 1.0):
	set(value):
		hull_param = value
		_update_ship()
	get: return hull_param

var _deck: CSGBox3D
var _stern: CSGPolygon3D
var _bow: CSGPolygon3D
var _hull: CSGPolygon3D

func _notification(what):
	if what == NOTIFICATION_ENTER_TREE:
		_build_ship()

func _build_ship() -> void:
	# remove existing children
	for c in get_children():
		remove_child(c)
		c.queue_free()

	_deck = CSGBox3D.new()
	
	_stern = CSGPolygon3D.new()
	_stern.mode = CSGPolygon3D.MODE_SPIN
	_stern.spin_degrees = 180
	_stern.rotation = Vector3(0, -PI*0.5, 0)
	
	_bow = CSGPolygon3D.new()
	_bow.mode = CSGPolygon3D.MODE_SPIN
	_bow.spin_degrees = 180
	_bow.rotation = Vector3(0, PI*0.5, 0)
	
	_hull = CSGPolygon3D.new()
	_hull.rotation = Vector3(0, PI*0.5, 0)

	add_child(_deck)
	add_child(_stern)
	add_child(_bow)
	add_child(_hull)
	
	_update_ship()

func _update_ship() -> void:
	if not is_inside_tree():
		return

	# Deck variables
	_deck.size = deck_size
	var half_deck_size = deck_size*0.5
	
	# Hull variables
	_hull.depth = deck_size.x
	_hull.position = Vector3(half_deck_size.x, -(deck_size.y + hull_height)*0.5, 0)
	var hull_polygon := PackedVector2Array()
	for i in range(hull_subdivisions):
		var x = (i - hull_subdivisions/2) / ((hull_subdivisions-1)/2.0)
		var y = (pow(abs(x), hull_param.x) + pow(abs(x), hull_param.y)) * 0.5 - 0.5
		hull_polygon.append(Vector2(x*half_deck_size.z, y*hull_height))
	_hull.polygon = hull_polygon
	
	# Stern variables
	_stern.position = Vector3(half_deck_size.x, 0, 0)
	var stern_total_height = half_deck_size.y + stern_height
	var stern_polygon := PackedVector2Array()
	stern_polygon.append(Vector2(0.0, stern_total_height))
	stern_polygon.append(Vector2(half_deck_size.z, stern_total_height))
	stern_polygon.append(Vector2(half_deck_size.z,-half_deck_size.y))
	for i in range((hull_subdivisions/2)+1):
		var p = hull_polygon[i]
		stern_polygon.append(Vector2(-p.x, _hull.position.y+p.y))
	_stern.polygon = stern_polygon
	_stern.spin_sides = stern_subdivisions
	
	# Bow variables
	_bow.position = Vector3(-half_deck_size.x, 0, 0)
	var bow_total_height = half_deck_size.y + bow_height
	var bow_polygon := PackedVector2Array()
	bow_polygon.append(Vector2(0.0, bow_total_height))
	bow_polygon.append(Vector2(half_deck_size.z, bow_total_height))
	bow_polygon.append(Vector2(half_deck_size.z,-half_deck_size.y))
	for i in range((hull_subdivisions/2)+1):
		var p = hull_polygon[i]
		bow_polygon.append(Vector2(-p.x, _hull.position.y+p.y))
	_bow.polygon = bow_polygon
	_bow.spin_sides = bow_subdivisions

	update_gizmos()
