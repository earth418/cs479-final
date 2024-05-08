bl_info = {
    "name": "Natural Cities",
    "author": "Ali Hafez, Ryan Tsai",
    "version": (4, 7, 9),
    "blender": (4, 0, 0),
    "location": "View3D > Add > New City Scene",
    "description": "Adds a City Scene",
}

import bpy, random, os, sys
import numpy as np
from bpy.types import Operator
from bpy.props import FloatProperty, IntProperty, FloatVectorProperty, BoolProperty, EnumProperty, IntVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from copy import deepcopy
from mathutils import Vector, noise
from math import sin,cos,pi, ceil, floor, radians, sqrt
import random
from random import randint, uniform
from mathutils import Euler

dirp = os.path.dirname(bpy.data.filepath)
if not dirp in sys.path:
    sys.path.append(dirp)

from Delaunator import Delaunator

settings = [('0', 'Universal', 'Universal'),
            ('1', 'Terrain', 'Terrain'),
            ('2', 'City', 'City'),
            ('3', 'Clouds', 'Clouds'),
            ('4', 'Birds', 'Birds')]
            
bird_types = [('0', 'Bird', 'Bird'),
              ('1', 'Cone', 'Cone')]
              
flock_methods = [('0', 'Weighted Average', 'Weighted Average'),
                  ('1', 'Priority Arbitration', 'Priority Arbitration')]

building_types = [('0', 'Solid', 'Solid'),
                  ('1', 'Brick', 'Brick'),
                  ('2', 'Checkered', 'Checkered')]


def create_building_mat(building_type):

    #--------------------------------------------
    #  Material: Buildings Material 
    #--------------------------------------------

    mat = bpy.data.materials.new("Buildings Material")
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    nodes.clear()
    links = node_tree.links

    node = nodes.new("ShaderNodeNewGeometry")
    node.location = (-1431.0294189453125, -330.44708251953125)

    node = nodes.new("ShaderNodeSeparateXYZ")
    node.location = (-1244.4266357421875, -301.31378173828125)

    node = nodes.new("ShaderNodeMath")
    node.location = (-1043.3829345703125, -216.1114501953125)

    node = nodes.new("ShaderNodeTexCoord")
    node.location = (-975.1100463867188, 37.48240280151367)

    node = nodes.new("ShaderNodeCombineXYZ")
    node.location = (-836.08056640625, -283.8024597167969)

    node = nodes.new("ShaderNodeVectorMath")
    node.location = (-809.06884765625, 33.85584259033203)
    node.operation = 'MULTIPLY'
    node.inputs["Vector"].default_value = (0.0, 0.0, 1.0)

    node = nodes.new("ShaderNodeTexChecker")
    node.location = (-638.6355590820312, 77.3689193725586)
    node.inputs["Color2"].default_value = (0.09830456972122192, 0.0, 0.010272961109876633, 1.0)
    node.inputs["Scale"].default_value = 52.70000076293945

    node = nodes.new("ShaderNodeVectorMath")
    node.location = (-600.3733520507812, -188.1200408935547)
    node.name = "Vector Math.001"
    node.operation = 'MULTIPLY'
    node.inputs["Vector"].default_value = (0.0, 0.0, 1.0)
    node.inputs["Vector"].default_value = (1.6000003814697266, 0.30000001192092896, 1.0)

    node = nodes.new("ShaderNodeAttribute")
    node.location = (-485.87030029296875, 275.9276428222656)
    node.attribute_name = "Col"

    node = nodes.new("ShaderNodeTexBrick")
    node.location = (-348.505859375, -132.22488403320312)
    node.show_texture = True
    node.offset = 0.0
    node.squash = 0.8999999761581421
    node.inputs["Color1"].default_value = (1.0, 0.8565634489059448, 0.891800045967102, 1.0)
    node.inputs["Mortar"].default_value = (0.005384417250752449, 0.0, 0.05494202300906181, 1.0)
    node.inputs["Scale"].default_value = 12.699999809265137
    node.inputs["Mortar Size"].default_value = 0.009999999776482582

    node = nodes.new("ShaderNodeMix")
    node.location = (-158.33932495117188, 272.6905212402344)
    node.data_type = 'RGBA'
    node.blend_type = 'MULTIPLY'
    node.inputs["Factor"].default_value = 0.6600000262260437
    node.inputs["A"].enabled = True
    node.inputs["B"].enabled = True
    node.outputs["Result"].enabled = True

    node = nodes.new("ShaderNodeBsdfPrincipled")
    node.location = (20.392353057861328, 294.3239440917969)

    node = nodes.new("ShaderNodeOutputMaterial")
    node.location = (300.0, 300.0)

    #Links

    links.new(
        nodes["Principled BSDF"].outputs["BSDF"],
        nodes["Material Output"].inputs["Surface"]
        )

    links.new(
        nodes["Texture Coordinate"].outputs["Generated"],
        nodes["Vector Math"].inputs["Vector"]
        )

    links.new(
        nodes["Vector Math"].outputs["Vector"],
        nodes["Checker Texture"].inputs[0]
        )

    links.new(
        nodes["Attribute"].outputs["Color"],
        nodes["Mix"].inputs["A"]
        )

    links.new(
        nodes["Mix"].outputs["Result"],
        nodes["Principled BSDF"].inputs["Base Color"]
        )

    links.new(
        nodes["Geometry"].outputs["Position"],
        nodes["Separate XYZ"].inputs["Vector"]
        )

    links.new(
        nodes["Separate XYZ"].outputs["Z"],
        nodes["Combine XYZ"].inputs["Z"]
        )

    links.new(
        nodes["Separate XYZ"].outputs["Z"],
        nodes["Combine XYZ"].inputs["Y"]
        )

    if building_type == '1': # brick
        links.new(
            nodes["Brick Texture"].outputs["Color"],
            nodes["Mix"].inputs["B"]
        )
    elif building_type == '2': # checkered
       links.new(
           nodes["Checker Texture"].outputs["Color"],
           nodes["Mix"].inputs["B"]
           )
       
    links.new(
        nodes["Combine XYZ"].outputs["Vector"],
        nodes["Vector Math.001"].inputs["Vector"]
        )

    links.new(
        nodes["Vector Math.001"].outputs["Vector"],
        nodes["Brick Texture"].inputs["Vector"]
        )

    links.new(
        nodes["Separate XYZ"].outputs["X"],
        nodes["Math"].inputs[0]
        )

    links.new(
        nodes["Separate XYZ"].outputs["Y"],
        nodes["Math"].inputs[1]
        )

    links.new(
        nodes["Math"].outputs["Value"],
        nodes["Combine XYZ"].inputs["X"]
        )

        
    return mat

def create_terrain_mat(bricks, brick_threshold):
    
    #--------------------------------------------
    #  Material: Terrain 
    #--------------------------------------------

    mat = bpy.data.materials.new("Terrain")
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    nodes.clear()
    links = node_tree.links

    node = nodes.new("ShaderNodeNewGeometry")
    node.location = (-1057.9560546875, 238.4239044189453)

    node = nodes.new("ShaderNodeVectorMath")
    node.location = (-863.0867309570312, 241.71493530273438)
    node.operation = 'DISTANCE'
    node.inputs[1].default_value = (0.0, 0.0, 1.0)

    node = nodes.new("ShaderNodeFloatCurve")
    node.location = (-692.4407348632812, 304.6192626953125)
    map = node.mapping
    for i in range(2):
        map.curves[0].points.new(0.0, 1.0)

    map.curves[0].points.foreach_set(
           "location", [
            0.0, 0.0,
            0.33090919256210327, 0.029999997466802597,
            0.7636359930038452, 0.19500000774860382,
            1.0, 1.0
            ])

    node = nodes.new("ShaderNodeTexBrick")
    node.location = (-414.4537658691406, 12.010777473449707)
    node.show_texture = True
    node.inputs["Color1"].default_value = (0.800315260887146, 0.05254798382520676, 0.03936794027686119, 1.0) # param brick color
    node.inputs["Scale"].default_value = 77.79999542236328

    node = nodes.new("ShaderNodeMath")
    node.location = (-397.08807373046875, 446.94158935546875)
    node.operation = 'COMPARE'
    node.use_clamp = True
    node.inputs[1].default_value = 0.0
    node.inputs[2].default_value = 0.1 * brick_threshold # 0.07000000029802322 # param for path

    node = nodes.new("ShaderNodeMix")
    node.location = (-394.97613525390625, 257.9634704589844)
    node.data_type = 'RGBA'
    node.inputs["A"].default_value = (0.00044031089055351913, 0.09851755946874619, 0.0, 1.0) # grass color
    node.inputs["B"].default_value = (0.03152122348546982, 0.03152122348546982, 0.03152122348546982, 1.0) # rock color

    node = nodes.new("ShaderNodeMix")
    node.location = (-185.16549682617188, 358.50408935546875)
    node.name = "Mix.001"
    node.data_type = 'RGBA'
    node.inputs["A"].default_value = (0.00044031089055351913, 0.09851755946874619, 0.0, 1.0) 
    node.inputs["B"].default_value = (0.3970089554786682, 0.40132981538772583, 0.21035198867321014, 1.0)

    node = nodes.new("ShaderNodeBsdfPrincipled")
    node.location = (82.87584686279297, 363.02154541015625)

    node = nodes.new("ShaderNodeOutputMaterial")
    node.location = (372.8758544921875, 363.02154541015625)

    #Links

    links.new(
        nodes["Principled BSDF"].outputs["BSDF"],
        nodes["Material Output"].inputs["Surface"]
        )

    links.new(
        nodes["Geometry"].outputs["Normal"],
        nodes["Vector Math"].inputs[0]
        )

    links.new(
        nodes["Vector Math"].outputs["Value"],
        nodes["Float Curve"].inputs["Value"]
        )

    links.new(
        nodes["Float Curve"].outputs["Value"],
        nodes["Math"].inputs["Value"]
        )

    links.new(
        nodes["Math"].outputs["Value"],
        nodes["Mix.001"].inputs["Factor"]
        )

    links.new(
        nodes["Float Curve"].outputs["Value"],
        nodes["Mix"].inputs["Factor"]
        )

    links.new(
        nodes["Mix"].outputs["Result"],
        nodes["Mix.001"].inputs["A"]
        )
    
    if bricks:
        links.new(
            nodes["Brick Texture"].outputs["Color"],
            nodes["Mix.001"].inputs["B"]
            )

    links.new(
        nodes["Mix.001"].outputs["Result"],
        nodes["Principled BSDF"].inputs["Base Color"]
        )
        
    return mat

def delete_object(name):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        bpy.data.objects.remove(obj, do_unlink=True)
        
def clear_bird_fbx():
    delete_object("Armature")
    delete_object("Plane")
    for i in range(1, 5):
        delete_object("Armature.00" + str(i))
        delete_object("Plane.00" + str(i))
        
def clamp_magnitude(v, m):
    return v * (m / np.linalg.norm(v))

def fractal_noise(location: Vector, persistence, lacunarity, octaves = 5):
    
    final_noise = 0.0
    noise_mag = 1.0
    
    # persistence = 0.32
    P = persistence
    
    # lacunarity = 2.4
    L = lacunarity
    
    for i in range(octaves):
        final_noise += (noise.noise(L * location, noise_basis='PERLIN_ORIGINAL')) * P
        noise_mag += P
        
        P *= persistence
        L *= lacunarity
        
    return final_noise / noise_mag
    
class Obstacle:
    def __init__(self, pos, r):
        self.position = pos
        self.radius = r

class Boid:
    def __init__(self, pos, vel, scale, min_height, max_height, friend_r, avoid_r, max_a, max_s, fm):
        self.position = pos
        self.velocity = vel
        self.flockmates = []
        self.think_timer = randint(0, 4)
        self.scale = scale
        self.min_height = min_height
        self.max_height = max_height
        self.friend_radius = friend_r
        self.avoid_radius = avoid_r
        self.max_acceleration = max_a
        self.max_speed = max_s
        self.flock_method = fm
        
    def fly(self, boids, obstacles):
        self.increment()
        self.wrap()
        
        if self.think_timer == 0:
            self.getFlockmates(boids)
        
        self.flock(obstacles)
            
        self.position = np.add(self.position, self.velocity)
        
    def increment(self):
        self.think_timer = (self.think_timer + 1) % 5
        
    def wrap(self):
        self.position[0] = self.position[0] % self.scale
        self.position[1] = self.position[1] % self.scale
        self.position[2] = (self.position[2] - self.min_height) % (self.max_height - self.min_height) + self.min_height
        
    def getFlockmates(self, boids):
        nearby = []
        for b in boids:
            if abs(self.position[0] - b.position[0]) < self.friend_radius and abs(self.position[1] - b.position[1]) < self.friend_radius and abs(self.position[2] - b.position[2]) < self.friend_radius:
                nearby.append(b)
        self.flockmates = nearby
        
    # COMPUTE VELOCITY
    def flock(self, obstacles):
        collision = self.getAvoidBoids() + self.getAvoidBorders() + self.getAvoidObstacles(obstacles) * 3
        align = self.getAverageVelocity()
        cohese = self.getCohesion()
        
        if self.flock_method == '0':
            self.velocity = sum([self.velocity, align/3, collision/3, cohese/3])
        else:
            # PRIORITIZED ACCELERATION ARBITRATION
            avoid_magnitude = np.linalg.norm(collision)
            if avoid_magnitude > self.max_acceleration:
                self.velocity = sum([self.velocity, clamp_magnitude(collision, self.max_acceleration)])
            else:
                align_magnitude = np.linalg.norm(align)
                if avoid_magnitude + align_magnitude > self.max_acceleration:
                    self.velocity = sum([self.velocity, collision, clamp_magnitude(align, self.max_acceleration - avoid_magnitude)])
                else:
                    cohese_magnitude = np.linalg.norm(cohese)
                    if avoid_magnitude + align_magnitude + cohese_magnitude > self.max_acceleration:
                        self.velocity = sum([self.velocity, collision, align, clamp_magnitude(cohese, self.max_acceleration - avoid_magnitude - align_magnitude)])
                    else:
                        self.velocity = sum([self.velocity, collision, align, cohese])
        
        # Check speed
        speed = np.linalg.norm(self.velocity)
        
        if speed > self.max_speed:
           self.velocity = self.velocity * (self.max_speed/speed)
        
    def getAverageVelocity(self):
        steer = np.zeros(3)
        count = 0
        
        for b in self.flockmates:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < self.friend_radius:
                copy = np.copy(b.velocity)
                copy = clamp_magnitude(copy, 1/max(d, 0.00001))
                
                steer = np.add(steer, copy)
                count += 1
                
        if count > 0:
            steer /= count 
            
        return steer
    
    def getAvoidBoids(self):
        steer = np.zeros(3)
        count = 0
        
        for b in self.flockmates:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < self.avoid_radius:
                diff = np.subtract(self.position, b.position)
                diff = clamp_magnitude(diff, 1/max(d, 0.00001))
                
                steer = np.add(steer, diff)
                count += 1
                
        if count > 0:
            steer /= count 
            
        return steer
    
    def getAvoidBorders(self):
        steer = np.zeros(3)
        
        if self.position[2] - self.min_height < self.avoid_radius and self.velocity[2] < 0:
            steer += np.array([self.velocity[0], self.velocity[1], 0.1]) / max(self.position[2] - self.min_height, 0.00001)
            
        if self.max_height - self.position[2] < self.avoid_radius and self.velocity[2] > 0:
            steer += np.array([self.velocity[0], self.velocity[1], -0.1]) / max(self.max_height - self.position[2], 0.00001)
            
        return steer
    
    def getAvoidObstacles(self, obstacles):
        steer = np.zeros(3)
        
        for a in obstacles:
            d = np.linalg.norm(np.subtract(self.position, a.position)) - a.radius
            
            if d < self.avoid_radius:
                diff = np.subtract(self.position, a.position)
                diff = clamp_magnitude(diff, 1/max(d, 0.00001))
                
                steer = np.add(steer, diff)
            
        return steer
    
    def getCohesion(self):
        sum = np.zeros(3)
        count = 0
        
        for b in self.flockmates:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < self.friend_radius:
                sum = np.add(sum, b.position)
                count += 1
                
        if count > 0:
            sum /= count
            steer = np.subtract(sum, self.position)
            return clamp_magnitude(steer, 0.05)
        else:
            return np.zeros(3)

def initialize_body(name, position, velocity, size, type):
    bpy.ops.mesh.primitive_cone_add(radius1=size,
                                        depth=2*size,
                                        location=position,
                                        rotation=Euler((0,np.arccos(velocity[2]/np.linalg.norm(velocity)),np.arctan2(velocity[1], velocity[0])), 'XYZ'),
                                        scale=(1,1,1))
                                        
    if type == '0':
        bpy.context.object.data = bpy.data.objects['Plane'].data
        bpy.context.object.rotation_euler = Euler((0,np.arccos(velocity[2]/np.linalg.norm(velocity))-pi/2,np.arctan2(velocity[1], velocity[0])), 'XYZ')
        bpy.context.object.scale = (size, size, size)
                                        
    bpy.context.object.name = name
    bpy.ops.object.shade_smooth()
    return bpy.context.object
    
def generate_terrain(size, horiz_scale, height_function, bricks, brick_threshold):

    vertices = []
    triangles = []
    edges = []
    
    heightmap = []
    
    for i in range(size):
        h_row = []
        
        
        for j in range(size):
            
            x = horiz_scale * i
            y = horiz_scale * j
            z = height_function(Vector((x, y, 0.0)))

            ind = len(vertices)            
            vertices.append(Vector((x, y, z)))
            
            if i != size - 1 and j != size - 1:
                triangles.append([ind, ind + 1, ind + size + 1])
                triangles.append([ind, ind + size, ind + size + 1])
                
            h_row.append(z)
            
        heightmap.append(h_row)    
        # colors.append(get_color(lr, vert))

    mesh = bpy.data.meshes.new(name="Terrain")
    mesh.from_pydata(vertices, edges, triangles)
    
    obj = object_data_add(bpy.context, mesh)
    obj.data.materials.append(create_terrain_mat(bricks, brick_threshold))
    
    return heightmap # return a heightmap here!
    
def get_buildings(size, building_scale = 1.0, building_height = 1.0, height_var = 0.1, area_ratio = 0.8, area_var = 0.1):
    
    init_verts = []
    pts = []
    
    for i in range(floor(size) + 1):
        for j in range(floor(size) + 1):
            
            pos = Vector((i,j, 0.0))
            cell = pos + noise.cell_vector(pos)
            
            init_verts.append(cell * Vector((building_scale, building_scale, 0.0)))
            
            pts.append([cell[0], cell[1]])
        
    
    del_tris = Delaunator(pts).triangles
    
    buildings = [] # array of tuples of tris, verts
    
    for i in range(0, len(del_tris), 3):
        
        b_tris = []
        b_verts = []
        
        tri = [del_tris[i+k] for k in range(3)]
        
            
        a = init_verts[tri[0]]
        b = init_verts[tri[1]]
        c = init_verts[tri[2]]
        
        area = 0.5 * abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
        
        if area < 0.4:
            continue
        
        centroid = Vector((0.0, 0.0, 0.0))                
        
        for p in tri:
            centroid += init_verts[p] / 3
        
        if centroid.x > size or centroid.y > size or centroid.x < 0.8 or centroid.y < 0.8:
            continue
        
        new_tri1 = []
        
        d = min(1.0, area_ratio + (random.random() - 0.5) * area_var) # size of bldng + rand
        
        if d < 0.4:
            continue
    
        if d > 0.99:
            l = 0
            # can you find neighboring buildings 
            # and make them all the same color?
        
        for p in tri:
            diff = init_verts[p] - centroid
            i = len(b_verts)
            
            b_verts.append(centroid + diff * d)
            new_tri1.append(i)
            
        b_tris.append(new_tri1)
        new_tri2 = []
        
        
        h_off = height_var * noise.noise(centroid)
        
        for p in tri:
            diff = init_verts[p] - centroid
            i = len(b_verts)
            
            b_verts.append(centroid + diff * d + Vector((0.0, 0.0, building_height + h_off)))
            new_tri2.append(i)
    
        b_tris.append(new_tri2)
        
        for index in range(3):
             
            nindex = (index + 1) % 3
            b_tris.append([new_tri1[index],  new_tri2[index], new_tri1[nindex]])
            b_tris.append([new_tri1[nindex], new_tri2[index], new_tri2[nindex]])
            
        buildings.append((b_verts, b_tris))
    
    return buildings

def add_buildings(input_buildings, height_function, single_color=None, building_type='0'):

    triangles = []
    vertices = []

    for building in input_buildings:
                        
        verts = building[0]
        tris = building[1]
            
        centroid = Vector((0.0, 0.0, 0.0))
    
        for p in tris[0]:
            centroid += verts[p] / 3

        eps = Vector((0.05, 0.0, 0.0))
                    
        ground_height = height_function(centroid)
        
        gradient = Vector((height_function(centroid + eps.xyy) - ground_height,
                            height_function(centroid + eps.yxy) - ground_height, 0.0)) / eps.x
        
#            print(verts)
        
        if gradient.length <= 0.3:
            
            # spawn the actual building
            ind = len(vertices)
            triangles += [[ind + k for k in tri] for tri in tris]
            vertices += [v + Vector((0,0,ground_height)) for v in verts]
        
            
    mesh = bpy.data.meshes.new(name="Buildings")
    mesh.from_pydata(vertices, [], triangles)
    
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    
    # print(len(triangles))
    
#        for building in input_buildings:
    for i in range(0, len(triangles), 8):
        if single_color is None:
            r, g, b = [random.random() for _k in range(3)]
        else:
            r, g, b = [single_color.x, single_color.y, single_color.z]
        
        for j in range(3*8):
            mesh.vertex_colors.active.data[3*i+j].color = (r, g, b, 1.0)

    
    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    obj = object_data_add(bpy.context, mesh)
    obj.data.materials.append(create_building_mat(building_type)) # this is gonna need some params for sure
        
    return mesh

def create_clouds(self, context):
    # Set Seed
    random.seed(self.seed)
    
    NEW_SCALE = self.scale / 10
    
    # Create Metaballs + Mesh
    for i in range(self.cloudiness):
        bpy.ops.object.metaball_add(radius=uniform(self.min_cloud_radius, self.max_cloud_radius), location=(uniform(0, NEW_SCALE), uniform(0, NEW_SCALE), uniform(self.cloud_height - self.cloud_depth/2, self.cloud_height + self.cloud_depth/2)))
    
    bpy.ops.object.convert(target='MESH')
    cloud_mesh = bpy.context.object
    cloud_mesh.name = "Clouds"
    cloud_mesh.hide_viewport = True
    cloud_mesh.hide_render = True
    
    # Mesh to Volume
    bpy.ops.object.volume_add()
    mod = bpy.context.object.modifiers.new(name='MeshToVolume', type='MESH_TO_VOLUME')
    mod.density = self.cloud_density
    mod.voxel_amount = 150
    mod.object = cloud_mesh
    
    # Cloud Displacement
    tex = bpy.data.textures.new("Displace", 'CLOUDS')
    tex.noise_depth = 3
    tex.noise_scale = 0.8
    tex.noise_type = 'SOFT_NOISE'
    
    mod = bpy.context.object.modifiers.new(name='VolumeDisplace', type='VOLUME_DISPLACE')
    mod.texture = tex
    mod.strength = self.wispiness

def create_boids(self, context):
    random.seed(self.seed)
    
    bodies = []
    brains = []
    obstacles = []
    
    NEW_SCALE = self.scale / 10
    MAX_ACCELERATION = 0.5
    
    for i in range(self.num_boids):
        p = np.array([uniform(0, NEW_SCALE), uniform(0, NEW_SCALE), uniform(self.min_bird_altitude, self.max_bird_altitude)])
        v = clamp_magnitude(np.array([uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)]), self.max_speed / 2)
    
        b = initialize_body("Boid " + str(i+1), p, v, self.boid_size, self.bird_type)
    
        bodies.append(b)
        brains.append(Boid(p, v, NEW_SCALE, self.min_bird_altitude, self.max_bird_altitude, self.friend_radius, self.avoid_radius, MAX_ACCELERATION, self.max_speed, self.flock_method))
        
        b.animation_data_create()
        b.animation_data.action = bpy.data.actions.new(name="Flight")

    for t in range(self.boid_preflocking):
        for i in range(self.num_boids):
            brains[i].fly(brains, obstacles)

    for t in range(self.animation_frames+1):
        for i in range(self.num_boids):
            bodies[i].keyframe_insert(data_path="location", frame=t)
            bodies[i].keyframe_insert(data_path="rotation_euler", frame=t)
            brains[i].fly(brains, obstacles)

            bodies[i].location = brains[i].position
            if self.bird_type == '0':
                bodies[i].rotation_euler = Euler((0, np.arccos(brains[i].velocity[2] / np.linalg.norm(brains[i].velocity)) - pi/2,
                                        np.arctan2(brains[i].velocity[1], brains[i].velocity[0])), 'XYZ')
            elif self.bird_type == '1':
                bodies[i].rotation_euler = Euler((0, np.arccos(brains[i].velocity[2] / np.linalg.norm(brains[i].velocity)),
                                        np.arctan2(brains[i].velocity[1], brains[i].velocity[0])), 'XYZ')
        
def create_terrain(self, context):
    
    def terrain(location):
        
        return self.height_scale * fractal_noise(self.noise_scale * self.horiz_scale * location,
                                        self.noise_persistence, self.noise_lacunarity)
    
    generate_terrain(self.scale, self.horiz_scale, terrain, self.brick_ground, self.brick_ground_threshold)
    
    pass

def create_buildings(self, context):
    
    builds = get_buildings(self.scale * self.horiz_scale, self.building_scale, self.building_height,
            self.building_height_variation, self.building_area_ratio, self.building_area_ratio_variation)
    
    def terrain(location):
        
        return self.height_scale * fractal_noise(self.noise_scale * self.horiz_scale * location,
                                        self.noise_persistence, self.noise_lacunarity)

    
    add_buildings(builds, terrain, self.building_colors if self.one_color_buildings else None, self.building_type)

def add_city(self, context):
    if self.include_birds:
        bpy.ops.import_scene.fbx(filepath=dirp+"/bird.fbx")
        create_boids(self, context)
        clear_bird_fbx()
        
    if self.include_clouds:
        create_clouds(self, context)
        
    create_terrain(self, context)
    
    create_buildings(self, context)

class OBJECT_OT_add_city(Operator, AddObjectHelper):
    """Create a new Custom City Scene"""
    bl_idname = "object.add_city"
    bl_label = "Add Custom City Scene"
    bl_options = {'REGISTER', 'UNDO'}
    
    #------------------------------- PARAMETERS --------------------------------#
    
    chooseSet: EnumProperty(
        name='Settings',
        description='Choose the settings to modify',
        items=settings,
        default='0',
    )
    
    seed: IntProperty(
        name='Seed',
        description='Random Seed',
        default=0,
        min=0,
    )
    
    scale: IntProperty(
        name='Scale',
        description='Scale',
        default=400,
        soft_min=50,
        soft_max=1000,
    )
    
    cloudiness: IntProperty(
        name='Cloudiness',
        description='Cloudiness',
        default=150,
        min=1,
    )
    
    min_cloud_radius: FloatProperty(
        name='Min Radius',
        description='Minimum Radius of Cloud Tuft',
        default=2,
        soft_min=0,
        soft_max=4,
    )
    
    max_cloud_radius: FloatProperty(
        name='Max Radius',
        description='Maximum Radius of Cloud Tuft',
        default=4,
        soft_min=2,
        soft_max=6,
    )
    
    cloud_height: FloatProperty(
        name='Altitude',
        description='Altitude',
        default=15,
        soft_min=0,
        soft_max=100,
    )
    
    cloud_density: FloatProperty(
        name='Density',
        description='Cloud Density',
        default=0.5,
        min=0,
        max=1,
    )
    
    cloud_depth: FloatProperty(
        name='Domain Depth',
        description='Depth of Cloud Domain',
        default=4,
        soft_min=0,
        soft_max=8,
    )
    
    include_clouds: BoolProperty(
        name='Add Clouds?',
        description='Add Clouds?',
        default=True,
    )
    
    wispiness: FloatProperty(
        name='Wispiness',
        description='Wispiness',
        default=2,
        min=0,
        soft_max=5,
    )
    
    num_boids: IntProperty(
        name='Number of Birds',
        description='Number of Birds',
        default=150,
        min=1,
        soft_max=200,
    )
    
    boid_size: FloatProperty(
        name='Bird Size',
        description='Bird Size',
        default=0.2,
        soft_min=0.05,
        soft_max=1,
    )
    
    max_speed: FloatProperty(
        name='Max Speed',
        description='Max Speed',
        default=0.15,
        soft_min=0.01,
        soft_max=1,
    )
    
    friend_radius: FloatProperty(
        name='Friend Radius',
        description='Friend Radius',
        default=3,
        soft_min=1,
        soft_max=5,
    )
    
    avoid_radius: FloatProperty(
        name='Avoid Radius',
        description='Avoid Radius',
        default=0.5,
        soft_min=0.5,
        soft_max=1,
    )
    
    animation_frames: IntProperty(
        name='Animation Frames',
        description='Animation Frames',
        default=240,
        min=1,
        soft_max=360,
    )
    
    include_birds: BoolProperty(
        name='Add Birds?',
        description='Add Birds?',
        default=True,
    )
    
    min_bird_altitude: FloatProperty(
        name='Min Altitude',
        description='Min Altitude',
        default=10,
        soft_min=0,
    )
    
    max_bird_altitude: FloatProperty(
        name='Max Altitude',
        description='Max Altitude',
        default=20,
        soft_min=0,
    )
    
    bird_type: EnumProperty(
        name='Bird Type',
        items=bird_types,
        default='0',
    )
    
    boid_preflocking: IntProperty(
        name='Preflocking',
        description="Let birds flock before animation for X frames",
        default=120,
        min=0,
        soft_max=240,
    )
    
    flock_method: EnumProperty(
        name='Flocking Method',
        items=flock_methods,
        default='0',
    )
    
    # TERRAIN AND BUILDINGS
    
    height_scale: FloatProperty(
        name='Height Scale',
        description='Vertical scale of the terrain',
        default=50.0,
        soft_min=1.0,
        soft_max=200.0,
    )
    
    horiz_scale: FloatProperty(
        name='Horizontal Scale',
        description='Scale (in flat area) of the generated buildings and terrain',
        default=0.1,
        soft_min=0.001,
        soft_max=10.0,
    )
    
    noise_scale: FloatProperty(
        name='Noise Scale',
        description='How clustered together noise features are on the terrain',
        default=0.1,
        soft_min=0.005,
        soft_max=5.0,
    )
    
    noise_persistence: FloatProperty(
        name='Noise Persistence',
        description='How persistent each consecutive layer of noise is',
        default=0.4,
        soft_min=0.005,
        soft_max=1.0,
    )
    
    noise_lacunarity: FloatProperty(
        name='Noise Lacunarity',
        description='How much denser each consecutive layer of noise is',
        default=2.2,
        soft_min=0.005,
        soft_max=1.0,
    )
    
    building_height: FloatProperty(
        name='Building height',
        description='How tall each building is',
        default=1.0,
        soft_min=0.05,
        soft_max=4.0,
    )
    
    building_height_variation: FloatProperty(
        name='Building height variation',
        description='How much the height of each building varies (as a \% of height)',
        default=0.2,
        soft_min=0.05,
        soft_max=1.0,
    )
    
    building_scale: FloatProperty(
        name='Building scale',
        description='How BIG each building is',
        default=1.0,
        soft_min=0.1,
        soft_max=6.0,
    )
    
    building_area_ratio: FloatProperty(
        name='Building Area Ratio',
        description='How much of each trangular "grid" cell each building takes up',
        default=0.8,
        soft_min=0.05,
        soft_max=1.0,
    )
    
    building_area_ratio_variation: FloatProperty(
        name='Ratio variation',
        description='How much the Building area ratio varies',
        default=0.2,
        soft_min=0.05,
        soft_max=0.5,
    )
    
    building_type: EnumProperty(
        name='Building Type',
        items=building_types,
        default='0',
    )
    
    one_color_buildings: BoolProperty(
        name='One-color buildings',
        description='Whether or not buildings should be one color',
        default=False
    )
    
    brick_ground: BoolProperty(
        name='Brick ground',
        description='Whether or not the ground should be painted with brick',
        default=False
    )
    
    brick_ground_threshold : FloatProperty(
        name='Brick ground variation',
        description='How much brick should there be',
        default=0.6,
        soft_min=0.01,
        soft_max=5.0,
    )
    
    building_colors: FloatVectorProperty(
        name='Building color',
        description='Color of buildings',
        subtype="RGB",
        default=(0.0, 0.0, 0.0),
    )
    
    
    def draw(self, context):
        
        layout = self.layout
        layout.prop(self, 'chooseSet')
        
        if self.chooseSet == '0':
            box = layout.box()
            box.label(text="Universal:")
            
            row = box.row()
            row.prop(self, 'scale')
            
            row = box.row()
            row.prop(self, 'animation_frames')
            
            row = box.row()
            row.prop(self, 'seed')
        
        elif self.chooseSet == '1':
            box = layout.box()
            box.label(text="Terrain:")
            
            # row = box.row()
            # row.prop(self, 'radius')
        
        elif self.chooseSet == '2':
            box = layout.box()
            box.label(text="City:")
            
            # row = box.row()
            # row.prop(self, 'include_bottom')
        
        elif self.chooseSet == '3':
            box = layout.box()
            box.label(text="Clouds:")
            
            row = box.row()
            row.prop(self, 'include_clouds')
            
            if self.include_clouds:
            
                row = box.row()
                row.prop(self, 'cloudiness')
                
                row = box.row()
                row.prop(self, 'cloud_density')
                
                row = box.row()
                row.prop(self, 'wispiness')
            
                row = box.row()
                row.prop(self, 'min_cloud_radius')
            
                row = box.row()
                row.prop(self, 'max_cloud_radius')
                
                row = box.row()
                row.prop(self, 'cloud_height')
                
                row = box.row()
                row.prop(self, 'cloud_depth')
            
        elif self.chooseSet == '4':
            box = layout.box()
            box.label(text="Birds:")
            
            row = box.row()
            row.prop(self, 'include_birds')
            
            if self.include_birds:
                
                row = box.row()
                row.prop(self, 'bird_type')
            
                row = box.row()
                row.prop(self, 'num_boids')
            
                row = box.row()
                row.prop(self, 'boid_size')
                
                row = box.row()
                row.prop(self, 'flock_method')
            
                row = box.row()
                row.prop(self, 'max_speed')
            
                row = box.row()
                row.prop(self, 'friend_radius')
            
                row = box.row()
                row.prop(self, 'avoid_radius')
                
                row = box.row()
                row.prop(self, 'min_bird_altitude')
                
                row = box.row()
                row.prop(self, 'max_bird_altitude')
                
                row = box.row()
                row.prop(self, 'boid_preflocking')

    def execute(self, context):

        add_city(self, context)

        return {'FINISHED'}


# Registration

def add_object_button(self, context):
    self.layout.operator(
        OBJECT_OT_add_city.bl_idname,
        text="Add City Scene",
        icon='PLUGIN')

def register():
    bpy.utils.register_class(OBJECT_OT_add_city)
    bpy.types.VIEW3D_MT_add.append(add_object_button)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_add_city)
    bpy.utils.unregister_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_add.remove(add_object_button)


if __name__ == "__main__":
    register()
