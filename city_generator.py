bl_info = {
    "name": "Natural Cities",
    "author": "Ali Hafez, Ryan Tsai",
    "version": (4, 7, 9),
    "blender": (4, 0, 0),
    "location": "View3D > Add > New City Scene",
    "description": "Adds a City Scene",
}

import bpy, random
import numpy as np
from bpy.types import Operator
from bpy.props import FloatProperty, IntProperty, FloatVectorProperty, BoolProperty, EnumProperty, IntVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from copy import deepcopy
from mathutils import Vector
from math import sin,cos,pi, ceil, floor, radians
from random import randint, uniform
from mathutils import Euler

settings = [('0', 'Universal', 'Universal'),
            ('1', 'Terrain', 'Terrain'),
            ('2', 'City', 'City'),
            ('3', 'Clouds', 'Clouds'),
            ('4', 'Birds', 'Birds')]
            
bird_types = [('0', 'Bird', 'Bird'),
              ('1', 'Cone', 'Cone')]
            
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
            
class Obstacle:
    def __init__(self, pos, r):
        self.position = pos
        self.radius = r

class Boid:
    def __init__(self, pos, vel, scale, min_height, max_height, friend_r, avoid_r, max_a, max_s):
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
        self.position[0] = (self.position[0] + self.scale) % (2*self.scale) - self.scale
        self.position[1] = (self.position[1] + self.scale) % (2*self.scale) - self.scale
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
        noise = np.array([uniform(-1,1), uniform(-1,1), uniform(-1,1)]) * 0.1 if len(self.flockmates) > 1 else np.zeros(3)
        
        self.velocity = sum([self.velocity, align, collision, cohese, noise])
        
        '''
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
        '''
        
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
            
def create_clouds(self, context):
    # Set Seed
    random.seed(self.seed)
    
    # Create Metaballs + Mesh
    for i in range(self.cloudiness):
        bpy.ops.object.metaball_add(radius=uniform(self.min_cloud_radius, self.max_cloud_radius), location=(uniform(-self.scale, self.scale), uniform(-self.scale, self.scale), uniform(self.cloud_height - self.cloud_depth/2, self.cloud_height + self.cloud_depth/2)))
    
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

    MAX_ACCELERATION = 0.5
    
    for i in range(self.num_boids):
        p = np.array([uniform(-self.scale, self.scale), uniform(-self.scale, self.scale), uniform(self.min_bird_altitude, self.max_bird_altitude)])
        v = clamp_magnitude(np.array([uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)]), self.max_speed / 2)
    
        b = initialize_body("Boid " + str(i+1), p, v, self.boid_size, self.bird_type)
    
        bodies.append(b)
        brains.append(Boid(p, v, self.scale, self.min_bird_altitude, self.max_bird_altitude, self.friend_radius, self.avoid_radius, MAX_ACCELERATION, self.max_speed))
        
        b.animation_data_create()
        b.animation_data.action = bpy.data.actions.new(name="Flight")

    for t in range(60):
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
                                        

def add_city(self, context):
    if self.include_birds:
        bpy.ops.import_scene.fbx(filepath="/Users/ryantsai/Desktop/Yale/CPSC479/Boids_Clouds/bird.fbx")
        create_boids(self, context)
        clear_bird_fbx()
        
    if self.include_clouds:
        create_clouds(self, context)

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
        default=10,
        soft_min=5,
        soft_max=20,
    )
    
    cloudiness: IntProperty(
        name='Cloudiness',
        description='Cloudiness',
        default=75,
        min=1,
        max=100,
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
        default=10,
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
        default=50,
        min=1,
        soft_max=100,
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
        default=2,
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
        default=5,
        soft_min=0,
    )
    
    max_bird_altitude: FloatProperty(
        name='Max Altitude',
        description='Max Altitude',
        default=15,
        soft_min=0,
    )
    
    bird_type: EnumProperty(
        name='Bird Type',
        items=bird_types,
        default='0',
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
                row.prop(self, 'max_speed')
            
                row = box.row()
                row.prop(self, 'friend_radius')
            
                row = box.row()
                row.prop(self, 'avoid_radius')
                
                row = box.row()
                row.prop(self, 'min_bird_altitude')
                
                row = box.row()
                row.prop(self, 'max_bird_altitude')

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
