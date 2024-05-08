import bpy
import numpy as np
from random import random, randint
from math import pi, ceil, floor, radians
from mathutils import Euler

class Obstacle:
    def __init__(self, pos, r):
        self.position = pos
        self.radius = r
        
class Boid:
    def __init__(self, pos, vel):
        self.position = pos
        self.move = vel
        self.friends = []
        self.think_timer = randint(0, 4)
        
    def fly(self):
        self.increment()
        self.wrap()
        
        if self.think_timer == 0:
            self.getFriends()
        
        self.flock()
            
        self.position = np.add(self.position, self.move)
        
    def increment(self):
        self.think_timer = (self.think_timer + 1) % 5
        
    def wrap(self):
        self.position[0] = (self.position[0] + SCALE/2) % SCALE - (SCALE/2)
        self.position[1] = (self.position[1] + SCALE/2) % SCALE - (SCALE/2)
        
    # COMPUTE VELOCITY
    def flock(self):
        align = self.getAverageDir()
        avoidBoids = self.getAvoidBoids()
        avoidObstacles = self.getAvoidObstacles() * 3
        cohese = self.getCohesion()
        noise = np.array([2*random() - 1, 2*random() - 1, 0]) * 0.1 if len(self.friends) > 1 else np.zeros(3)
        
        self.move = sum([self.move, align, avoidBoids, avoidObstacles, cohese, noise])
        
        n = np.linalg.norm(self.move)
        
        if n > MAX_SPEED:
            self.move = self.move * (MAX_SPEED/n)
        
    def getFriends(self):
        nearby = []
        for b in brains:
            if abs(self.position[0] - b.position[0]) < FRIEND_RADIUS and abs(self.position[1] - b.position[1]) < FRIEND_RADIUS and abs(self.position[2] - b.position[2]) < FRIEND_RADIUS:
                nearby.append(b)
        self.friends = nearby
        
    def getAverageDir(self):
        steer = np.zeros(3)
        count = 0
        
        for b in self.friends:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < FRIEND_RADIUS:
                copy = np.copy(b.move)
                copy = clamp_magnitude(copy, 1/d)
                
                steer = np.add(steer, copy)
                count += 1
                
        if count > 0:
            steer /= count 
            
        return steer
    
    def getAvoidBoids(self):
        steer = np.zeros(3)
        count = 0
        
        for b in self.friends:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < CROWD_RADIUS:
                diff = np.subtract(self.position, b.position)
                diff = clamp_magnitude(diff, 1/d)
                
                steer = np.add(steer, diff)
                count += 1
                
        if count > 0:
            steer /= count 
            
        return steer
    
    def getAvoidObstacles(self):
        steer = np.zeros(3)
        
        for a in obstacles:
            d = np.linalg.norm(np.subtract(self.position, a.position))
            
            if d < a.radius:
                diff = np.subtract(self.position, a.position)
                diff = clamp_magnitude(diff, 1/d)
                
                steer = np.add(steer, diff)
            
        return steer
    
    def getCohesion(self):
        sum = np.zeros(3)
        count = 0
        
        for b in self.friends:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < COHESION_RADIUS:
                sum = np.add(sum, b.position)
                count += 1
                
        if count > 0:
            sum /= count
            steer = np.subtract(sum, self.position)
            return clamp_magnitude(steer, 0.05)
        else:
            return np.zeros(3)

def clamp_magnitude(v, m):
    return v * (m / np.linalg.norm(v))

def initialize_body(name, position, velocity):
    b = bpy.ops.mesh.primitive_cone_add(radius1=BOID_SIZE,
                                        depth=2*BOID_SIZE,
                                        location=position,
                                        rotation=Euler((0,pi/2,np.arctan2(velocity[1], velocity[0])), 'XYZ'),
                                        scale=(1,1,1))
    bpy.context.object.name = name
    bpy.ops.object.shade_smooth()
    return bpy.context.object
    

def delete_object(name):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        bpy.data.objects.remove(obj, do_unlink=True)
    
        
bodies = []
brains = []
obstacles = []

BOID_SIZE = 0.2

SCALE = 20
MAX_SPEED = 0.1
FRIEND_RADIUS = 1
CROWD_RADIUS = 0.5
COHESION_RADIUS = FRIEND_RADIUS

START_FRAME = 1
END_FRAME = 240

NUM_BOIDS = 50

for i in range(NUM_BOIDS):
    delete_object("Boid " + str(i+1))
    
for i in range(NUM_BOIDS):
    p = np.array([SCALE * random() - SCALE/2, SCALE * random()- SCALE/2, 0.0])
    v = clamp_magnitude(np.array([random(), random(), 0.0]), MAX_SPEED / 2)
    
    b = initialize_body("Boid " + str(i+1), p, v)
    
    bodies.append(b)
    brains.append(Boid(p, v))
    
    b.animation_data_create()
    b.animation_data.action = bpy.data.actions.new(name="Flight")
    
for t in range(START_FRAME, END_FRAME+1):
    for i in range(NUM_BOIDS):
        bodies[i].keyframe_insert(data_path="location", frame=t)
        bodies[i].keyframe_insert(data_path="rotation_euler", frame=t)
        brains[i].fly()
        
        bodies[i].location = brains[i].position
        bodies[i].rotation_euler = Euler((0, pi/2, np.arctan2(brains[i].move[1], brains[i].move[0])), 'XYZ')