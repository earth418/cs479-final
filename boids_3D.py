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
        self.velocity = vel
        self.flockmates = []
        self.think_timer = randint(0, 4)
        
    def fly(self):
        self.increment()
        self.wrap()
        
        if self.think_timer == 0:
            self.getFlockmates()
        
        self.flock()
            
        self.position = np.add(self.position, self.velocity)
        
    def increment(self):
        self.think_timer = (self.think_timer + 1) % 5
        
    def wrap(self):
        self.position[0] = (self.position[0] + SCALE/2) % SCALE - (SCALE/2)
        self.position[1] = (self.position[1] + SCALE/2) % SCALE - (SCALE/2)
        self.position[2] = (self.position[2] + HEIGHT/2) % HEIGHT - (HEIGHT/2)
        
    def getFlockmates(self):
        nearby = []
        for b in brains:
            if abs(self.position[0] - b.position[0]) < FRIEND_RADIUS and abs(self.position[1] - b.position[1]) < FRIEND_RADIUS and abs(self.position[2] - b.position[2]) < FRIEND_RADIUS:
                nearby.append(b)
        self.flockmates = nearby
        
    # COMPUTE VELOCITY
    def flock(self):
        collision = self.getAvoidBoids() + self.getAvoidBorders() + self.getAvoidObstacles() * 3
        align = self.getAverageVelocity()
        cohese = self.getCohesion()
        noise = np.array([2*random() - 1, 2*random() - 1, 2*random() - 1]) * 0.1 if len(self.flockmates) > 1 else np.zeros(3)
        
        self.velocity = sum([self.velocity, align, collision, cohese, noise])
        
        '''
        # PRIORITIZED ACCELERATION ARBITRATION
        avoid_magnitude = np.linalg.norm(collision)
        if avoid_magnitude > MAX_ACCELERATION:
            self.velocity = sum([self.velocity, clamp_magnitude(collision, MAX_ACCELERATION)])
        else:
            align_magnitude = np.linalg.norm(align)
            if avoid_magnitude + align_magnitude > MAX_ACCELERATION:
                self.velocity = sum([self.velocity, collision, clamp_magnitude(align, MAX_ACCELERATION - avoid_magnitude)])
            else:
                cohese_magnitude = np.linalg.norm(cohese)
                if avoid_magnitude + align_magnitude + cohese_magnitude > MAX_ACCELERATION:
                    self.velocity = sum([self.velocity, collision, align, clamp_magnitude(cohese, MAX_ACCELERATION - avoid_magnitude - align_magnitude)])
                else:
                    self.velocity = sum([self.velocity, collision, align, cohese])
        '''
        
        # Check speed
        speed = np.linalg.norm(self.velocity)
        
        if speed > MAX_SPEED:
           self.velocity = self.velocity * (MAX_SPEED/speed)
        
    def getAverageVelocity(self):
        steer = np.zeros(3)
        count = 0
        
        for b in self.flockmates:
            d = np.linalg.norm(np.subtract(self.position, b.position))
            
            if d > 0 and d < FRIEND_RADIUS:
                copy = np.copy(b.velocity)
                copy = clamp_magnitude(copy, 1/d)
                
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
            
            if d > 0 and d < AVOID_RADIUS:
                diff = np.subtract(self.position, b.position)
                diff = clamp_magnitude(diff, 1/max(d, 0.00001))
                
                steer = np.add(steer, diff)
                count += 1
                
        if count > 0:
            steer /= count 
            
        return steer
    
    def getAvoidBorders(self):
        steer = np.zeros(3)
        
        if self.position[2] + HEIGHT/2 < AVOID_RADIUS and self.velocity[2] < 0:
            steer += np.array([self.velocity[0], self.velocity[1], 0.1]) / max(self.position[2] + HEIGHT/2, 0.00001)
            
        if HEIGHT/2 - self.position[2] < AVOID_RADIUS and self.velocity[2] > 0:
            steer += np.array([self.velocity[0], self.velocity[1], -0.1]) / max(HEIGHT/2 - self.position[2], 0.00001)
            
        return steer
    
    def getAvoidObstacles(self):
        steer = np.zeros(3)
        
        for a in obstacles:
            d = np.linalg.norm(np.subtract(self.position, a.position)) - a.radius
            
            if d < AVOID_RADIUS:
                diff = np.subtract(self.position, a.position)
                diff = clamp_magnitude(diff, 1/max(d, 0.00001))
                
                steer = np.add(steer, diff)
            
        return steer
    
    def getCohesion(self):
        sum = np.zeros(3)
        count = 0
        
        for b in self.flockmates:
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
                                        rotation=Euler((0,np.arccos(velocity[2]/np.linalg.norm(velocity)),np.arctan2(velocity[1], velocity[0])), 'XYZ'),
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
HEIGHT = 10
MAX_SPEED = 0.15
MAX_ACCELERATION = 0.5
FRIEND_RADIUS = 2
AVOID_RADIUS = 0.5
COHESION_RADIUS = FRIEND_RADIUS

START_FRAME = 1
END_FRAME = 240

NUM_BOIDS = 50

for i in range(NUM_BOIDS):
    delete_object("Boid " + str(i+1))
    
for i in range(NUM_BOIDS):
    p = np.array([SCALE * random() - SCALE/2, SCALE * random()- SCALE/2, HEIGHT * random() - HEIGHT/2])
    v = clamp_magnitude(np.array([random()-1, random()-1, random()-1]), MAX_SPEED / 2)
    
    b = initialize_body("Boid " + str(i+1), p, v)
    
    bodies.append(b)
    brains.append(Boid(p, v))
    
    b.animation_data_create()
    b.animation_data.action = bpy.data.actions.new(name="Flight")

for t in range(60):
    for i in range(NUM_BOIDS):
        brains[i].fly()

for t in range(START_FRAME, END_FRAME+1):
    for i in range(NUM_BOIDS):
        bodies[i].keyframe_insert(data_path="location", frame=t)
        bodies[i].keyframe_insert(data_path="rotation_euler", frame=t)
        brains[i].fly()
        
        bodies[i].location = brains[i].position
        bodies[i].rotation_euler = Euler((0, np.arccos(brains[i].velocity[2] / np.linalg.norm(brains[i].velocity)),
                                        np.arctan2(brains[i].velocity[1], brains[i].velocity[0])), 'XYZ')