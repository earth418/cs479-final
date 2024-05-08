bl_info = {
    "name": "Create City",
    "author": "Ali Hafez",
    "version": (4, 7, 9),
    "blender": (2, 80, 0),
    "location": "View3D > Add > Mesh > New Object",
    "description": "Adds a new Mesh Object",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}


import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector, noise
from copy import deepcopy
from math import floor, sqrt

import math

EPSILON = math.pow(2,-52)
EDGE_STACK =[None] * 512

class Delaunator:

    def __init__(self,points):
        n = len(points)

        if (len(points) < 3):
            raise ValueError("Need at least 3 points")
        coords = [None] * n * 2

        for i in range(0,n):
            p = points[i]
            coords[2 * i] = (p[0])
            coords[2 * i+1] = (p[1])
        triangles = self.constructor(coords)

    def constructor(self, coords):
        n = len(coords) >> 1

        self.coords = coords

        # arrays that will store the triangulation graph
        maxTriangles = max(2 * n - 5, 0)
        self._triangles = [None] * maxTriangles * 3
        self._halfedges = [None] * maxTriangles * 3

        # temporary arrays for tracking the edges of the advancing convex hull
        self.hashSize = math.ceil(math.sqrt(n))
        self.hullPrev = [None] * n # edge to prev edge
        self.hullNext = [None] * n # edge to next edge
        self.hullTri = [None] * n # edge to adjacent triangle
        self.hullHash = [-1] * self.hashSize # angular edge hash

        # temporary arrays for sorting points
        self._ids =  [None] * n
        self._dists = [None] * n
        triangles = self.update(coords)

        return triangles

    def update(self,coords):
        n = len(coords) >> 1

        # populate an array of point indices; calculate input data bbox
        minX = math.inf
        minY = math.inf
        maxX = -math.inf
        maxY = -math.inf

        for i in range(0,n):
            x = coords[2 * i]
            y = coords[2 * i + 1]
            if (x < minX): minX = x
            if (y < minY): minY = y
            if (x > maxX): maxX = x
            if (y > maxY): maxY = y
            self._ids[i] = i

        cx = (minX + maxX) / 2
        cy = (minY + maxY) / 2

        minDist = math.inf
        i0 = 0
        i1 = 0
        i2 = 0

        # pick a seed point close to the center
        for i in range(0,n):
            d = dist(cx, cy, coords[2 * i], coords[2 * i + 1])

            if (d < minDist):
                i0 = i
                minDist = d

        i0x = coords[2 * i0]
        i0y = coords[2 * i0 + 1]
        minDist = math.inf

        # find the point closest to the seed
        for i in range(0,n):
            if (i == i0): continue
            d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1])

            if (d < minDist and d > 0):
                i1 = i
                minDist = d

        i1x = coords[2 * i1]
        i1y = coords[2 * i1 + 1]

        minRadius = math.inf

        # find the third point which forms the smallest circumcircle with the first two
        for i in range(0,n):
            if (i == i0 or i == i1): continue
            r = circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1])

            if (r < minRadius):
                i2 = i
                minRadius = r

        i2x = coords[2 * i2]
        i2y = coords[2 * i2 + 1]

        if (minRadius == math.inf):
            # order collinear points by dx (or dy if all x are identical)
            # and return the list as a hull
            for i in range(0,n):
                self._dists[i] = (coords[2 * i] - coords[0]) or (coords[2 * i + 1] - coords[1])

            quicksort(self._ids, self._dists, 0, n - 1)
            hull =  [None] * n
            j = 0
            d0 = -math.inf

            for i in range(0,n):
                id = self._ids[i]

                if (self._dists[id] > d0):
                    hull[j] = id
                    j+=1
                    d0 = self._dists[id]

            self.hull = hull[0:j]
            self.triangles =  []
            self.halfedges =  []

        # swap the order of the seed points for counter-clockwise orientation
        if (orient(i0x, i0y, i1x, i1y, i2x, i2y)):
            i = i1
            x = i1x
            y = i1y
            i1 = i2
            i1x = i2x
            i1y = i2y
            i2 = i
            i2x = x
            i2y = y

        center = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y)
        self._cx = center[0]
        self._cy = center[1]

        for i in range(0,n):
            self._dists[i] = dist(coords[2 * i], coords[2 * i + 1], center[0], center[1])

        # sort the points by distance from the seed triangle circumcenter
        quicksort(self._ids, self._dists, 0, n - 1)

        # set up the seed triangle as the starting hull
        self._hullStart = i0
        hullSize = 3

        self.hullNext[i0] = self.hullPrev[i2] = i1
        self.hullNext[i1] = self.hullPrev[i0] = i2
        self.hullNext[i2] = self.hullPrev[i1] = i0

        self.hullTri[i0] = 0
        self.hullTri[i1] = 1
        self.hullTri[i2] = 2

        self.hullHash[self._hashKey(i0x, i0y)] = i0
        self.hullHash[self._hashKey(i1x, i1y)] = i1
        self.hullHash[self._hashKey(i2x, i2y)] = i2

        self.trianglesLen = 0
        self._addTriangle(i0, i1, i2, -1, -1, -1)

        xp=0
        yp=0

        for k in range(0,len(self._ids)):
            i = self._ids[k]
            x = coords[2 * i]
            y = coords[2 * i + 1]

            # skip near-duplicate points
            if (k > 0 and abs(x - xp) <= EPSILON and abs(y - yp) <= EPSILON): continue

            xp = x
            yp = y

            # skip seed triangle points
            if (i == i0 or i == i1 or i == i2): continue

            # find a visible edge on the convex hull using edge hash
            start = 0
            key = self._hashKey(x, y)

            for j in range(0,self.hashSize):
                start = self.hullHash[(key + j) % self.hashSize]
                if (start != -1 and start != self.hullNext[start]): break

            start = self.hullPrev[start]
            e = start

            while True:
                q = self.hullNext[e]
                if orient(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1]): break
                e = q

                if (e == start):
                    e = -1
                    break

            if (e == -1): continue # likely a near-duplicate point; skip it

            # add the first triangle from the point
            t = self._addTriangle(e, i, self.hullNext[e], -1, -1, self.hullTri[e])

            # recursively flip triangles from the point until they satisfy the Delaunay condition
            self.hullTri[i] = self._legalize(t + 2,coords)
            self.hullTri[e] = t # keep track of boundary triangles on the hull
            hullSize+=1

            # walk forward through the hull, adding more triangles and flipping recursively
            n = self.hullNext[e]

            while True:
                q = self.hullNext[n]
                if not (orient(x, y, coords[2 * n], coords[2 * n + 1], coords[2 * q], coords[2 * q + 1])): break
                t = self._addTriangle(n, i, q, self.hullTri[i], -1, self.hullTri[n])
                self.hullTri[i] = self._legalize(t + 2,coords)
                self.hullNext[n] = n # mark as removed
                hullSize-=1
                n = q

            # walk backward from the other side, adding more triangles and flipping
            if (e == start):
                while True:
                    q = self.hullPrev[e]
                    if not (orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1])): break
                    t = self._addTriangle(q, i, e, -1, self.hullTri[e], self.hullTri[q])
                    self._legalize(t + 2,coords)
                    self.hullTri[q] = t
                    self.hullNext[e] = e # mark as removed
                    hullSize-=1
                    e = q

            # update the hull indices
            self._hullStart = self.hullPrev[i] = e
            self.hullNext[e] = self.hullPrev[n] = i
            self.hullNext[i] = n

            # save the two new edges in the hash table
            self.hullHash[self._hashKey(x, y)] = i
            self.hullHash[self._hashKey(coords[2 * e], coords[2 * e + 1])] = e

        self.hull = [None] * hullSize
        e = self._hullStart
        for i in range(0,hullSize):
            self.hull[i] = e
            e = self.hullNext[e]

        # trim typed triangle mesh arrays
        self.triangles = self._triangles[0:self.trianglesLen]
        self.halfedges = self._halfedges[0:self.trianglesLen]

        return self.triangles

    def _hashKey(self,x, y):
        return math.floor(pseudoAngle(x - self._cx, y - self._cy) * self.hashSize) % self.hashSize

    def _legalize(self,a,coords):
        i = 0
        ar = 0

        # recursion eliminated with a fixed-size stack
        while True:
            b = self._halfedges[a]
            """
              if the pair of triangles doesn't satisfy the Delaunay condition
              (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
              then do the same check/flip recursively for the new pair of triangles
             
                        pl                    pl
                       /||\                  /  \
                    al/ || \bl            al/    \a
                     /  ||  \              /      \
                    /  a||b  \    flip    /___ar___\
                  p0\   ||   /p1   =>   p0\---bl---/p1
                     \  ||  /              \      /
                    ar\ || /br             b\    /br
                       \||/                  \  /
                        pr                    pr
             
            """
            a0 = a - a % 3
            ar = a0 + (a + 2) % 3

            if (b == -1): # convex hull edge
                if (i == 0): break
                i-=1
                a = EDGE_STACK[i]
                continue

            b0 = b - b % 3
            al = a0 + (a + 1) % 3
            bl = b0 + (b + 2) % 3

            p0 = self._triangles[ar]
            pr = self._triangles[a]
            pl = self._triangles[al]
            p1 = self._triangles[bl]

            illegal = inCircle(
                coords[2 * p0], coords[2 * p0 + 1],
                coords[2 * pr], coords[2 * pr + 1],
                coords[2 * pl], coords[2 * pl + 1],
                coords[2 * p1], coords[2 * p1 + 1])

            if (illegal):
                self._triangles[a] = p1
                self._triangles[b] = p0

                hbl = self._halfedges[bl]

                # edge swapped on the other side of the hull (rare); fix the halfedge reference
                if (hbl == -1):
                    e = self._hullStart
                    
                    while True:
                        if (self.hullTri[e] == bl):
                            self.hullTri[e] = a
                            break

                        e = self.hullPrev[e]
                        if (e == self._hullStart): break

                self._link(a, hbl)
                self._link(b, self._halfedges[ar])
                self._link(ar, bl)

                br = b0 + (b + 1) % 3

                # don't worry about hitting the cap: it can only happen on extremely degenerate input
                if (i < len(EDGE_STACK)):
                    EDGE_STACK[i] = br
                    i+=1

            else:
                if (i == 0): break
                i-=1
                a = EDGE_STACK[i]

        return ar

    def _link(self,a, b):
        self._halfedges[a] = b
        if (b != -1):
            self._halfedges[b] = a

    # add a new triangle given vertex indices and adjacent half-edge ids
    def _addTriangle(self,i0, i1, i2, a, b, c):
        t = self.trianglesLen

        self._triangles[t] = i0
        self._triangles[t + 1] = i1
        self._triangles[t + 2] = i2

        self._link(t, a)
        self._link(t + 1, b)
        self._link(t + 2, c)

        self.trianglesLen += 3

        return t

# monotonically increases with real angle, but doesn't need expensive trigonometry
def pseudoAngle(dx, dy):
    p = dx / (abs(dx) + abs(dy))

    if (dy > 0):
        return (3 - p) / 4 # [0..1]
    else:
        return (1 + p) / 4 # [0..1]

def dist(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

# return 2d orientation sign if we're confident in it through J. Shewchuk's error bound check
def orientIfSure(px, py, rx, ry, qx, qy):
    l = (ry - py) * (qx - px)
    r = (rx - px) * (qy - py)

    if (abs(l - r) >= 3.3306690738754716e-16 * abs(l + r)):
        return l - r
    else:
        return 0

# a more robust orientation test that's stable in a given triangle (to fix robustness issues)
def orient(rx, ry, qx, qy, px, py):
    return (orientIfSure(px, py, rx, ry, qx, qy) or\
        orientIfSure(rx, ry, qx, qy, px, py) or\
        orientIfSure(qx, qy, px, py, rx, ry)) < 0

def inCircle(ax, ay, bx, by, cx, cy, px, py):
    dx = ax - px
    dy = ay - py
    ex = bx - px
    ey = by - py
    fx = cx - px
    fy = cy - py

    ap = dx * dx + dy * dy
    bp = ex * ex + ey * ey
    cp = fx * fx + fy * fy

    return dx * (ey * cp - bp * fy) -\
           dy * (ex * cp - bp * fx) +\
           ap * (ex * fy - ey * fx) < 0

def circumradius(ax, ay, bx, by, cx, cy):
    dx = bx - ax
    dy = by - ay
    ex = cx - ax
    ey = cy - ay

    bl = dx * dx + dy * dy
    cl = ex * ex + ey * ey
    try:
        d = 0.5/(dx * ey - dy * ex)
    except ZeroDivisionError:
        d = float('inf')

    x = (ey * bl - dy * cl) * d
    y = (dx * cl - ex * bl) * d

    return x*x + y*y

def circumcenter(ax, ay, bx, by, cx, cy):
    dx = bx - ax
    dy = by - ay
    ex = cx - ax
    ey = cy - ay

    bl = dx * dx + dy * dy
    cl = ex * ex + ey * ey
    try:
        d = 0.5/(dx * ey - dy * ex)
    except ZeroDivisionError:
        d = float('inf')

    x = ax + (ey * bl - dy * cl) * d
    y = ay + (dx * cl - ex * bl) * d

    return x, y

def quicksort(ids, dists, left, right):
    if (right - left <= 20):
        for i in range(left + 1,right+1):
            temp = ids[i]
            tempDist = dists[temp]
            j = i-1
            while (j >= left and dists[ids[j]] > tempDist):
                ids[j + 1] = ids[j]
                j-=1
            ids[j + 1] = temp;

    else:
        median = (left + right) >> 1
        i = left + 1
        j = right
        swap(ids, median, i)

        if (dists[ids[left]] > dists[ids[right]]):
            swap(ids, left, right)

        if (dists[ids[i]] > dists[ids[right]]):
            swap(ids, i, right)

        if (dists[ids[left]] > dists[ids[i]]):
            swap(ids, left, i)

        temp = ids[i]
        tempDist = dists[temp]

        while True:
            while True:
                i+=1
                if (dists[ids[i]] >= tempDist): break

            while True:
                j-=1
                if (dists[ids[j]] <= tempDist): break

            if (j < i): break
            swap(ids, i, j);

        ids[left + 1] = ids[j];
        ids[j] = temp;

        if (right - i + 1 >= j - left):
            quicksort(ids, dists, i, right)
            quicksort(ids, dists, left, j - 1)

        else:
            quicksort(ids, dists, left, j - 1)
            quicksort(ids, dists, i, right)

def swap(arr, i, j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


class height_functions():
    def flat(location):
        return 0.0    

    def wavy(location: Vector):
        from math import sin, cos
        return -sin(location.x) * cos(location.y) + 1 * sin(location.z) - 2 * cos(location.z)
        
    # Noisy heights!
    def noisy_height1(location: Vector):
        
        n = noise.noise(location, noise_basis='PERLIN_ORIGINAL')
        return n
    
#    create_terrain(Vector((-4.0, 0.0, 0.0)), 4, 1.0, noisy_height1)

    def noisy_height2(location: Vector):
        
        final_noise = 0.0
        
        persistence = 0.32
        P = persistence
        
        lacunarity = 2.4
        L = lacunarity
        
        for i in range(5):
            final_noise += (noise.noise(L * location * 0.75, noise_basis='PERLIN_ORIGINAL')) * P
            P *= persistence
            L *= lacunarity
            
        return final_noise * 0.4


def create_terrain(location, size, feature_scale = 1.0, height_scale = 1.0, horiz_scale = 0.1, height_function = height_functions.flat):

    vertices = []
    triangles = []
    edges = []
    
    heightmap = []
    
    for i in range(size):
        h_row = []
        
        
        for j in range(size):
            
            x = horiz_scale * i
            y = horiz_scale * j
            z = height_scale * height_function(0.1 * feature_scale * Vector((x, y, 0.0)))

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
    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    obj = object_data_add(bpy.context, mesh)
    obj.location = location
    
    return heightmap # return a heightmap here!


def spawn_buildings(location, size, building_scale = 1.0, building_height = 1.0):
    
    init_verts = []
    
    pts = []
    
    for i in range(floor(size) + 1):
        for j in range(floor(size) + 1):
            
            pos = Vector((i,j, 0.0))
            cell = pos + noise.cell_vector(pos)
            
            init_verts.append(cell * Vector((1.0, 1.0, 0.0)))
            
            pts.append([cell[0], cell[1]])
        
    
    del_tris = Delaunator(pts).triangles
    
    triangles = []
    vertices = []    
    
    for i in range(0, len(del_tris), 3):
        
        tri = [del_tris[i+k] for k in range(3)]
        
        centroid = Vector((0.0, 0.0, 0.0))
        
        a = init_verts[tri[0]]
        b = init_verts[tri[1]]
        c = init_verts[tri[2]]
        
        area = 0.5 * abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
        
        if area < 0.4:
            continue
                        
        for p in tri:
            centroid += init_verts[p] / 3
        
        if centroid.x > size or centroid.y > size or centroid.x < 0.8 or centroid.y < 0.8:
            continue
        
        new_tri1 = []
        for p in tri:
            diff = init_verts[p] - centroid
            i = len(vertices)
            
            vertices.append(centroid + diff * 0.8)
            new_tri1.append(i)
            
        triangles.append(new_tri1)
        new_tri2 = []
        
        for p in tri:
            diff = init_verts[p] - centroid
            i = len(vertices)
            
            vertices.append(centroid + diff * 0.8 + Vector((0.0, 0.0, building_scale + noise.noise(centroid))))
            new_tri2.append(i)
    
        triangles.append(new_tri2)
        
        for index in range(3):
             
            nindex = (index + 1) % 3
            triangles.append([new_tri1[index],  new_tri2[index], new_tri1[nindex]])
            triangles.append([new_tri1[nindex], new_tri2[index], new_tri2[nindex]])
        
    
    mesh = bpy.data.meshes.new(name="Buildings")
    mesh.from_pydata(vertices, [], triangles)
    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    obj = object_data_add(bpy.context, mesh)
    obj.location = location
    
    return mesh

def get_buildings(location, size, building_scale = 1.0, building_height = 1.0):
    
    init_verts = []
    pts = []
    
    for i in range(floor(size) + 1):
        for j in range(floor(size) + 1):
            
            pos = Vector((i,j, 0.0))
            cell = pos + noise.cell_vector(pos)
            
            init_verts.append(cell * Vector((1.0, 1.0, 0.0)))
            
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
        for p in tri:
            diff = init_verts[p] - centroid
            i = len(b_verts)
            
            b_verts.append(centroid + diff * 0.8)
            new_tri1.append(i)
            
        b_tris.append(new_tri1)
        new_tri2 = []
        
        for p in tri:
            diff = init_verts[p] - centroid
            i = len(b_verts)
            
            b_verts.append(centroid + diff * 0.8 + Vector((0.0, 0.0, building_scale + noise.noise(centroid))))
            new_tri2.append(i)
    
        b_tris.append(new_tri2)
        
        for index in range(3):
             
            nindex = (index + 1) % 3
            b_tris.append([new_tri1[index],  new_tri2[index], new_tri1[nindex]])
            b_tris.append([new_tri1[nindex], new_tri2[index], new_tri2[nindex]])
            
        buildings.append((b_verts, b_tris))
    
    return buildings

def barycentric(input : list, triangle : list):
    
    # all vector3, ignor3e z coordinate
    inx, iny = input
    
    xa, ya = triangle[0]
    xb, yb = triangle[1]
    xc, yc = triangle[2]
    A = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
    
    beta = ((xa - xc) * (iny - yc) - (ya - yc) * (inx - xc)) / A
    gamma = ((xb - xa) * (iny - ya) - (yb - ya) * (inx - xa)) / A
    alpha = 1 - beta - gamma
    
    return alpha, beta, gamma

def register():

    SIZE = 500
    VERT_SCALE = 50.0
    FT_SCALE = 0.2
    HORIZ_SCALE = 0.1
    LOC = Vector((0.0, 0.0, 0.0))

#    heightmap = create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, height_functions.noisy_height2)
    
    
#    distances, cells = noise.voronoi(Vector((SIZE * HORIZ_SCALE, 0.0, 0.0)), distance_metric='DISTANCE')
#    print(cells)
    
    def terrain(location):
        return height_functions.noisy_height2(location)
    
    def street_func(location, metric="DISTANCE"):
        distances, cells = noise.voronoi(location, distance_metric=metric)
        
#        edge = [cells[0], cells[1]] # delaunay edge!
#        
#        numerator =  (edge[1].x - edge[0].x) * (location.y - edge[0].y) - \
#                     (location.x - edge[0].x) * (edge[1].y - edge[0].y)
#        denominator = sqrt((edge[1].x - edge[0].x) ** 2 + (edge[1].y - edge[0].y) ** 2)
#        
#        dist = numerator / denominator
        
        EPS1 = 0.1
        EPS2 = 0.2
        
#
        height = 0.0
        
        # big blocks
        if distances[1] - distances[0] > EPS1:
            height += 1.0
        
        # plazas?
#        if distances[2] - distances[0] > EPS2:
#            height -= 1.0
        
        return max(0.0, height)
    
    def combined_function(location):
        
        c = combined(location)
        
        return terrain(location) + 0.02 * street_func(25.0 * location) * (1.0 - c)
        
#    def new_func(location):
#        
#        return street_func(location) * street_func(5 * location, metric="MANHATTAN")
    
    def gradient_plot(location):
        
        ground_height = terrain(location)
        eps = Vector((0.05, 0.0, 0.0))
        
        gradient = Vector((terrain(location + eps.xyy) - ground_height,
                           terrain(location + eps.yxy) - ground_height, 0.0)) / eps.x
    
        if gradient.length > 0.25:
            return 1.0
        
        return 0.0
    
    
    builds = buildings(LOC, SIZE * HORIZ_SCALE)
    
    def add_buildings(input_buildings):
    
        triangles = []
        vertices = []
    
        for building in input_buildings:
                
            verts = building[0]
            tris = building[1]
                
            centroid = Vector((0.0, 0.0, 0.0))
        
            for p in tris[:3]:
                centroid += verts[p] / 3

            eps = Vector((0.05, 0.0, 0.0))
                        
            ground_height = terrain(centroid)
            
            gradient = Vector((terrain(centroid + eps.xyy) - ground_height,
                               terrain(centroid + eps.yxy) - ground_height, 0.0)) / eps.x
            
            if gradient.length <= 0.25:
                
                # spawn the actual building
                ind = len(vertices)
                triangles += [[ind + k for k in tri] for tri in tris]
                vertices += verts
            
        mesh = bpy.data.meshes.new(name="Buildings")
        mesh.from_pydata(vertices, [], triangles)
        
        
        # useful for development when the mesh may be invalid.
        # mesh.validate(verbose=True)
        obj = object_data_add(bpy.context, mesh)
        # obj.location = location
        
        return mesh

#       
    heightmap = create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, height_functions.noisy_height2)

#    create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, combined)
    
#    create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, combined_function)

#    create_terrain(LOC + Vector((SIZE * HORIZ_SCALE, 0.0, 0.0)), SIZE, FT_SCALE, 10.0, HORIZ_SCALE, combined)
#    create_terrain(LOC + Vector((SIZE * HORIZ_SCALE, 0.0, 0.0)), SIZE, 25.0 * FT_SCALE, VERT_SCALE, HORIZ_SCALE, street_func)
    
    
# triangle = [[k.x, k.y] for k in cells[:3]]
        
#        triangle = [[0.0, 0.0], [2.0, 2.0], [0.0, 1.0]]
#        a, b, g = barycentric([location.x, location.y], triangle)
#        
#        if a > EPS and b > EPS and g > EPS:
#            return 1.0

def unregister():
    return

if __name__ == "__main__":
    register()


#class OBJECT_OT_add_object(Operator, AddObjectHelper):
#    """Create a new Mesh Object"""
#    bl_idname = "mesh.add_object"
#    bl_label = "Add Potatoid"
#    bl_options = {'REGISTER', 'UNDO'}

##    order: IntProperty(
##        name="order",
##        default=3,
##        description="scaling"
##    )
##    
#    scale: FloatVectorProperty(
#        name="scale",
#        default=(1.0, 1.0, 1.0),
#        subtype='TRANSLATION',
#        description="scaling",
#    )

#    def execute(self, context):

#        add_object(self, context)

#        return {'FINISHED'}


# Registration

#def add_object_button(self, context):
#    self.layout.operator(
#        OBJECT_OT_add_object.bl_idname,
#        text="Add Potatoid",
#        icon='PLUGIN')


# This allows you to right click on a button and link to documentation
#def add_object_manual_map():
#    url_manual_prefix = "https://docs.blender.org/manual/en/latest/"
#    url_manual_mapping = (
#        ("bpy.ops.mesh.add_object", "scene_layout/object/types.html"),
#    )
#    return url_manual_prefix, url_manual_mapping


#def register():
#    bpy.utils.register_class(OBJECT_OT_add_object)
#    bpy.utils.register_manual_map(add_object_manual_map)
#    bpy.types.VIEW3D_MT_mesh_add.append(add_object_button)


#def unregister():
#    bpy.utils.unregister_class(OBJECT_OT_add_object)
#    bpy.utils.unregister_manual_map(add_object_manual_map)
#    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)
