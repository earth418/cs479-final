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
import os
import sys
import random

import math

dirp = os.path.dirname(bpy.data.filepath)
if not dirp in sys.path:
    sys.path.append(dirp)

from Delaunator import Delaunator

def create_building_mat():

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

#    if bricked:
    links.new(
        nodes["Brick Texture"].outputs["Color"],
        nodes["Mix"].inputs["B"]
        )
#    elif checkered:
#        links.new(
#            nodes["Checker Texture"].outputs["Color"],
#            nodes["Mix"].inputs["B"]
#            )


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


def create_terrain_mat():
    
    import bpy
    groups = {}  # for node groups

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
    node.inputs[2].default_value = 0.07000000029802322 # param for path

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

    links.new(
        nodes["Brick Texture"].outputs["Color"],
        nodes["Mix.001"].inputs["B"]
        )

    links.new(
        nodes["Mix.001"].outputs["Result"],
        nodes["Principled BSDF"].inputs["Base Color"]
        )
        
    return mat

def flat(location : Vector):
    return 0.0

def noisy_height2(location: Vector):
    
    final_noise = 0.0
    noise_mag = 1.0
    
    persistence = 0.32
    P = persistence
    
    lacunarity = 2.4
    L = lacunarity
    
    for i in range(5):
        final_noise += (noise.noise(L * location, noise_basis='PERLIN_ORIGINAL')) * P
        noise_mag += P
        
        P *= persistence
        L *= lacunarity
        
    return final_noise / noise_mag

def create_terrain(location, size, feature_scale = 1.0, height_scale = 1.0, horiz_scale = 0.1, height_function = flat):

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
    obj.location = location
    obj.data.materials.append(create_terrain_mat())
    
    return heightmap # return a heightmap here!

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
        return VERT_SCALE * noisy_height2(FT_SCALE * HORIZ_SCALE * location)
    
    def gradient_plot(location):
        
        ground_height = terrain(location)
        eps = Vector((0.05, 0.0, 0.0))
        
        gradient = Vector((terrain(location + eps.xyy) - ground_height,
                           terrain(location + eps.yxy) - ground_height, 0.0)) / eps.x
    
        if gradient.length > 0.25:
            return 1.0
        
        return 0.0
    
    def add_buildings(location, input_buildings, height_function):
    
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
        
        print(len(triangles))
        
#        for building in input_buildings:
        for i in range(0, len(triangles), 8):
            r, g, b = [random.random() for _k in range(3)]
            
            for j in range(3*8):
                mesh.vertex_colors.active.data[3*i+j].color = (r, g, b, 1.0)

        
        # useful for development when the mesh may be invalid.
        # mesh.validate(verbose=True)
        obj = object_data_add(bpy.context, mesh)
        obj.data.materials.append(create_building_mat())
        
#        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        obj.location = location
        
        return mesh
#       
    heightmap = create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, terrain)
    
    builds = get_buildings(LOC, SIZE * HORIZ_SCALE)
    
    add_buildings(LOC, builds, terrain)
    

#    create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, combined)
    
#    create_terrain(LOC, SIZE, FT_SCALE, VERT_SCALE, HORIZ_SCALE, combined_function)

#    create_terrain(LOC + Vector((SIZE * HORIZ_SCALE, 0.0, 0.0)), SIZE, FT_SCALE, 10.0, HORIZ_SCALE, combined)
#    create_terrain(LOC + Vector((SIZE * HORIZ_SCALE, 0.0, 0.0)), SIZE, 25.0 * FT_SCALE, VERT_SCALE, HORIZ_SCALE, street_func)
    
    
#   triangle = [[k.x, k.y] for k in cells[:3]]
        
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
