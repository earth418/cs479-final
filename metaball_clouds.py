import bpy
from random import randrange

def delete_object(name):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        bpy.data.objects.remove(obj, do_unlink=True)
        
def create_cloud(name, num_mballs, x_min, x_max, y_min, y_max, z_min, z_max, r_min, r_max):
    
    # Create Metaballs + Mesh
    for i in range(num_mballs):
        bpy.ops.object.metaball_add(radius=randrange(r_min, r_max), location=(randrange(x_min, x_max), randrange(y_min, y_max), randrange(z_min, z_max)))
    
    bpy.ops.object.convert(target='MESH')
    cloud_mesh = bpy.context.object
    cloud_mesh.name = name
    cloud_mesh.hide_viewport = True
    cloud_mesh.hide_render = True
    
    # Mesh to Volume
    bpy.ops.object.volume_add()
    mod = bpy.context.object.modifiers.new(name='MeshToVolume', type='MESH_TO_VOLUME')
    mod.voxel_amount = 150
    mod.object = cloud_mesh
    
    # Cloud Displacement
    tex = bpy.data.textures.new("Displace", 'CLOUDS')
    tex.noise_depth = 3
    tex.noise_scale = 0.8
    tex.noise_type = 'SOFT_NOISE'
    
    mod = bpy.context.object.modifiers.new(name='VolumeDisplace', type='VOLUME_DISPLACE')
    mod.texture = tex
    mod.strength = 2
    
delete_object("Cloud")
delete_object("Volume")
create_cloud("Cloud", 75, -10, 10, -10, 10, -2, 2, 2, 4)