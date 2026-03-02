import bpy, sys, os, math, json, argparse
from mathutils import Vector
import traceback
# Force use of Cycles + CPU rendering (solve no GPU/EEVEE error)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 32
bpy.context.scene.cycles.use_adaptive_sampling = True
bpy.context.scene.cycles.use_transparent_shadows = False
bpy.context.scene.view_settings.view_transform = 'Standard'

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)  # Used the new Blender OBJ importer, no problem
    # Move the model origin to(0,0,0)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.location = (0,0,0)


def setup_lighting():
    bpy.ops.object.light_add(type='SUN', location=(5,5,5))
    bpy.context.object.data.energy = 5.0
    # # Increase the ambient light a bit
    # bpy.context.scene.world.use_nodes = True
    # bg = bpy.context.scene.world.node_tree.nodes['Background']
    # bg.inputs[1].default_value = 0.5  # 环境光强度


def setup_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    return camera


def xzy_to_location(x_deg, y_deg, radius=2.5, z_off=0):
    # x = pitch(up/down), y = yaw(left/right) ,z_off The height of the perspective can be adjusted
    pr = math.radians(90 - x_deg)
    ar = math.radians(y_deg)
    x = radius * math.sin(pr) * math.sin(ar)
    y = radius * math.sin(pr) * math.cos(ar)
    z = radius * math.cos(pr) + z_off
    return (x, y, z)


def xyz_to_euler(x_deg, y_deg, z_deg):
    # eular in radians: (X_pitch, Y_roll, Z_raw)
    return (math.radians(x_deg), math.radians(z_deg), math.radians(y_deg))


def look_at(obj_camera, target=(0,0,0)):
    """
    Let the camera face the target
    """
    direction = Vector(target) - obj_camera.location
    # In Blender, the camera looks along the -Z axis, and the Y axis points up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()


def render_views(model_dir, output_dir, camera_json):
    obj_path = os.path.join(model_dir, "models", "model_normalized.obj")
    if not os.path.exists(obj_path):
        print(f"❌ Missing model OBJ: {obj_path}")
        return

    # Prepare scene
    clear_scene()
    import_obj(obj_path)
    setup_lighting()
    cam = setup_camera()

    # Render settings
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    # Load camera definitions
    with open(camera_json, "r") as f:
        cams = json.load(f)

    # Loop over views
    for entry in cams:
        vid = entry["view_id"]
        x = entry["az"]
        y = entry["el"]
        z = entry["roll"]

        cam.location = Vector(xzy_to_location(x, y))
        cam.rotation_euler = xyz_to_euler(x, y, z)
        look_at(cam, Vector((0, 0, 0)))
        # cam.rotation_euler.rotate_axis('Z', math.pi)

        out_path = os.path.join(output_dir, f"{vid:03d}.png")
        scene.render.filepath = os.path.abspath(out_path)
        bpy.ops.render.render(write_still=True)
        print("Rendered view", vid, "->", out_path)


if __name__ == "__main__":
    try:
        # parse args after "--"
        argv = sys.argv
        argv = argv[sys.argv.index("--")+1:]
        p = argparse.ArgumentParser()
        p.add_argument("--model_dir",   required=True)
        p.add_argument("--output_dir",  required=True)
        p.add_argument("--camera_json", required=True)
        args = p.parse_args(argv)

        render_views(args.model_dir, args.output_dir, args.camera_json)

    except Exception:
        print("🔥 Error in render_script2.py:")
        traceback.print_exc()


