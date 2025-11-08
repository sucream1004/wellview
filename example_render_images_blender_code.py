# This code is included in the Blender file.

import bpy
import time
import numpy as np

window_objs = [x for x in bpy.data.objects if "window" in x.name]

# ------------------------ #
# -------- reset --------- #
# ------------------------ #

for window_obj in window_objs:
    window_obj.location = (0, 0, 0)
    window_obj.dimensions = (1,0.5,1)

# ------------------------ #
# -----new window size---- #
# ------------------------ #

# new_win_dim = (1.0, 0.5, 1.0)
# Window size change is determined by i & j.
dims = (lambda: [(x, 0.5, z) for x in [1 + i*1.0 for i in range(int((8-1)/0.5)+1)] for z in [1 + j*0.5 for j in range(int((2-1)/0.5)+1)]])()

for new_win_dim in dims:
    # new_win_dim is already defined and is a sequence (e.g., a tuple) with at least 3 values.
    start_x = -4 + new_win_dim[0] / 2
    end_x   = 4  - new_win_dim[0] / 2
    start_z = round(-1.22 + new_win_dim[2] / 2, 2)
    end_z   = round(1.22 - new_win_dim[2] / 2, 2)
    step_x = 1.0
    step_z = 0.5
    # min_win_loc_x = -4 + new_win_dim.x/2
    # max_win_loc_x = 4 - new_win_dim.x/2
    for x in np.arange(start_x, end_x, step_x):
        for z in np.arange(start_z, end_z, step_z):
            x = round(x,2)
            z = round(z,2)
            print(x,z)
            new_win_loc = (x, 0, z)

            window_param = str(new_win_loc) + "_" + str(new_win_dim)

            for window_obj in window_objs:
                window_obj.location = new_win_loc
                window_obj.dimensions = new_win_dim
                
            bpy.ops.object.visual_transform_apply()

            winmasks = [x for x in bpy.data.objects if "mask" in x.name]
            for obj in winmasks:
                obj.hide_render = True
            # First render the RGB image
            bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

            scene = bpy.context.scene
            scene.render.use_multiview = True
            scene.render.filepath = f'./rgb_{window_param}.png'
            cam = bpy.data.cameras['Camera_X']
            cam.stereo.convergence_mode = 'PARALLEL'
            cam.stereo.interocular_distance = 0.065
            bpy.ops.render.render(write_still=True)

            # turn on win_mask render
            # Due to FOV, other windows can be included in the rendered image so mask is needed to filter to extract the right window.
            winmasks = [x for x in bpy.data.objects if "mask" in x.name]
            for obj in winmasks:
                obj.hide_render = False

            bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'

            scene = bpy.context.scene
            scene.render.use_multiview = True
            scene.render.filepath = f'./mask_{window_param}.png'
            cam = bpy.data.cameras['Camera_X']
            cam.stereo.convergence_mode = 'PARALLEL'
            cam.stereo.interocular_distance = 0.065
            bpy.ops.render.render(write_still=True)
