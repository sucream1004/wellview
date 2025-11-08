# rq3\Scripts\activate
# set PATH=c:\\Program Files\\Blender Foundation\\Blender 4.3;%PATH%
# blender rq3_model_window_fix.blend --python train_rl.py

import time
import sys
import os
import glob
import numpy as np

# ------------------------
# Utility functions
# ------------------------

import mathutils

ada_palette = np.asarray([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255], [0, 0, 0]
])

def get_combined_state(image_files):
    """
    Load a list of 9 image files and stack them into one state.
    If the images are grayscale, the resulting shape will be (9, 512, 512) for channel-first.
    Alternatively, to get a shape of (512, 512, 9) for channels-last, change the axis in np.stack.
    """
    images = []
    for img_file in image_files:
        try:
            # Open each image, convert to grayscale ("L"), and resize if necessary.
            img = Image.open(img_file).convert("L")
            img = img.resize((512, 512))  # Ensure size consistency
            img_array = np.array(img, dtype=np.uint8)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            # Append a blank image if there's an error.
            images.append(np.zeros((512, 512), dtype=np.uint8))
    
    # Stack images along axis 0 for channel-first (shape: (9, 512, 512))
    state_channel_first = np.stack(images, axis=0)
    
    # If you need channels-last instead (shape: (512, 512, 9)):
    # state_channels_last = np.stack(images, axis=-1)
    
    return state_channel_first  # or state_channels_last

def switch_to_camera_view(camera_name):
    # Set the active camera
    cam_obj = bpy.data.objects.get(camera_name)
    bpy.context.scene.camera = cam_obj

    # Find a VIEW_3D area and override its context
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = bpy.context.copy()
            override['area'] = area
            for region in area.regions:
                if region.type == 'WINDOW':
                    override['region'] = region
                    break
            break

# Delete old *.txt and *.png files
for ext in ["*.txt", "*.png"]:
    for file in glob.glob(ext):
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def save_semseg(idx):
    from PIL import Image
    # Fixed canvas size for each individual image
    canvas_width = 256
    canvas_height = 256
    num_images = 9  # Number of images in vertical merge

    semseg_files = [
        "cam1_semseg.png",
        "cam2_semseg.png",
        "cam3_semseg.png",
        "cam4_semseg.png"
        "cam5_semseg.png",
        "cam6_semseg.png",
        "cam7_semseg.png",
        "cam8_semseg.png",
        "cam9_semseg.png"
    ]
    cam_files = [
        "unit_cam1_image.png",
        "unit_cam2_image.png",
        "unit_cam3_image.png",
        "unit_cam4_image.png",
        "unit_cam5_image.png",
        "unit_cam6_image.png",
        "unit_cam7_image.png",
        "unit_cam8_image.png",
        "unit_cam9_image.png"
    ]
    
    # Merge semantic segmentation images vertically
    merged_semseg = Image.new("RGBA", (canvas_width, canvas_height * num_images), (255, 255, 255, 0))
    for i, fname in enumerate(semseg_files):
        try:
            img = Image.open(fname).convert("RGBA")
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue
        
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
        offset_x = (canvas_width - img.width) // 2
        offset_y = (canvas_height - img.height) // 2
        canvas.paste(img, (offset_x, offset_y), mask=img)
        top = i * canvas_height
        merged_semseg.paste(canvas, (0, top))
    
    # Merge unit camera images vertically
    merged_cam = Image.new("RGBA", (canvas_width, canvas_height * num_images), (255, 255, 255, 0))
    for i, fname in enumerate(cam_files):
        try:
            img = Image.open(fname).convert("RGBA")
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue
        
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
        offset_x = (canvas_width - img.width) // 2
        offset_y = (canvas_height - img.height) // 2
        canvas.paste(img, (offset_x, offset_y), mask=img)
        top = i * canvas_height
        merged_cam.paste(canvas, (0, top))
    
    # Merge the two vertical images side by side
    final_width = canvas_width * 2
    final_height = canvas_height * num_images
    final_merged = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 0))
    final_merged.paste(merged_semseg, (0, 0))
    final_merged.paste(merged_cam, (canvas_width, 0))
    final_merged.save(f"chk/merged_semseg_and_cam_{idx}.png")

def render_views():
    import bpy
    win_mask = bpy.data.objects.get("windowmask")
    for cam_name in ["unit_cam1", "unit_cam2", "unit_cam3", "unit_cam4", "unit_cam5", "unit_cam6", "unit_cam7", "unit_cam8", "unit_cam9"]:
        switch_to_camera_view(cam_name)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'MATERIAL'
        win_mask.hide_viewport = True
        bpy.context.scene.render.filepath = "//" + cam_name + "_image"
        bpy.ops.render.opengl(write_still=True)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'SOLID'
        win_mask.hide_viewport = False
        bpy.context.scene.render.filepath = "//" + cam_name + "_mask"
        bpy.ops.render.opengl(write_still=True)

# ------------------------
# Semantic Segmentation & Reward Module
# ------------------------
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2

# Global model cache
_processor = None
_model = None

def load_models():
    global _processor, _model
    if _processor is None or _model is None:
        print("Loading models...")
        _processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        _model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        print("Models loaded!")
    return _processor, _model

def count_pixels_for_colors(img, colors):
    mask_total = np.zeros(img.shape[:2], dtype=bool)
    for color in colors:
        mask_color = (
            (img[:, :, 0] == color[0]) &
            (img[:, :, 1] == color[1]) &
            (img[:, :, 2] == color[2])
        )
        mask_total |= mask_color
    return np.sum(mask_total)

def semseg_cut_mask(im_path):
    from torch import nn
    import numpy as np
    import matplotlib.pyplot as plt
    processor, model = load_models()
    
    image = Image.open(im_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    seg = predicted_map
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    img = color_seg
    img = img.astype(np.uint8)
    semseg_im = Image.fromarray(img).convert("RGBA")
    
    import numpy as np
    import cv2
    mask_path = im_path.replace("_image.png", "_mask.png")
    
    # Load your image with PIL and ensure it has an alpha channel
    im = Image.open(mask_path).convert("RGBA")
    im_np = np.array(im)

    mask = (im_np[:, :, 0] >= 200)

    # Set alpha = 0 for pixels that do NOT satisfy the condition.
    im_np[~mask, 3] = 0

    # Extract the alpha channel
    alpha = im_np[:, :, 3]

    # Create a binary mask where alpha > 0
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask.
    # cv2.findContours returns contours and hierarchy.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or len(contours[0]) < 3:
        print("Warning: No valid contours found")
        return None, 0.0
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) < 3:
        print("Warning: Not enough points for polygon approximation")
        return None, 0.0
        
    ordered = approx.reshape(-1, 2)
    coords = []
    for x in ordered:
        if hasattr(x, 'tolist'):
            coords.append(tuple(x.tolist()))
        else:
            coords.append(tuple(map(float, x)))  # fallback

    # Create a mask the same size as the image, initially black (transparent)
    mask = Image.new("L", semseg_im.size, 0)
    draw = ImageDraw.Draw(mask)

    # Define the four coordinates of your trapezoid.
    # For example: top-left, top-right, bottom-right, bottom-left.
    coords = [tuple(x.tolist()) for x in ordered]
    draw.polygon(coords, fill=255)

    # Create a new image for the output, with a transparent background.
    output = Image.new("RGBA", semseg_im.size, (0, 0, 0, 0))
    # Paste the original image using the mask; only the trapezoid area is pasted.
    output.paste(semseg_im, mask=mask)
    return output

def compute_view_quality():
    """
    Compute the view quality Q = (fraction_nature - fraction_building)
    using segmentation results.
    """
    # For our purpose, we define:
    green_colors = [(4, 200, 3), (4, 250, 7), (143, 255, 140), (255, 0, 0)]
    water_colors = [(61, 230, 250), (9, 7, 230), (10, 190, 212)]
    sky_colors = [(6, 230, 230)]
    building_colors = [(180, 120, 120), (140, 140, 140)]
    
    # define empty positive and negative (building pixels now)
    pos_pixels = 0
    building_pixels = 0
     
    for idx in range(1,10):
        im_path = f"unit_cam{idx}_image.png"
        final_output = semseg_cut_mask(im_path)
        final_output.save(f"cam{idx}_semseg.png")
        img_array = np.array(final_output)
        green_pixel_count = count_pixels_for_colors(img_array, green_colors)
        water_pixel_count = count_pixels_for_colors(img_array, water_colors)
        sky_pixel_count = count_pixels_for_colors(img_array, sky_colors)
        building_pixel_count = count_pixels_for_colors(img_array, building_colors)
        
        pos_pixels += green_pixel_count + water_pixel_count + sky_pixel_count
        building_pixels += building_pixel_count

    total_pixels = 512 * 512 * 9  # 4 cameras
    fraction_nature = pos_pixels / total_pixels
    fraction_building = building_pixels / total_pixels
    return (fraction_nature - fraction_building)

# ------------------------
# Environment Definition
# ------------------------
import gymnasium
from gymnasium import spaces
import bpy
import mathutils
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback


class BlenderEnv(gymnasium.Env):
    """
    Custom Blender environment.
    State: [relative window position (3),. scale (3), seg_nature, seg_building] = 8 dims.
    Action space: 7 discrete actions (0-5: move/scale, 6: finish).
    Uses delta (difference) based dense rewards and final view quality as sparse reward.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        super(BlenderEnv, self).__init__()
        self.config = config or {}
        # self.action_space = spaces.Discrete(9)  # 8 actions + finish
        self.action_space = spaces.Box(
            low=np.array([-4.0, -4.0, 1.0, 1.0]),  # Min values for x, z, width, height
            high=np.array([4.0, 4.0, 8.0, 2.4]),   # Max values for x, z, width, height
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "bird_view": spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8),  # if resized to 64x64, single channel, for example
            "internal": spaces.Box(low=0, high=255, shape=(9, 64, 64), dtype=np.uint8),    # your 9 perspective images
            "state_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)  # where N is the number of normalized variables; e.g., wall/window locations and dimensions
        })
        self._setup_scene()
        self.prev_quality = None
        self.state = self._get_observation()
        self.steps = 0

    def _setup_scene(self):
        if "window" in bpy.data.objects:
            print("Initiating scene...")
            cube = bpy.data.objects["window"]
            cube.location = (0, 0, 0)

    def _get_observation(self):
        from PIL import Image
        import numpy as np

        internal_files = [f"cam{i}_semseg.png" for i in range(1, 10)]
        internal_images = []
        for fname in internal_files:
            try:
                # RGBA 이미지를 그레이스케일로 변환하고, 64x64 사이즈로 리사이즈
                img = Image.open(fname).convert("L").resize((64, 64))
                internal_images.append(np.array(img, dtype=np.uint8))
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                internal_images.append(np.zeros((64, 64), dtype=np.uint8))
        internal_stack = np.stack(internal_images, axis=0)  # shape: (9, 64, 64)
        
        bird_img = Image.open("bird/bird_view_64.png").convert("L").resize((64, 64))
        bird_image = np.array(bird_img, dtype=np.uint8)[None, ...]  # shape: (1, 64, 64)
        
        
        # define window obj and wall obj
        window_obj = bpy.data.objects.get("window")
        wall_obj = bpy.data.objects.get("window_wall")

        win_loc = window_obj.location
        win_dim = window_obj.dimensions
        
        # wall_loc = wall_obj.location
        # wall_dim = wall_obj.dimensions
        
        window_angle = 0
        
        # wall_pixel = np.array([34, 64], dtype=np.float32)
        # wall_pixel_norm = wall_pixel / 64.0
        
        # observation wall/window locations and dimensions
        vector_obs = np.array([win_loc.x, win_loc.y, win_loc.z, win_dim.x, win_dim.z,
                               0.53125, 1.0,
                                # wall_loc.x, wall_loc.y, wall_loc.z, wall_dim.x, wall_dim.z,
                                window_angle], dtype=np.float32)

        
        # Return a dict observation combining both modalities
        return {"internal": internal_stack, "bird_view": bird_image, "state_vec": vector_obs}

    def _compute_reward(self):
        quality = compute_view_quality()
        STEP_PENALTY = 0.01
        
        window_obj = bpy.data.objects["window"]
        dimension_diversity = 0.01 * (abs(window_obj.dimensions.x - 1.0) + abs(window_obj.dimensions.z - 1.0))

        # If previous quality is defined, compute delta reward.
        if self.prev_quality is None:
            delta_reward = 0.0
        else:
            delta_reward = quality - self.prev_quality

        self.prev_quality = quality

        # If finish action is taken, we return the final quality directly.
        return delta_reward - STEP_PENALTY + dimension_diversity, quality

    def _check_done(self):
        return self.steps >= 100

    def step(self, action):
        start = time.time()
        print(f"Step {self.steps}, Action: {action}")
        self.steps += 1

        finish_flag = False
        
        if "window" in bpy.data.objects:
            window_obj = bpy.data.objects["window"]
            window_obj.location.x = action[0]
            window_obj.location.z = action[1]
            window_obj.dimensions.x = action[2]
            window_obj.dimensions.z = action[3]

        bpy.ops.object.visual_transform_apply()
        render_views()
        self.state = self._get_observation()
        # Compute reward; get both delta reward and current quality.
        delta_reward, current_quality = self._compute_reward()

        # If finish action is taken, override reward with final quality (without delta).
        if finish_flag:
            reward = current_quality  # final view quality as sparse reward
        else:
            reward = delta_reward

        done_env = self._check_done() or finish_flag
        info = {}

        # if abs(reward) > 0:
            # save_semseg(self.steps)
        end = time.time()
        with open('step_action.txt', 'a') as f:
            f.write(f"{self.steps}, {action}, {reward}, {current_quality}, {end - start}\n")

        terminated = done_env
        truncated = False

        return self.state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        self.prev_quality = None
        self._setup_scene()
        self.state = self._get_observation()
        return self.state, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

# ------------------------
# Custom Feature Extractor

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

class CustomMultiModalExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for dictionary observations which
    include:
      - "bird_view": a 1-channel image (global view) of shape (1, 64, 64),
      - "internal": a 9-channel image (9 perspectives) of shape (9, 64, 64),
      - "state_vec": a vector of state features (e.g., wall/window state) of shape (8,).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Call parent constructor. The features_dim will be the sum of the feature sizes from the three branches.
        super(CustomMultiModalExtractor, self).__init__(observation_space, features_dim)
        
        # Branch for the bird_view image.
        bird_space = observation_space.spaces["bird_view"]
        self.bird_cnn = nn.Sequential(
            nn.Conv2d(bird_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            dummy_bird = th.zeros(1, *bird_space.shape)
            bird_flat_size = self.bird_cnn(dummy_bird).shape[1]
        self.bird_linear = nn.Sequential(
            nn.Linear(bird_flat_size, 64),
            nn.ReLU()
        )
        
        # Branch for the internal views.
        internal_space = observation_space.spaces["internal"]
        self.internal_cnn = nn.Sequential(
            nn.Conv2d(internal_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            dummy_internal = th.zeros(1, *internal_space.shape)
            internal_flat_size = self.internal_cnn(dummy_internal).shape[1]
        self.internal_linear = nn.Sequential(
            nn.Linear(internal_flat_size, 64),
            nn.ReLU()
        )
        
        # Branch for the vector state.
        state_vec_space = observation_space.spaces["state_vec"]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_vec_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Total combined feature size:
        self._features_dim = 64 + 64 + 32  # You can adjust these numbers as needed.
    
    def forward(self, observations):
        # Process the bird_view image.
        bird_input = observations["bird_view"].float() / 255.0  # Normalize to [0,1]
        bird_features = self.bird_linear(self.bird_cnn(bird_input))
        
        # Process the internal images.
        internal_input = observations["internal"].float() / 255.0
        internal_features = self.internal_linear(self.internal_cnn(internal_input))
        
        # Process the vector state.
        state_features = self.state_mlp(observations["state_vec"].float())
        
        # Concatenate all features.
        return th.cat([bird_features, internal_features, state_features], dim=1)

# ------------------------
# Main training/testing loop
# ------------------------
def main():
    env = BlenderEnv()
    env = Monitor(env, filename="monitor_log.csv")
    
    eval_env = BlenderEnv()
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    policy_kwargs = dict(
        features_extractor_class=CustomMultiModalExtractor,
        features_extractor_kwargs=dict(features_dim=64 + 64 + 32)  # matching _features_dim of the extractor
    )

    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./a2c_tensorboard_log/", device="cuda")
    print("Model policy parameters device:", next(model.policy.parameters()).device)
    # with open("model.txt", "w") as f:
        # f.write(next(model.policy.parameters()).device)
    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save("a2c_cnn_model")

    # Test the model.
    obs, info = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        print("Predicted action:", action)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, info = env.reset()


if __name__ == "__main__":
    main()
