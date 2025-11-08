# ------------------------
# Semantic Segmentation
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
    mask_path = im_path.replace("rgb_", "mask_")
    
    im = Image.open(mask_path).convert("RGBA")
    im_np = np.array(im)

    r, g, b = im_np[:, :, 0], im_np[:, :, 1], im_np[:, :, 2]
    mask = (r > 200) & (g < 50) & (b < 50)

    im_np[~mask, 3] = 0

    alpha = im_np[:, :, 3]

    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No valid contours found")
        return None

    h, w = binary.shape
    center = np.array([w/2, h/2])

    def contour_center(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return np.array([0, 0])
        return np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])

    centers = [contour_center(c) for c in contours]
    distances = [np.linalg.norm(c - center) for c in centers]
    closest_idx = int(np.argmin(distances))
    selected_contour = contours[closest_idx]

    epsilon = 0.01 * cv2.arcLength(selected_contour, True)
    approx = cv2.approxPolyDP(selected_contour, epsilon, True)
    if len(approx) < 3:
        print("Warning: Not enough points for polygon approximation")
        return None

    ordered = approx.reshape(-1, 2)
    coords = [tuple(x) for x in ordered]
    for x in ordered:
        if hasattr(x, 'tolist'):
            coords.append(tuple(x.tolist()))
        else:
            coords.append(tuple(map(float, x)))  # fallback

    mask = Image.new("L", semseg_im.size, 0)
    draw = ImageDraw.Draw(mask)

    coords = [tuple(x.tolist()) for x in ordered]
    draw.polygon(coords, fill=255)

    output = Image.new("RGBA", semseg_im.size, (0, 0, 0, 0))
    output.paste(semseg_im, mask=mask)
    return output

def compute_view_quality():
    """
    Compute the view quality Q = (fraction_nature - fraction_building)
    using segmentation results.
    """
    green_colors = [(4, 200, 3), (4, 250, 7), (143, 255, 140), (255, 0, 0)]
    water_colors = [(61, 230, 250), (9, 7, 230), (10, 190, 212)]
    sky_colors = [(6, 230, 230)]
    building_colors = [(180, 120, 120), (140, 140, 140)]
    
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


import os

for fname in fnames:
    save_name = fname.replace('\\rgb_', '\\semseg_')
    if not os.path.exists(save_name):
        result = semseg_cut_mask(fname)
        result.save(save_name)
