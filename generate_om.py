import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def create_glowing_om(output_path="om_glow.png", size=(250, 250)):
    # 1. Setup Canvas (Transparent)
    W, H = size
    # Create a large canvas for high quality downscaling if needed, but 250 is fine
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 2. Load Font (Try standard Windows fonts that support Devanagari)
    font_names = ["Nirmala.ttf", "NirmalaB.ttf", "arial.ttf", "seguiemj.ttf", "mangal.ttf"]
    font = None
    for name in font_names:
        try:
            # Load font with size ~80% of canvas
            font = ImageFont.truetype(name, int(H * 0.8))
            break
        except IOError:
            continue
    
    if font is None:
        print("[ERROR] Could not load a suitable font for Om symbol.")
        return None

    text = "ॐ"
    
    # Get text size to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (W - text_w) // 2
    y = (H - text_h) // 2 - int(H * 0.1) # Shift up slightly

    # 3. Create Glow Layers
    # We draw the text multiple times with different blurs
    glow_color = (255, 215, 0) # Gold
    core_color = (255, 255, 255) # White core

    # Layer 1: Wide Glow (faint)
    glow1 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(glow1)
    d1.text((x, y), text, font=font, fill=glow_color + (100,)) # Semi-transparent
    glow1 = glow1.filter(ImageFilter.GaussianBlur(radius=15))

    # Layer 2: Medium Glow (stronger)
    glow2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(glow2)
    d2.text((x, y), text, font=font, fill=glow_color + (180,))
    glow2 = glow2.filter(ImageFilter.GaussianBlur(radius=8))

    # Layer 3: Sharp Glow / Inner Stroke
    glow3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d3 = ImageDraw.Draw(glow3)
    d3.text((x, y), text, font=font, fill=glow_color + (255,))
    glow3 = glow3.filter(ImageFilter.GaussianBlur(radius=2))

    # Layer 4: White Core (Sharpest)
    core = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d4 = ImageDraw.Draw(core)
    d4.text((x, y), text, font=font, fill=core_color + (255,))

    # 4. Composite
    final = Image.alpha_composite(img, glow1)
    final = Image.alpha_composite(final, glow2)
    final = Image.alpha_composite(final, glow3)
    final = Image.alpha_composite(final, core)

    # 5. Save
    final.save(output_path)
    print(f"[SUCCESS] Saved glowing Om to {output_path}")
    return final

if __name__ == "__main__":
    create_glowing_om()
