import cv2
import os
import pandas as pd
import numpy as np

IMAGE_FOLDER = "dataset/"
OUTPUT_CSV = "index.csv"
DISPLAY_SIZE = (800, 600)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.6 
FONT_SCALE_CAPTION = 0.8 
FONT_COLOR = (255, 255, 255) 
FONT_THICKNESS = 1
BACKGROUND_COLOR = (0, 0, 0) 

if os.path.exists(OUTPUT_CSV):
    df = pd.read_csv(OUTPUT_CSV)
    done_files = set(df['filepath'])
else:
    df = pd.DataFrame(columns=['filepath', 'caption'])
    done_files = set()

images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

cv2.namedWindow("Image Tagger")

for filename in images:
    full_path = os.path.join(IMAGE_FOLDER, filename)
    if full_path in done_files:
        continue

    img_original = cv2.imread(full_path)
    if img_original is None:
        print(f"Skipping unreadable file: {full_path}")
        continue

    #preserving aspect ratio during resize
    h, w = img_original.shape[:2]
    aspect_ratio = w / h
    if w > h:
        new_w = DISPLAY_SIZE[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = DISPLAY_SIZE[1]
        new_w = int(new_h * aspect_ratio)
    resized_img = cv2.resize(img_original, (new_w, new_h))

    current_caption = ""
    while True:
        display_img = resized_img.copy()

        cv2.putText(display_img, f"File: {filename}", (10, 30), FONT, FONT_SCALE_SMALL, FONT_COLOR, FONT_THICKNESS)

        instructions = "ESC: Quit | ENTER: Save | BACKSPACE: Del | Type:"
        cv2.putText(display_img, instructions, (10, display_img.shape[0] - 60), FONT, FONT_SCALE_SMALL, FONT_COLOR, FONT_THICKNESS)

        (text_width, text_height), baseline = cv2.getTextSize(current_caption, FONT, FONT_SCALE_CAPTION, FONT_THICKNESS)
        text_x = 10
        text_y = display_img.shape[0] - 20

        cv2.rectangle(display_img, (text_x - 5, text_y - text_height - 5),
                      (text_x + text_width + 5, text_y + baseline + 5), BACKGROUND_COLOR, -1)
        cv2.putText(display_img, current_caption, (text_x, text_y), FONT, FONT_SCALE_CAPTION, FONT_COLOR, FONT_THICKNESS)

        cv2.imshow("Image Tagger", display_img)

        key = cv2.waitKey(1)

        if key == 27: # esc key
            print("Operation cancelled by user.")
            cv2.destroyAllWindows()
            exit()
        elif key == 13: # enter key
            if current_caption.strip():
                df.loc[len(df)] = [full_path, current_caption.strip()]
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"Saved caption for '{filename}': '{current_caption.strip()}'")
            else:
                print(f"Skipping '{filename}' - no caption entered.")
            break
        elif key == 8: # backspace
            current_caption = current_caption[:-1]
        elif key != -1:
            if 32 <= key <= 126:
                current_caption += chr(key)
            elif key == 95: 
                current_caption += '_'
            elif key == 45: 
                current_caption += '-'

cv2.destroyAllWindows()
print("✅ All images labeled and saved.")