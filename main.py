
# # ==============================
# # Step 3b: Test Conditioning Encoder Visualization
# # File: main.py (append this test)
# # ==============================
# from PIL import Image
# import numpy as np
# from perception.yolo_detector import YOLODetector
# from diffusion.conditioned_generator import ConditionedDiffusionGenerator

# # -----------------------------
# # 1. بارگذاری تصویر اصلی
# # -----------------------------
# image_path = "data/inputs/test.jpeg"
# init_image = Image.open(image_path).convert("RGB")

# # -----------------------------
# # 2. تشخیص اشیا با YOLO
# # -----------------------------
# detector = YOLODetector(model_name="yolov8n.pt", device="cpu")
# results = detector.detect(init_image)
# print("Detected objects:", results["objects"])

# # -----------------------------
# # 3. ساخت layout_map از نتایج YOLO
# # -----------------------------
# W, H = init_image.size
# layout_map = np.zeros((512, 512), dtype=np.float32)

# for obj in results["objects"]:
#     if obj["class"] == "person":
#         x1, y1, x2, y2 = obj["bbox"]
#         x1 = int((x1 / W) * 512)
#         x2 = int((x2 / W) * 512)
#         y1 = int((y1 / H) * 512)
#         y2 = int((y2 / H) * 512)
#         layout_map[y1:y2, x1:x2] = 1.0

# # -----------------------------
# # 4. تولید تصویر با ControlNet
# # -----------------------------
# generator = ConditionedDiffusionGenerator(device="cpu")

# prompt = "A realistic portrait of the person, cinematic lighting, detailed face"
# negative_prompt = "blurry, low quality, deformed face, cartoon, anime"

# final_img = generator.generate(
#     prompt=prompt,
#     init_image=init_image,
#     layout_map=layout_map,
#     num_inference_steps=30,
#     guidance_scale=7.5,
#     negative_prompt=negative_prompt
# )

# final_img.save("data/outputs/result.png")
# final_img.show()



#--------------------------------------------------
# background change

from PIL import Image
import os

from perception.yolo_detector import YOLODetector
from diffusion.background_generator import BackgroundGenerator


def main():

    # -------------------------
    # 1️⃣ Load input image
    # -------------------------
    input_path = "data/inputs/test.jpeg"
    image = Image.open(input_path).convert("RGB")

    # -------------------------
    # 2️⃣ Extract person (Segmentation)
    # -------------------------
    detector = YOLODetector(model_name="yolov8n-seg.pt")
    person_rgba, mask = detector.extract_main_person(image)

    os.makedirs("outputs", exist_ok=True)
    person_rgba.save("outputs/person_cutout.png")

    print("✅ Person extracted")

    # -------------------------
    # 3️⃣ Generate new background
    # -------------------------
    # مسیر دقیق پوشه‌ای که همین الان آماده کردید
    model_path = "stable-diffusion-v1-5" 

    bg_generator = BackgroundGenerator(
        model_path=model_path,
        device="mps"
    )

    prompt = "a futuristic neon city at night, cinematic lighting, ultra realistic"

    background = bg_generator.generate(
        prompt=prompt,
        width=512,
        height=512
    ).convert("RGBA")

    background.save("data/outputs/generated_background.png")

    print("✅ Background generated")

    # -------------------------
    # 4️⃣ Resize person
    # -------------------------
    person_resized = person_rgba.resize((300, 500))

    # -------------------------
    # 5️⃣ Paste person onto background
    # -------------------------
    position = (100, 20)   # تغییر مکان انسان

    background.paste(person_resized, position, person_resized)

    # -------------------------
    # 6️⃣ Save final result
    # -------------------------
    final_path = "data/outputs/final_result.png"
    background.save(final_path)

    print("🔥 Final image saved at:", final_path)


if __name__ == "__main__":
    main()