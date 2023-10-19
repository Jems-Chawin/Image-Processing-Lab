import cv2
import os

# อ่านไฟล์ภาพโดยใช้ cv2
image1 = cv2.imread("../images/bird.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("../images/galaxy.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# ตรวจสอบว่า ภาพถูกอ่านสำเร็จหรือไม่
if image1 is None or image2 is None:
    print("Error: Image(s) not found.")
    exit()

# check ขนาดของทั้งสองไฟล์
if image1.shape != image2.shape:
    print("Error: Images have different dimensions.")
    exit()


# กำหนด weight
weights = [0, 1]

# ทำการ add weight ด้วยค่าที่ได้กำหนดไว้
result = cv2.addWeighted(image1, weights[0], image2, weights[1], 0)

# สร้าง object สำหรับ video writer
output_file = os.path.expanduser("~/Desktop/added_images_video.mp4")
fps = 60.0
frame_width, frame_height = image1.shape[1], image1.shape[0]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), isColor=False)

# ตรวจสอบว่าทำการสร้าง video สำเร็จหรือไม่
if not out.isOpened():
    print("Error: Failed to create VideoWriter.")
    exit()

# กำหนดจำนวนเฟรม
num_frames = 200
# ลูป for เพื่อเพิ่มค่า i ทำให้ weight_image2 ค่อยๆเพิ่มขึ้นจน weight_image1 เหลือ 0
for i in range(num_frames):
    weight_image2 = i / num_frames

    weight_image1 = 1 - weight_image2

    # ทำการ add weight เรื่อยๆในทุกค่าของ i
    result = cv2.addWeighted(image1, weight_image1, image2, weight_image2, 0)

    # write ไฟล์
    out.write(result)

# จบการ write ด้วยฟังก์ชัน release()
out.release()

print(f"Video written to {output_file}")