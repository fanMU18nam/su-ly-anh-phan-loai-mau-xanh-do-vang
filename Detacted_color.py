import cv2
import numpy as np
import snap7
from snap7.util import set_bool

# Kết nối với PLC Siemens S7-1200
PLC_IP = "192.168.0.1"
DB_NUMBER = 1
plc = snap7.client.Client()
plc.connect(PLC_IP, 0, 1)

def detect_color(frame):
    """Phát hiện màu sắc trong hình ảnh và loại bỏ nhiễu từ tay người"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Chọn vùng trung tâm ảnh để tránh tay người (ROI)
    h, w, _ = frame.shape
    roi = hsv[h//4: 3*h//4, w//4: 3*w//4]  # Chỉ lấy phần trung tâm ảnh

    # Định nghĩa phạm vi màu HSV (thu hẹp phạm vi để tránh nhiễu)
    color_ranges = {
        "red": ([0, 150, 100], [10, 255, 255]),  # Giảm phạm vi đỏ
        "yellow": ([25, 150, 150], [35, 255, 255]),  # Giảm phạm vi vàng
        "blue": ([100, 170, 50], [130, 255, 255])  # Giữ phạm vi xanh
    }

    detected_colors = {"red": False, "yellow": False, "blue": False}

    # Làm mờ ảnh để giảm nhiễu
    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(roi, np.array(lower), np.array(upper))

        # Loại bỏ nhiễu nhỏ bằng phép mở (morphological open)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Tìm các vùng có diện tích lớn nhất
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Chỉ chấp nhận vùng có diện tích đủ lớn (lọc bỏ vùng nhỏ do nhiễu)
            if area > 1000:
                detected_colors[color] = True

    return detected_colors["red"], detected_colors["yellow"], detected_colors["blue"]

def update_plc(red, yellow, blue):
    """Gửi dữ liệu Boolean đến Data Block trong PLC"""
    data = plc.db_read(DB_NUMBER, 0, 1)  # Đọc trạng thái hiện tại từ DB1

    set_bool(data, 0, 0, red)
    set_bool(data, 0, 1, yellow)
    set_bool(data, 0, 2, blue)

    plc.db_write(DB_NUMBER, 0, data)

# Khởi động Camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện màu trong hình ảnh
    red, yellow, blue = detect_color(frame)

    # Gửi tín hiệu đến PLC
    update_plc(red, yellow, blue)

    # Hiển thị thông tin phát hiện trên màn hình
    cv2.putText(frame, f"Red: {red}, Yellow: {yellow}, Blue: {blue}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Camera", frame)

    # Thoát chương trình khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plc.disconnect()
