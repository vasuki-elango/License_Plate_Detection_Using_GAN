from ultralytics import YOLO
import cv2
import csv
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sort.sort import *
from util import get_car, read_license_plate, write_csv, estimate_speed

results = {}

mot_tracker = Sort()
prev_bbox_centers = {}

# load model
coco_model = YOLO("models/yolov8n.pt")
license_plate_detector = YOLO('models/license_plate_detector.pt')

# Define speed limit
speed_limit = 80  # km/h

#Email Configuration
SMTP_SERVER = "smtp.gmail.com"  # For Gmail (Change for Outlook, Yahoo, etc.)
SMTP_PORT = 587
SENDER_EMAIL = "j.e.vasuki@gmail.com"
SENDER_PASSWORD = "wvvw mmhm csas axze"
RECIPIENT_EMAIL = "21urcs053@aaacet.ac.in"  # Change this to the recipient's email

# load video
cap = cv2.VideoCapture('source/sample.mp4')
video_fps = cap.get(cv2.CAPv_PROP_FPS)
vehicles = [2, 3, 5, 7]

frame_nmr = -1

# Ensure fines.csv exists with headers
fine_file = 'fines_data.csv'
if not os.path.exists(fine_file):
    with open(fine_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['car_id', 'license_plate', 'speed', 'fine_amount', 'timestamp'])


# Function to send email alert
def send_email_alert(license_plate, speed, fine_amount):
    """Sends an email notification for a speed violation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = "🚨 Speed Violation Alert 🚨"
    body = f"""🚨 Speed Violation Alert 🚨  
Vehicle Plate: {license_plate}  
Detected Speed: {speed} km/h (Limit: {speed_limit} km/h)  
Fine Amount: ₹{fine_amount}  
Timestamp: {timestamp}  
"""

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print(f"✅ Email alert sent for {license_plate} (Speed: {speed} km/h)")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# Function to save fined vehicles separately
def save_fine(car_id, license_plate, speed, fine_amount):
    """Save fine details for a car ID if not already recorded and send email alert."""
    already_fined = False

    # Read existing fines to avoid duplicates
    with open(fine_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row and row[0] == str(car_id):
                already_fined = True
                break

    # If not already fined, append to fines.csv and send email
    if not already_fined:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(fine_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([car_id, license_plate, speed, fine_amount, timestamp])

        send_email_alert(license_plate, speed, fine_amount)  # Send email alert

# read frames
prev_frame = None
prev_results = None
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect licence plate
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    # Get the speed and license plate information for the car
                    car_data = {
                        'locations': track_ids,  # Assuming you have the car's locations over time in track_ids
                    }
                    car_info = estimate_speed(car_id, car_data)

                    speed = int(car_info['speed_label'].replace("km/h", "")) if "km/h" in car_info['speed_label'] else 0
                    # Fine calculation
                    fine_amount = 0
                    if speed > speed_limit:
                        if speed <= speed_limit + 10:
                            fine_amount = 500
                        elif speed <= speed_limit + 20:
                            fine_amount = 1000
                        else:
                            fine_amount = 2000
                        save_fine(car_id, license_plate_text, speed, fine_amount)

                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'car_speed': car_info['speed_label'],
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                          'text': license_plate_text,
                                          'bbox_score': score,
                                          'text_score': license_plate_text_score}}

# write results
write_csv(results, 'Result/speed_test.csv')