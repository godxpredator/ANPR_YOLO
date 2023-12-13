from ultralytics import YOLO
import cv2
import imutils
import easyocr
reader = easyocr.Reader(['en'], gpu=False)

def write(text, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{}\n'.format('Plate_no'))

def draw(img, top_left, bottom_right, vh_type, color=(0, 255, 0), thickness=10, line_length_x=200,
                line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.putText(img, coco_model.names[int(vh_type)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0,0, 255), 3, cv2.LINE_AA)

    return img


img = cv2.imread('img.jpg')
img = imutils.resize(img,width=800)
cv2.imshow("Original Image", img)
cv2.waitKey(0)

coco_model = YOLO('yolov8n.pt')
plate_model = YOLO('last.pt')
vehicle = [2, 3, 5, 7]
object_detection = coco_model(img)[0]

vehicles_loc = []

for detection in object_detection.boxes.data.tolist():
    car_x1, car_y1, car_x2, car_y2, score, class_id = detection
    img = draw(img, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), class_id, (0, 255, 0), 2,
                        line_length_x=200, line_length_y=200)

plate_detection = plate_model(img)[0]

for detection in plate_detection.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection


    license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
    cv2.imshow("plate",license_plate_crop)
    cv2.waitKey(0)

    # process license plate
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh" , license_plate_crop_thresh)
    cv2.waitKey(0)

    # read license plate number
    detections = reader.readtext(license_plate_crop_thresh)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')


    license_plate_text = text
    print(license_plate_text)


    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
    cv2.putText(img,license_plate_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2, cv2.LINE_AA)

    with open('./data.csv', 'w') as f:
        f.write('{}\n'.format(text))


# Show the final image
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()