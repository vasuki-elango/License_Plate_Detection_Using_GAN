import string
import easyocr
import math
from collections import deque
import re

data_deque = {}

speed_line_queue = {}

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'car_speed',
                                                   'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                   'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                               car_id,
                                                               '[{} {} {} {}]'.format(
                                                                   results[frame_nmr][car_id]['car']['bbox'][0],
                                                                   results[frame_nmr][car_id]['car']['bbox'][1],
                                                                   results[frame_nmr][car_id]['car']['bbox'][2],
                                                                   results[frame_nmr][car_id]['car']['bbox'][3]),
                                                               results[frame_nmr][car_id]['car_speed'],
                                                               '[{} {} {} {}]'.format(
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       0],
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       1],
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       2],
                                                                   results[frame_nmr][car_id]['license_plate']['bbox'][
                                                                       3]),
                                                               results[frame_nmr][car_id]['license_plate'][
                                                                   'bbox_score'],
                                                               results[frame_nmr][car_id]['license_plate']['text'],
                                                               results[frame_nmr][car_id]['license_plate'][
                                                                   'text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def estimatespeed(Location1, Location2):
    # Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6
    # distance = speed/time
    speed = d_meters * time_constant

    return int(speed)


def estimate_speed(car_id, car_data):
    global data_deque, speed_line_queue

    # Get the track_ids (locations) for the given car_id
    track_ids = car_data['locations']

    # Remove tracked point from buffer if the object is lost
    if car_id not in track_ids[:, -1]:
        if car_id in data_deque:
            data_deque.pop(car_id)

    x1, y1, x2, y2, _ = track_ids[-1]

    # Code to find the center of the bottom edge
    center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

    # Create a new buffer for the object if it doesn't exist
    if car_id not in data_deque:
        data_deque[car_id] = deque(maxlen=64)
        speed_line_queue[car_id] = []

    # Add center to the buffer
    data_deque[car_id].appendleft(center)
    if len(data_deque[car_id]) >= 2:
        object_speed = estimatespeed(data_deque[car_id][1], data_deque[car_id][0])
        speed_line_queue[car_id].append(object_speed)

    # Calculate and display average speed label for the tracked object
    speed_label = "No speed data available"
    if car_id in speed_line_queue and len(speed_line_queue[car_id]) > 0:
        speed_label = str(sum(speed_line_queue[car_id]) // len(speed_line_queue[car_id])) + "km/h"

    return {'speed_label': speed_label, 'license_plate_info': None}


def extract_numeric_values(string):
    def decode_bytes(string):
        if isinstance(string, bytes):
            return string.decode('utf-8')
        elif isinstance(string, str):
            return string
        elif isinstance(string, list):
            return [decode_bytes(item) for item in string]
        elif isinstance(string, tuple):
            return tuple(decode_bytes(item) for item in string)
        elif isinstance(string, dict):
            return {decode_bytes(key): decode_bytes(value) for key, value in string.items()}
        else:
            return string

        # Decode any bytes-like objects in the input data

    decoded_data = decode_bytes(string)

    # Define the regular expression pattern to match valid numeric values
    pattern = r'\d+'

    # Use the findall method to find all occurrences of the pattern in the string
    numeric_values = re.findall(pattern, decoded_data)

    # Convert the extracted numeric values to floats (if there are decimals) or integers
    numeric_values = [float(value) if '.' in value else int(value) for value in numeric_values]

    return numeric_values