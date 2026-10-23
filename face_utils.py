
# import cv2

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades
#     + "haarcascade_frontalface_default.xml"
# )

# def crop_face(image, margin=0.2):
#     """
#     Detect the largest face and crop it with a margin.
#     Returns None if no face is detected.
#     """
#     if image is None:
#         return None

#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(60, 60)
#     )

#     if len(faces) == 0:
#         return None

#     x, y, w, h = max(
#         faces,
#         key=lambda b: b[2] * b[3]
#     )

#     h_img, w_img = image.shape[:2]

#     x_margin = int(w * margin)
#     y_margin = int(h * margin)

#     x1 = max(0, x - x_margin)
#     y1 = max(0, y - y_margin)
#     x2 = min(w_img, x + w + x_margin)
#     y2 = min(h_img, y + h + y_margin)

#     cropped = image[y1:y2, x1:x2]

#     if cropped.size == 0:
#         return None

#     return cropped
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(image_bgr, margin=0.25):
    """
    Detects and crops the primary face with a margin.
    Uses multi-scale detection and relaxed thresholds for tilted/glasses faces.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram so harsh outdoor light doesn't blind the detector
    gray_eq = cv2.equalizeHist(gray)

    # 1. First attempt: standard frontal face
    faces = face_cascade.detectMultiScale(
        gray_eq, 
        scaleFactor=1.1, 
        minNeighbors=4, 
        minSize=(60, 60)
    )

    # 2. Second attempt: relaxed threshold if tilted or wearing glasses
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(50, 50)
        )

    # If still no face detected, perform a center-crop focus rather than passing full landscape
    if len(faces) == 0:
        h, w = image_bgr.shape[:2]
        # Crop the upper-middle region where a selfie face naturally sits
        top = int(h * 0.10)
        bottom = int(h * 0.70)
        left = int(w * 0.15)
        right = int(w * 0.85)
        return image_bgr[top:bottom, left:right]

    # Pick the largest detected face box
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Add a margin around the face (hairline, chin, ears)
    img_h, img_w = image_bgr.shape[:2]
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    face_crop = image_bgr[y1:y2, x1:x2]
    return face_crop if face_crop.size > 0 else image_bgr