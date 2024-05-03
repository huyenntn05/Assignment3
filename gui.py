from tkinter import Tk, Button, filedialog
import cv2
from ultralytics import YOLO

def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        return image
    else:
        return None

def detect_objects():
    # Load a pretrained YOLOv8n model
    model = YOLO("model/yolov8n.pt", "v8")

    # Choose the image to detect
    img_source = select_image()

    # Perform object detection
    if img_source is not None:
        detection_output = model.predict(source=img_source, conf=0.25, save=True, show=True)

# Function to close the GUI
def close_gui():
    root.destroy()

# Create the main GUI window
root = Tk()
root.title("Object Detection")

# Position the GUI window in the middle of the screen
window_width = 400
window_height = 100
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (window_width / 2)
y = (screen_height / 2) - (window_height / 2)
root.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y)}")

# Create a button to trigger object detection
detect_button = Button(root, text="Detect Objects", command=detect_objects)
detect_button.pack()

# Create a button to close the GUI
close_button = Button(root, text="Close", command=close_gui)
close_button.pack()

# Run the GUI loop
root.mainloop()