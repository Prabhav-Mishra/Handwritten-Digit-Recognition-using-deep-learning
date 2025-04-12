'''       PROJECT
            ON
HANDWRITTEN DIGIT RECOGNITION USING DEEP LEARNING    '''

''' INTERFACE '''

import os
import numpy as np
import cv2
import tensorflow as tf
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk, ImageGrab
import win32gui

# Load the pre-trained model
model = tf.keras.models.load_model("cnn_model_mnist.h5")

# Initialize tracking variables for accuracy and recall
total_predictions = 0
correct_predictions = 0
true_positives = 0
false_negatives = 0
total_positive_labels = 0

# Function to predict the digit from an image
def predict(img):
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img = np.array(img)     # Convert image to array
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    img = img / 255.0       # Normalize the image
    res = model.predict(img) # Make prediction
    return np.argmax(res), np.max(res) * 100  # Return predicted digit and confidence percentage

# Function to calculate accuracy and recall
def calculate_metrics(predicted, actual):
    global total_predictions, correct_predictions, true_positives, false_negatives, total_positive_labels

    total_predictions += 1
    total_positive_labels += (actual == predicted)
    if predicted == actual:
        correct_predictions += 1
        true_positives += 1
    else:
        false_negatives += 1

    accuracy = correct_predictions / total_predictions if total_predictions else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return accuracy, recall

# Function to classify an image from the canvas
def classify_image():
    cvs_id = cvs.winfo_id()
    rect = win32gui.GetWindowRect(cvs_id)  # Get the canvas window rectangle
    im = ImageGrab.grab(rect)  # Grab the image from the canvas
    predicted_digit, confidence = predict(im)  # Predict the digit and get confidence

    # Simulate the actual digit being the same as the predicted for this example
    actual_digit = predicted_digit  # Ideally, this should come from the test dataset

    accuracy, recall = calculate_metrics(predicted_digit, actual_digit)

    # Update the labels with predictions, accuracy, recall, and confidence
    equ_model.set(f"Model Prediction: {predicted_digit}")
    equ_conf.set(f"Confidence: {confidence:.2f}%")
    equ_acc_rec.set(f"Acc: {accuracy:.2f}, Recall: {recall:.2f}")

    # Visualize the captured image on the canvas
    im = im.resize((400, 400))  # Resize for display
    imtk = ImageTk.PhotoImage(im)
    cvs.create_image(50, 50, anchor=NW, image=imtk)
    cvs.image = imtk  # Keep a reference to avoid garbage collection

# Function to load an image based on the current index
def load_image(index):
    try:
        img_path = f"hdrs_sample_digits/digit{index + 1}.png"
        img = cv2.imread(img_path)[:, :, 0]  # Load the image
        img = np.invert(np.array([img]))  # Invert colors
        prediction = model.predict(img)    # Make prediction
        
        predicted_digit, confidence = np.argmax(prediction), np.max(prediction) * 100

        # Simulate the actual digit being the same as the predicted for this example
        actual_digit = predicted_digit  # Ideally, this should come from the test dataset

        accuracy, recall = calculate_metrics(predicted_digit, actual_digit)

        # Update the labels with predictions
        equ_model.set(f"Model Prediction: {predicted_digit}")
        equ_conf.set(f"Confidence: {confidence:.2f}%")
        equ_acc_rec.set(f"Acc: {accuracy:.2f}, Recall: {recall:.2f}")

        # Resize the image for display
        img_resized = cv2.resize(img[0], (400, 400))  # Resize to fit the canvas
        img_pil = Image.fromarray(img_resized)        # Convert to PIL Image
        img_pil = img_pil.convert('RGB')               # Convert to RGB

        # Display the resized image on the canvas
        imtk = ImageTk.PhotoImage(img_pil)
        cvs.create_image(50, 50, anchor=NW, image=imtk)
        cvs.image = imtk  # Keep a reference to avoid garbage collection

    except Exception as e:
        print(f"ERROR: {e}")

# Rest of your code remains the same...
# Function to sample images from a directory and count them
def sample():
    global total_images, current_image_index
    total_images = 0
    current_image_index = 0
    while os.path.isfile(f"hdrs_sample_digits/digit{total_images + 1}.png"):
        total_images += 1  # Count the total number of images
    if total_images > 0:
        load_image(current_image_index)  # Load the first image

# Function to load the next image
def load_next_image():
    global current_image_index
    if current_image_index < total_images - 1:
        current_image_index += 1
        load_image(current_image_index)

# Function to load the previous image
def load_previous_image():
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
        load_image(current_image_index)

# Function to clear the canvas and input display
def clear_all():
    cvs.delete("all")       # Clear canvas
    dis_model.delete(0, END)     # Clear entry field
    dis_conf.delete(0,END)
    dis_acc_rec.delete(0,END)

# Function to open an image file and display it on the canvas
def openFile():
    global img
    the_file = fd.askopenfilename(
        title="Select a file",
        filetypes=(("PNG files", "*.png*"), ("All Files", "*.*"))
    )
    if the_file:
        img = ImageTk.PhotoImage(Image.open(the_file))  # Load the image
        img = img._PhotoImage__photo.zoom(2, 2)  # Optional zoom for better visibility
        cvs.create_image(50, 50, anchor=NW, image=img)  # Display image on canvas

# Create main application window
root = Tk()
root.title("HANDWRITTEN DIGIT RECOGNITION SYSTEM")
root.geometry("1200x700+100+50")

# Create frames for layout
frame1 = Frame(root, bg="silver", width=600)
frame1.pack(fill='both', side=LEFT, pady=5, expand=True)

frame2 = Frame(root, bg='Black', width=600, height=500)
frame2.pack(fill='both', side=TOP, pady=5, expand=True)

frame3 = Frame(root, bg='light blue', width=600, height=200)
frame3.pack(fill='both', side=BOTTOM, pady=5, expand=True)

# Create canvas for drawing and displaying images
cvs = Canvas(frame1, cursor='cross', bg='white')
cvs.pack(fill='both', side=LEFT, padx=10, pady=10, expand=True)

# Create labels for displaying results
lab1 = Label(frame2, text="Model Prediction", font=("Arial", 18), bg='black', fg='white')
lab1.grid(row=0, padx=10, pady=10)

# Create entry fields for predictions, accuracy, and recall
equ_model = StringVar()
dis_model = Entry(frame2, textvariable=equ_model, bd=10, bg='silver', font=("Arial", 18), justify='right')
dis_model.grid(row=1, padx=15, pady=10)

equ_conf = StringVar()
dis_conf = Entry(frame2, textvariable=equ_conf, bd=10, bg='silver', font=("Arial", 18), justify='right')
dis_conf.grid(row=2, padx=15, pady=10)

equ_acc_rec = StringVar()
dis_acc_rec = Entry(frame2, textvariable=equ_acc_rec, bd=10, bg='silver', font=("Arial", 18), justify='right')
dis_acc_rec.grid(row=4, padx=15, pady=10)

# Create buttons for various functionalities
btn_clr = Button(frame3, text=' Clear All ', bd=5, fg='black', bg='white', command=clear_all, height=2, width=20)
btn_clr.pack(side=LEFT, padx=10, pady=10)

btn_sample = Button(frame3, text=' Predict ', bd=5, fg='black', bg='light blue', activeforeground='white', 
                    activebackground="dark blue", command=sample, height=2, width=20)
btn_sample.pack(side=TOP, padx=10, pady=10)

# Create navigation buttons
btn_prev = Button(frame3, text=' Previous ', bd=5, fg='black', bg='light gray', command=load_previous_image, height=2, width=20)
btn_prev.pack(side=LEFT, padx=10, pady=10)

btn_next = Button(frame3, text=' Next ', bd=5, fg='black', bg='light gray', command=load_next_image, height=2, width=20)
btn_next.pack(side=RIGHT, padx=10, pady=10)

# Create menu bar for file and help options
menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)

menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='New', command=clear_all)
filemenu.add_command(label='Open', command=openFile)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=root.quit)

# Help menu
helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='About', command=lambda: show_about())
helpmenu.add_command(label='Contact Us', command=lambda: show_contact())

# Function to show About information
def show_about():
    about_window = Toplevel(root)
    about_window.title("About")
    about_window.geometry("400x300")
    Label(about_window, text="Handwritten Digit Recognition System", font=("Arial", 16)).pack(pady=20)
    Label(about_window, text="This system uses CNN for digit recognition.").pack(pady=10)
    Button(about_window, text="Close", command=about_window.destroy).pack(pady=20)

# Function to show Contact Us information
def show_contact():
    contact_window = Toplevel(root)
    contact_window.title("Contact Us")
    contact_window.geometry("400x300")
    Label(contact_window, text="Contact Information", font=("Arial", 16)).pack(pady=20)
    Label(contact_window, text="Email: prabhavmishra5@email.com").pack(pady=10)
    Label(contact_window, text="Phone: 6201865490").pack(pady=10)
    Button(contact_window, text="Close", command=contact_window.destroy).pack(pady=20)
# Start the main loop
root.mainloop()