import customtkinter as ctk 
import tkinter as tk 
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf 

model = tf.keras.models.load_model('mnist_model_test.h5')

class_names = ['zero','one','two','three','four','five','six','seven','eight','nine']

CANVAS_WIDTH , CANVAS_HEIGHT = 256, 256

class DrawingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('Live Drawing Prediction')
        self.geometry('800x600')

        self.title_label = ctk.CTkLabel(self, text='Live Drawing Prediction', font=('Arial',24))
        self.title_label.pack(pady=10)

        self.canvas = tk.Canvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg = 'black',cursor= 'cross')
        self.canvas.pack(pady=10)
        self.canvas.bind('<B1-Motion>',self.draw)  # left mouse to draw
        self.canvas.bind('<Button-3>',self.clear_canvas) # right mouse to erase

        self.results_text = tk.StringVar()
        self.results_labels = ctk.CTkLabel(self, textvariable = self.results_text,font=('Arial', 16), justify='left')
        self.results_labels.pack(pady=10)

        # Display controls
        self.controls_label = ctk.CTkLabel(
            self,
            text="Left mouse: Draw  |  Right mouse: Erase  |  Clear: C",
            font=("Arial", 14),
        )
        self.controls_label.pack(pady=5)

        # Image for drawing
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=0)  # Grayscale image
        self.draw_obj = ImageDraw.Draw(self.image)

        # Start live prediction loop
        self.update_predictions()

    def draw(self, event):
        """Draw on the canvas."""
        x, y = event.x, event.y
        radius = 4  # Thickness of the stroke
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
        self.draw_obj.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)


    def clear_canvas(self, event=None):
        """Clear the entire canvas."""
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=0)  # Reset image
        self.draw_obj = ImageDraw.Draw(self.image)

    def preprocess_image(self):
        """Preprocess the drawn image for model prediction."""
        # Resize the image to the model's input size (e.g., 32x32)
        resized_image = self.image.resize((28, 28))  # Adjust size based on your model
        image_array = np.array(resized_image) / 255.0  # Normalize the pixel values
        return np.expand_dims(image_array, axis=(0, -1))  # Add batch and channel dimensions

    def update_predictions(self):
        """Continuously predict the drawing on the canvas."""
        processed_image = self.preprocess_image()
        predictions = model.predict(processed_image)
        probabilities = predictions[0] * 100  # Convert to percentages

        # Combine class names and probabilities
        results = [(class_names[i], round(probabilities[i], 2)) for i in range(len(class_names))]
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability

        # Update results display
        self.results_text.set("\n".join([f"{name}: {prob}%" for name, prob in results[:9]]))  # Show top 9 predictions

        # making the predictions 'after' every time interval given .
        self.after(400, self.update_predictions)  # Update every 400ms


# Run the app
if __name__ == "__main__":
    app = DrawingApp()
    app.mainloop()