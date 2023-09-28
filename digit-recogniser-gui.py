import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import tensorflow.keras as keras
from tensorflow.python.keras.callbacks import EarlyStopping 
import numpy as np

FONT = ("Helvetica", 48)

model = keras.models.load_model("mnist-cnn.h5")


class Classifier:
    @staticmethod
    def predict_digit(image):
        image = image.resize((28, 28))
        image = image.convert('L')
        image = np.array(image)
        image = image.reshape(1, 28, 28, 1)
        image /= 255.0
        image = 1 - image

        res = model.predict([image])[0]
        return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0

        self.canvas = tk.Canvas(self, width=400, height=400, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking...", font=FONT)
        self.classify_button = tk.Button(self, text="Classify", command=Classifier.predict_digit)
        self.clear_button = tk.Button(self, text="Clear",command=self.clear_all)
        
        self.canvas.grid(row=0, column=0, pady=2, sticky="W")
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_button.grid(row=1, column=1, pady=2)
        self.clear_button.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", func=self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_digit(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        img = ImageGrab.grab(rect)
        digit, accuracy = Classifier.predict_digit(img)
        self.label.configure(text=f"{digit}, {int(accuracy * 100)}%")

    def draw_lines(self, e):
        self.x = e.x
        self.y = e.y
        radius = 7
        self.canvas.create_oval(self.x - radius, self.y - radius, self.x + radius, self.y + radius, fill = 'black')
        self.canvas.update()

def main():
    app = App()
    app.mainloop()

if __name__ == '__main__':
    main()