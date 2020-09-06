# Importing the required modules

# To avoid getting info, warnings and error messages from tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For loading and showing images
import cv2
import matplotlib.pyplot as plt

# For Frontend
import pygame
from tkinter import Tk, messagebox

# For data manipulation
import numpy as np

# For CNN Model
import tensorflow as tf


class Pixel(object):
    """Pixel Class for representing the pixel values"""

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 255, 255)
        self.neighbors = []

    def draw(self, surface):
        """
        Method for drawing the pixel on the pygame window
        """
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def getNeighbors(self, g):
        """
        Method for getting the neighbors of the pixel
        """

        # As the width and height of the pygame window is (560, 560)
        # We are dividing it by 20 to get value representation for (28, 28)

        j = self.x // 20
        i = self.y // 20

        # Total number of rows and columns
        rows = 28
        cols = 28

        # Horizontal and Vertical Neighbors
        if i < cols - 1:    # Right
            self.neighbors.append(g.pixels[i + 1][j])

        if i > 0:   # Left
            self.neighbors.append(g.pixels[i - 1][j])

        if j < rows - 1:    # Up
            self.neighbors.append(g.pixels[i][j + 1])

        if j > 0:   # Down
            self.neighbors.append(g.pixels[i][j - 1])

        # Diagonal Neighbors
        if j > 0 and i > 0: # Top Left
            self.neighbors.append(g.pixels[i - 1][j - 1])

        if j < rows - 1 and i > 0:  # Bottom Left
            self.neighbors.append(g.pixels[i - 1][j + 1])

        if j > 0 and i < cols - 1:  # Top Right
            self.neighbors.append(g.pixels[i + 1][j - 1])

        if j < rows - 1 and i < cols - 1:   # Bottom Right
            self.neighbors.append(g.pixels[i + 1][j + 1])

class Grid(object):
    """
    Grid class for representing a grid of pixel objects
    """

    # List for holding the pixel objects
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generatePixels()

    def draw(self, surface):
        """
        Method which draws the grid on the pygame window
        """
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generatePixels(self):
        """
        Method which resets the grid, by generating a new grid
        """
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows

        self.pixels = []

        for r in range(self.rows):
            self.pixels.append([])

            for c in range(self.cols):
                self.pixels[r].append(Pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
             for c in range(self.cols):
                 self.pixels[r][c].getNeighbors(self)

    def clicked(self, pos):
        """
        Method which return the position of the pixel
        which has been clicked within the grid
        """
        try:
            h = pos[0]
            w = pos[1]

            p1 = int(h) // self.pixels[0][0].width
            p2 = int(w) // self.pixels[0][0].height

            return self.pixels[p2][p1]

        except Exception as e:
            print(e)
            pass

    def convert_binary(self):
        """
        Method which converts the image into a grayscale image

        Note : As the models is trained in a binary image, 
        the image for prediction needs to be a grayscale image
        """

        lst = self.pixels

        Matrix = [[] for x in range(len(lst))]

        for i in range(len(lst)):
            for j in range(len(lst)):
                if lst[i][j].color == (255, 255, 255):
                    Matrix[i].append(0)
                else:
                    Matrix[i].append(1)

        plt.imsave("./imgs/img.jpg", Matrix, cmap = "gray")

        return Matrix

def predict(lst):
    """
    Function which predicts the number
    """

    # Loading the serialized CNN model
    model = tf.keras.models.load_model('./model/CNN_MNIST.h5')

    # Adding Batch and Channel Dimension
    lst = np.expand_dims(lst, axis = [0, -1])

    # Predicting the Number (or) Forward Pass of the image
    predictions = model.predict(lst)

    # print("Output Layer Activations = ", predictions)

    # Taking the element with the highest value from the prediction array
    num = np.argmax(predictions)

    # Prints the prediction on to the console
    print("The Number is predicted by the model as :", num)

    # Initalizing Tkinter window
    window = Tk()

    # Hiding Tkinter window, as we only need the messagebox
    window.withdraw()

    # Tkinter Messagebox for showing prediction
    messagebox.showinfo("Prediction", "The Number is predicted by the model as : " + str(num))

    # Destroying all the tkinter windows created
    window.destroy()

# Main Loop
if __name__ == "__main__":

    # Initializing the pygame window
    pygame.init()

    # Width, Height of the pygame window
    width = height = 560

    # Initializing the pygame window
    win = pygame.display.set_mode((width, height))

    # Title of Pygame Window
    pygame.display.set_caption("MNIST Digit Recognizer")

    # Initializing the Grid Class
    g = Grid(28, 28, width, height)

    run = True
    while run:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

            # Listens if any keyboard button is pressed
            if event.type == pygame.KEYDOWN:
                lst = g.convert_binary()
                predict(cv2.imread("./imgs/img.jpg", 0))
                g.generatePixels()

            # Listens for mouse left button clicks
            if pygame.mouse.get_pressed()[0]:

                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)

                clicked.color = (0, 0, 0)

                for n in clicked.neighbors:
                    n.color = (0, 0, 0)

            # Listens for mouse right button clicks
            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255, 255, 255)
                except Exception as e:
                    print(e)
                    pass
        
        # Resetting the pygame window
        g.draw(win)
        pygame.display.update()

    pygame.quit()
    quit()