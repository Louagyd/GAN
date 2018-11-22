import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.colorchooser import *
from PIL import Image, ImageTk
import os

class Visualization_Form:
    def __init__(self, GAN_config, num_sliders = None, slider_width = 300, gen_fn = None):
        self.GAN = GAN_config
        self.z_init = np.random.normal(0, self.GAN["z_sd"], self.GAN["z_len"])
        if num_sliders is None:
            self.num_sliders = self.GAN["z_len"]
        else:
            self.num_sliders = num_sliders

        self.root = Tk()
        self.slider_width = slider_width
        self.sliders_per_grid = self.num_sliders // 9
        self.remaining_sliders = self.num_sliders - 9*self.sliders_per_grid
        self.gen_fn = gen_fn

        self.sliders = []
        for i in range(self.num_sliders):
            if i < self.sliders_per_grid * 2:
                this_row = i//2
                this_col = (i % 2)*2
            else:
                new_i = i - self.sliders_per_grid * 2
                this_row = new_i//3 + self.sliders_per_grid
                this_col = new_i % 3

            this_slider = Scale(self.root, from_=-1, to=1, resolution = 0.01, length = slider_width, orient = HORIZONTAL)
            this_slider.grid(row = this_row, column = this_col)
            self.sliders.append(this_slider)

        self.face_label = Label()
        self.face_label.grid(row=0, column=1, rowspan = self.sliders_per_grid, sticky = W+E+N+S)

        for i in range(self.num_sliders):
            self.sliders[i].configure(command=self.set_image)
            self.sliders[i].command = self.set_image

        self.set_image()

        self.indexes_text = Text(self.root, width = 0, height = 0)
        self.indexes_text.grid(row = 3*self.sliders_per_grid+1, column = 0, sticky = W+E+N+S)
        Button(text='Randomize Inserted Indexes', command=self.random_indexes).grid(row=3*self.sliders_per_grid+1, column = 1, sticky = W+E+N+S)
        Button(text='Reset Inserted Indexes', command=self.reset_indexes).grid(row=3*self.sliders_per_grid+1, column = 2, sticky = W+E+N+S)
        Button(text='Randomize This Indexes', command=self.random_sliders).grid(row=3*self.sliders_per_grid+2, column = 0, sticky = W+E+N+S)
        Button(text='Randomize All Indexes', command=self.random_all).grid(row=3*self.sliders_per_grid+2, column = 1, sticky = W+E+N+S)
        Button(text='Randomize Other Indexes', command=self.random_other).grid(row=3*self.sliders_per_grid+2, column = 2, sticky = W+E+N+S)
        Button(text='Reset This Indexes', command=self.reset_sliders).grid(row=3*self.sliders_per_grid+3, column = 0, sticky = W+E+N+S)
        Button(text='Reset All Indexes', command=self.reset_all).grid(row=3*self.sliders_per_grid+3, column = 1, sticky = W+E+N+S)
        Button(text='Reset Other Indexes', command=self.reset_other).grid(row=3*self.sliders_per_grid+3, column = 2, sticky = W+E+N+S)

        self.root.mainloop()

    def set_image(self, a = 0):
        first_z = [self.sliders[i].get() for i in range(self.num_sliders)]
        z_list = []
        z = self.z_init
        z[0:self.num_sliders] = first_z
        z_list.append(np.asarray(z))
        # the_face = clv.generate_face(self.GAN, num_faces = 1, z_list = z_list)
        the_face = self.gen_fn(z_list = z_list)
        plt.imsave('Loading.png', the_face[0])
        image = Image.open('Loading.png')
        image = image.resize((self.slider_width, self.slider_width), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.face_label.configure(image=photo)
        self.face_label.image = photo
        os.remove('Loading.png')

    def random_all(self):
        self.z_init = np.random.normal(0, self.GAN["z_sd"], self.GAN["z_len"])
        for i in range(self.num_sliders):
            self.sliders[i].set(self.z_init[i]/self.GAN["z_sd"])
        self.set_image()

    def random_sliders(self):
        self.z_init[0:self.num_sliders] = np.random.normal(0, self.GAN["z_sd"], self.num_sliders)
        for i in range(self.num_sliders):
            self.sliders[i].set(self.z_init[i]/self.GAN["z_sd"])
        self.set_image()

    def random_other(self):
        self.z_init[self.num_sliders:self.GAN["z_len"]] = np.random.normal(0, self.GAN["z_sd"], self.GAN["z_len"] - self.num_sliders)
        self.set_image()

    def read_inserted_indexes(self):
        the_text = self.indexes_text.get("1.0",END)
        text_array = the_text.split(',')
        indexes = []
        for text in text_array:
            if ':' not in text:
                indexes.append(int(text))
            else:
                from_to = text.split(':')
                [indexes.append(int(from_to[0]) + i) for i in range(int(from_to[1]) - int(from_to[0]))]
        return indexes

    def reset_all(self):
        self.z_init = [0]*self.GAN["z_len"]
        for i in range(self.num_sliders):
            self.sliders[i].set(0)
        self.set_image()

    def reset_sliders(self):
        self.z_init[0:self.num_sliders] = [0]*self.num_sliders
        for i in range(self.num_sliders):
            self.sliders[i].set(0)
        self.set_image()

    def reset_other(self):
        self.z_init[self.num_sliders:self.GAN["z_len"]] = [0]*(self.GAN["z_len"] - self.num_sliders)
        self.set_image()

    def random_indexes(self):
        indexes = self.read_inserted_indexes()
        for i in indexes:
            this_rand = float(np.random.normal(0, self.GAN["z_sd"], 1))
            self.z_init[i]= this_rand
            if i < self.num_sliders:
                self.sliders[i].set(this_rand)

        self.set_image()

    def reset_indexes(self):
        indexes = self.read_inserted_indexes()
        for i in indexes:
            self.z_init[i] = 0
            if i < self.num_sliders:
                self.sliders[i].set(0)


    def reset(self):
        for i in range(self.num_sliders):
            self.sliders[i].set(0)
        self.set_image()

