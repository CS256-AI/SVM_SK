from PIL import Image, ImageDraw
import shutil
import random
import os
import math
import sys
import numpy as np


class Data:

    def __init__(self):
        self.padding = 1  # padding to be used to draw the image
        self.color = 0  # color for drawing the image. All images ddrawn in balck
        self.brush_stroke_max = 2  # Max brush stroke
        self.pos_shift_max = 3  # Max shift in position of image in shift transformation
        self.rotation_max = 20  # Max positive/ negative rotation in degrees orientation transformation
        self.resize_ratio_min = 0.1  # (1- resize_ratio_min) gives lower bound on resized images size
        self.ellip_num_max = 3  # Max no. of stray sllipsoids to be drawn in image
        self.ellip_size_max = 3  # Max width, height of ellipsoid in pixels
        self.size = 25  # size of canvas

    def _draw_o(self, width):
        """ Draw zener card with circle symbol
        :param size: Width and length of the circle's bounding box
        :return: None
        """
        # print('width O --> {}'.format(width))
        image = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(image)
        # Draw the circle with padding pixel gap from either end
        draw.ellipse((self.padding,
                      self.padding,
                      self.size-(self.padding + 1),
                      self.size-(self.padding + 1)), outline=self.color)
        del draw
        print('self.size -->{}'.format(self.size))
        print('In O --> {}'.format(image.size))
        return image
        # image.show()

    def _draw_p(self, width):
        """ Draw zener card with plus symbol
        :param size_x: Width of the line's bounding box
        :param size_y: Height of the line's bounding box
        :return: None
        """
        # print('width P --> {}'.format(width))
        image = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(image)
        # Draw the vertical line
        draw.line([((self.size-1)//2, self.padding), ((self.size-1)//2, self.size-(self.padding + 1))],
                  fill=self.color,
                  width=width)
        # Draw the horizontal line
        draw.line([(self.padding, (self.size-1)//2), (self.size - (self.padding + 1), (self.size-1)//2)],
                  fill=self.color,
                  width=width)
        del draw
        print('self.size -->{}'.format(self.size))
        print('In P --> {}'.format(image.size))
        return image
        # image.show()

    def _draw_q(self, width):
        """ Draw zener card with square symbol
        :param size: size of the square
        :return: None
        """
        # print('width Q --> {}'.format(width))
        image = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(image)
        #  Drawing the rectangle with lines to incorporate thickness of brush(width)
        #  x0 refers to the left most and bottom most vertex of the rectangle
        x0 = (self.padding, self.size - (self.padding + 1))
        x1 = (self.padding, self.padding)
        x2 = ((self.size-1) - x1[0], x1[1])
        x3 = ((self.size-1) - x0[0], x0[1])

        draw.line([x0, x1], fill=self.color, width=width)
        draw.line([x1, x2], fill=self.color, width=width)
        draw.line([x2, x3], fill=self.color, width=width)
        draw.line([x3, x0], fill=self.color, width=width)
        del draw
        print('self.size -->{}'.format(self.size))
        print('In q --> {}'.format(image.size))
        return image
        # image.show()

    def _draw_s(self, width):
        print('width S --> {}'.format(width))
        image = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(image)
        # x0 is the center point in lower half and the numbering continues clockwise
        x0 = (self.size/2, 3 * self.size/4 - self.padding/3)
        x1 = (self.size/8 + self.padding, self.size - (self.padding + 1))
        x2 = (self.size/4 + self.padding, 0.6 * self.size)
        x3 = (self.padding, self.size/3 + self.padding)
        x4 = (self.size/3, self.size/3 + self.padding)
        x5 = (self.size/2, self.padding)
        x6 = (self.size-1 - x4[0], x4[1])
        x7 = (self.size-1 - x3[0], x3[1])
        x8 = (self.size - 1 - x2[0], x2[1])
        x9 = (self.size-1 - x1[0], x1[1])

        draw.line([x0, x1], fill=self.color, width=width)
        draw.line([x1, x2], fill=self.color, width=width)
        draw.line([x2, x3], fill=self.color, width=width)
        draw.line([x3, x4], fill=self.color, width=width)
        draw.line([x4, x5], fill=self.color, width=width)
        draw.line([x5, x6], fill=self.color, width=width)
        draw.line([x6, x7], fill=self.color, width=width)
        draw.line([x7, x8], fill=self.color, width=width)
        draw.line([x8, x9], fill=self.color, width=width)
        draw.line([x9, x0], fill=self.color, width=width)
        # image.show()
        del draw
        print('self.size -->{}'.format(self.size))
        print('In S --> {}'.format(image.size))
        return image

    def _draw_w(self, width):
        # print('width W --> {}'.format(width))
        image = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(image)
        size = self.size - 2 * self.padding
        points1 = []
        points2 = []
        points3 = []
        for x in range(360):
            # point = (x, math.sin(math.radians(x)))
            # print("x --> {}".format(x))
            amp = size*0.5
            point1 = (self.padding +
                      (x/360.0) * size, self.padding +
                      (1.0 - math.sin(math.radians(x))) * (amp/2.0) + (amp*0.1))
            point2 = (self.padding +
                      (x / 360.0) * size, self.padding +
                      (1.0 - math.sin(math.radians(x))) * (amp/2.0) + (amp*0.5))
            point3 = (self.padding +
                      (x / 360.0) * size, self.padding +
                      (1.0 - math.sin(math.radians(x))) * (amp / 2.0) + (amp * 0.9))
            # print(point)
            points1.append(point1)
            points2.append(point2)
            points3.append(point3)
        draw.line(points1, fill=self.color, width=width)
        draw.line(points2,  fill=self.color, width=width)
        draw.line(points3,  fill=self.color, width=width)
        del draw
        image = image.rotate(-90)
        print('self.size -->{}'.format(self.size))
        print('In W --> {}'.format(image.size))
        return image
        # image.show()

    def _trans_resize(self, in_image):
        width, height = in_image.size
        final_image = Image.new('L', (self.size, self.size), 255)
        resize_ratio = 1 - random.random() * self.resize_ratio_min
        print('resize factor --> {}'.format(resize_ratio))
        print((width, height))
        print((int(width * resize_ratio), int(height * resize_ratio)))
        # Maintaining aspect ratio by multiplying height and widht by the same factor
        in_image = in_image.resize((int(width * resize_ratio), int(height * resize_ratio)))

        # print(numpy.array(in_image))
        # print(in_image)
        # print(in_image.size)
        # in_image.show()
        # Calculate the position where the image is to be pasted so that it is in the center
        diff = (width - in_image.size[0])/2
        final_image.paste(in_image, (diff, diff))
        # print(final_image.size)
        # print('trans_resize img mode --> {}'.format(final_image.mode))
        return final_image

    def _trans_pos(self, in_image):
        width, height = in_image.size
        final_image = Image.new('L', (width, height), 255)
        pos_shift = random.randint(1, self.pos_shift_max)
        # print('pos_shift --> {}'.format(pos_shift))
        final_image.paste(in_image, (pos_shift, pos_shift))
        # print('_trans_pos img mode --> {}'.format(final_image.mode))
        return final_image

    def _trans_orientation(self, in_image):
        # print('_trans_orientation input img mode --> {}'.format(in_image.mode))
        rotation_degree = random.randint(-self.rotation_max, self.rotation_max)
        # print('rotation_degree --> {}'.format(rotation_degree))
        # converted to have an alpha layer
        im2 = in_image.convert('RGBA')
        # im2.show()
        rot = im2.rotate(rotation_degree)
        # rot.show()
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        final_image = Image.composite(rot, fff, rot)
        final_image = final_image.convert(in_image.mode)
        # out_image.show()
        # print('_trans_orientation img mode --> {}'.format(final_image.mode))
        return final_image

    def _trans_stray(self, in_image):
        ellip_num = random.randint(1, self.ellip_num_max)
        # print('ellip_num --> {}'.format(ellip_num))
        width, height = in_image.size
        draw = ImageDraw.Draw(in_image)
        for i in range(ellip_num):
            ellip_size_x = random.randint(1, self.ellip_size_max)
            ellip_size_y = random.randint(1, self.ellip_size_max)
            # print('ellip_size_x --> {}'.format(ellip_size_x))
            # print('ellip_size_y --> {}'.format(ellip_size_y))
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            draw.ellipse((x, y, x + ellip_size_x, y + ellip_size_y), fill=self.color, outline=self.color)
        # print('_trans_stray img mode --> {}'.format(in_image.mode))
        return in_image

    def gen_images(self, folder_name, num_examples):
        symbols = ['o', 'p', 'w', 'q', 's']
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, True)  # Delete the directory if it is already there
        os.mkdir(folder_name)  # Create directory with the folder name
        os.chdir(folder_name)  # Change the directory so training images are store there
        for i in xrange(num_examples):
            symbol_sel = random.choice(symbols)  # Uniformly randomly select an image form the list
            func = getattr(Data, "_draw_" + symbol_sel)  # Get the function object to draw appropriate symbol
            # print("func --> {}".format(func))
            # The variation for thickness of brush is ensured by random value of width
            width = random.randint(1, self.brush_stroke_max)
            image = func(self, width)
            print('Image size --> {}'.format(image.size))
            image = self._trans_resize(image)
            image = self._trans_pos(image)
            image = self._trans_orientation(image)
            image = self._trans_stray(image)
            image.save(str(i + 1)+'_'+symbol_sel.upper()+".png")

    def get_data(self, folder_name, symbol_name):
        file_names = []
        y_list = []  # whether file contains symbol or nor
        x_list = []  # input matrix containing image data
        os.chdir(folder_name)
        symbol_name = symbol_name.upper()
        for file_name in os.listdir('.'):
            if not (file_name.endswith('.png')):
                continue
            image = Image.open(file_name)
            image_array = np.array(image).flatten()
            if symbol_name in file_name:
                y_list.append(1)
            else:
                y_list.append(-1)
            file_names.append(file_name)
            x_list.append(image_array)
        x = np.array(x_list)
        x = x/255
        y = np.array(y_list)
        return (file_names,x,y)

    def scale_data(self, input_data):
        file_names, x, y = input_data
        x_pos = x[y > 0]
        x_neg = x[y < 0]
        m_pos = np.mean(x_pos, axis=0)
        m_neg = np.mean(x_neg, axis=0)
        np.set_printoptions(threshold=np.inf)
        r = np.linalg.norm(m_pos - m_neg)
        rdiff_pos = x_pos - m_pos
        r_pos = max(np.linalg.norm(rdiff_pos, axis=1))
        rdiff_neg = x_neg - m_neg
        r_neg = max(np.linalg.norm(rdiff_neg, axis=1))
        lam = (1/2.0)*(r/(r_pos + r_neg))
        # print('lam --> {}'.format(lam))
        x_pos_prime = lam * x_pos
        x_neg_prime = lam * x_neg
        return(lam, x_pos_prime, x_neg_prime)

# if len(sys.argv) < 2:
#     print("Insufficient number of arguments.\nPattern : python zener_generator.py data 10000")
#     sys.exit()
# else:
#     fol_name, num_examples = sys.argv[1:3]
#     print(fol_name, num_examples)
#     data = Data()
#     data.gen_images(fol_name, int(num_examples))
