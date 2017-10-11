from PIL import Image, ImageDraw
import shutil
import random
import os

class Data():
    def draw_o(self, size):
        """ Draw zener card with circle symbol
        :param size: Width and length of the circle's bounding box
        :return: None
        """
        image = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((1, 1, size-2, size-2), outline=0)  # Draw the circle with 1 pixel gap from either end
        del draw
        return image
        #image.show()


    def draw_p(self, size):
        """ Draw zener card with plus symbol
        :param size_x: Width of the line's bounding box
        :param size_y: Height of the line's bounding box
        :return: None
        """
        image = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(image)
        draw.line([((size-1)//2, 1), ((size-1)//2, size-2)], fill=0, width=1)  # Draw the certical line
        draw.line([(1, (size-1)//2), (size-2, (size-1)//2)], fill=0, width=1)  # Draw the certical line
        del draw
        return image
        #image.show()


    def draw_q(self, size):
        """ Draw zener card with square symbol
        :param size: Size of the square
        :return: None
        """
        image = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((1,1,size-2,size-2))
        del draw
        return image
        #image.show()

    def draw_s(self, size):
        image = Image.new('L', (25, 25), 255)
        draw = ImageDraw.Draw(image)
        draw.line([(0, 9), (9, 9)])
        draw.line([(9, 9), (12, 0)])
        draw.line([(12, 0), (15, 9)])
        draw.line([(15, 9), (25, 9)])
        draw.line([(25, 9), (16, 15)])
        draw.line([(16, 15), (19, 24)])
        draw.line([(19, 24), (12, 19)])
        draw.line([(12, 19), (4, 24)])
        draw.line([(4, 24), (7, 15)])
        draw.line([(7, 15), (0, 9)])
        return image


    def gen_data(self, folder_name, num_examples):
        symbols = ['o', 'p', 'q', 's']
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, True) # Delete the directory if it is already there
        os.mkdir(folder_name) # Create directory with the folder name
        os.chdir(folder_name) # Change the directory so training images are store there
        for i in xrange(1,num_examples+1):
            symbol_sel = random.choice(symbols) # Uniformly randomly select an image form the list
            func = getattr(Data, "draw_" + symbol_sel) # Get the function object to draw appropriate symbol
            #print("func --> {}".format(func))
            image = func(self, 25)
            image.save(str(i)+'_'+symbol_sel.upper()+".png")

data = Data()
data.gen_data('training',10)