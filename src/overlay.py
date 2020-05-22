from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


def write_text_on_pic(im, text = 'Hello', text_size = 50, text_pos = (25,25), text_color = 'red', 
                      background_color = 'black', background_dim = [(10,10), (500,120)]):
    """ 
    To write text on a picture. the image im must be an numpy array of shape (length, width, 4) or (length, width, 3)
    
    """
    copy = np.copy(im)
    if im.shape[-1] == 4:           
        im_PIL = Image.fromarray(np.uint8(copy[:,:,0:3]))    # 1) Convert to PIL.Image.Image type
    elif im.shape[-1] ==3:
        im_PIL = Image.fromarray(np.uint8(copy[:,:,0:3]))   # 1) Convert to PIL.Image.Image type
    else:
        print("Unknown type for the input image. Please Check it's either a RGB or RGBA numpy array.")
        
    d = ImageDraw.Draw(im_PIL)                          # 2) Create an object pointing on our picture we want to modify 
    d.rectangle(background_dim, fill=background_color, outline=None)     # 3) Draw a rectangle, which we can use as background
    try:
        fnt = ImageFont.truetype('arial.ttf', text_size)                     # 4) Create the font for the text
    except Exception as e:
        fnt = ImageFont.load_default()                     # 4) Create the font for the text

    d.text(text_pos, text, font = fnt, fill=text_color) # 5) Write the text on our picture

    #img.save('pil_text_font.png')
    #plt.imshow(im_PIL)
    
    return np.array(im_PIL) # 6) Convert back the PIL.Image.Image into a numpy array.



def draw_traj_on_pic(im, listPos, color = 'red', width = 2, rad = 4) :   
    revListPos = []
    for p in listPos :
        revListPos.append((p[1], p[0]))

    copy = np.copy(im)
    if im.shape[-1] == 4:           
        im_PIL = Image.fromarray(np.uint8(copy[:,:,0:3]))    # 1) Convert to PIL.Image.Image type
    elif im.shape[-1] ==3:
        im_PIL = Image.fromarray(np.uint8(copy[:,:,0:3]))   # 1) Convert to PIL.Image.Image type
    else:
        print("Unknown type for the input image. Please Check it's either a RGB or RGBA numpy array.")
        
    d = ImageDraw.Draw(im_PIL) 
    d.line(revListPos, fill=color, width=2)
    for p in revListPos :
        bbox = tuple(map(tuple, [np.subtract(p, (rad,rad)), np.subtract(p, (-rad, -rad))]))
        d.ellipse(bbox, outline=color)

    return np.array(im_PIL)

if __name__ == '__main__':
    pic = np.zeros((200,200,3))
    listPos = [
        (100,100),
        (150,122),
        (56,98)
        ]
    plt.imshow(draw_traj_on_pic(pic, listPos))
    plt.show()