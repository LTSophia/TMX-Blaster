import cv2
import numpy as np
import PIL
from PIL import Image
import scipy
import os
import argparse

import TMX

FORMAT_STRINGS = {
    '32' : TMX.TMXFile.PSMTC32,
    '24' : TMX.TMXFile.PSMTC24,
    '16' : TMX.TMXFile.PSMTC16,
    '16S': TMX.TMXFile.PSMTC16S,
    '8'  : TMX.TMXFile.PSMT8,
    '8H' : TMX.TMXFile.PSMT8H,
    '4'  : TMX.TMXFile.PSMT4,
    '4HH': TMX.TMXFile.PSMT4HH,
    '4HL': TMX.TMXFile.PSMT4HL,
    }

ACCEPTED_SIZES = 2 ** np.arange(16)

DONT_RESIZE_THRESH = 0.2  # Dont resize the image, just crop or expand if the difference is 4% of the width/height (guesswork value)

def _get_resized_with_bg(image : Image, re_width : int, re_height : int, bg_width : int, bg_height : int, palette_size = 0):
    if palette_size > 0: 
        reduce_color = palette_size
    else:
        reduce_color = 256
    
    o_width = image.size[0]
    o_height = image.size[1]
    if (o_width - re_width) < re_width * DONT_RESIZE_THRESH or abs(o_height - re_height) < re_height * DONT_RESIZE_THRESH:
        r_width = image.size[0]
        r_height = o_height
    else:
        r_width = re_width
        r_height = re_height
        print(f'Resizing image from {o_width}x{o_height} to {r_width}x{r_height}...')
        image = image.resize((re_width, re_height))
    
    if image.mode.endswith(('a', 'A')):
        color_mode =  'RGBA'
        filler = (0, 0, 0, 0)
    else:
        color_mode = 'RGB'
        filler = _get_mode_color(image, reduce_color)
    bg_image = Image.new(color_mode, (bg_width, bg_width), filler)
    
    print(f'Cropping image to {bg_width}x{bg_width}...')
    location = (0 - max(0, (r_width - bg_width) // 2), 0 - max(0, (r_height - bg_height) // 2)) # centers image if container is too small
    bg_image.paste(image, location)
    return bg_image

def _get_mode_color(image : Image, palette_size : int):
    temp_palettized = image.quantize(palette_size, Image.Quantize.MEDIANCUT)
    return tuple(np.array(temp_palettized.getpalette()).reshape((palette_size, -1))[scipy.stats.mode(np.array(temp_palettized).reshape((-1, 1)))[0][0]])
    
def _image_rescale(image_file : str, width : int, height : int, palette_size = 0):
    image = None
    
    with Image.open(image_file) as in_image:
        if in_image.mode.upper().endswith('A'):
            image = in_image.convert('RGBA')
        else:
            image = in_image.convert('RGB')
        
    assert image is not None, 'Error loading image.'
    w, h = image.size
        
    if width != w or height != h:
        if width is not None:
            w = width
        if height is not None:
            h = height
        
        if width is not None and height is not None:
            image = _get_resized_with_bg(image, width, height, width, height, palette_size)
        else:
            if w not in ACCEPTED_SIZES or h not in ACCEPTED_SIZES:
                closest = (np.abs(ACCEPTED_SIZES - w)).argmin()
                n_width = ACCEPTED_SIZES[closest]
                closest = (np.abs(ACCEPTED_SIZES - h)).argmin()
                n_height = ACCEPTED_SIZES[closest]
                
                if round(n_width/n_height, 3) == round(w/h, 3):
                    image = _get_resized_with_bg(image, n_width, n_height, n_width, n_height, palette_size)
                elif abs(n_width - w) < abs(n_height - h):
                    nn_height = round((h/w) * n_width)
                    image = _get_resized_with_bg(image, n_width, nn_height, n_width, n_height, palette_size)
                else:
                    nn_width = round((w/h) * n_height)
                    image = _get_resized_with_bg(image, nn_width, n_height, n_width, n_height, palette_size)
    out_image = np.array(image, dtype=np.uint8)
    
    return out_image
def _solidify(image : np.ndarray):
    alpha = image[:, :, 3]

    print("Solidifying...")
    max_alpha = np.max(alpha)
    inpaint_mask = cv2.bitwise_not(np.maximum(255,  alpha + (255 - max_alpha) + (max_alpha//2)))
    image[:,:,:3] = cv2.inpaint(image[:, :, :3], inpaint_mask, 4, cv2.INPAINT_TELEA)
    
    return image
    
def _quantize(image : np.ndarray, palette_size : int, is_16_color = False):
    if image.shape[2] == 4: # Set all Fully transparent colors to consistent black
        alpha = np.dstack((image[:, :, 3], image[:, :, 3], image[:, :, 3], image[:, :, 3]))
        image = np.where(alpha > 10, image, np.zeros_like(image))
    
    if is_16_color:
        image[:, :, :3] = np.round(( np.round((image[:, :, :3] / 255) * 31, decimals=0).astype(dtype=np.uint8) / 31) * 255).astype(dtype=np.uint8)
        if image.shape[2] == 4:
            image[:, :, 3]  = (np.round((image[:, :, 3] / 255) * 2, decimals=0).astype(dtype=np.uint8) / 2 * 255).astype(dtype=np.uint8)

    print("Quantizing...")
    out_image = Image.fromarray(image)
    out_image = out_image.quantize(palette_size, kmeans=1, dither=Image.Dither.FLOYDSTEINBERG )
    
    palette = np.array(out_image.getpalette(None), dtype=np.uint8).reshape((palette_size, -1))
    if palette.shape[-1] == 4:
        palette[:, 3] = np.where(palette[:, 3] <= 245, palette[:, 3], 255)
        palette[:, 3] = np.where(palette[:, 3] >= 10, palette[:, 3], 0)
            
    return np.array(out_image, dtype=np.uint8), palette
    
def _image_process(image_file : str, palette_size : int, width=None, height=None, solidify=True, is_16_color = False):
    image = _image_rescale(image_file, width, height, palette_size)
    
    if solidify and image.shape[2] == 4:
        image = _solidify(image)
                
    if palette_size > 0:
        image, palette = _quantize(image, palette_size, is_16_color)
    else:
        palette = None
       
    return image, palette

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='TMX Blaster',
                    description='The only TMX tool you should ever need.')
    parser.add_argument('input',
                        metavar='input filename',
                        type=str,
                        help='Can be a PNG, JPEG, etc. or TGA for encoding, ' + 
                             'or TMX for decoding.')
    parser.add_argument('--height',
                        type=int,
                        choices=ACCEPTED_SIZES,
                        default=None,
                        help='Sets the value to resize the image\'s height to.')
    parser.add_argument('--width',
                        type=int,
                        choices=ACCEPTED_SIZES,
                        default=None,
                        help='Sets the value to resize the image\'s width to.')
    parser.add_argument('-p', '--palette',
                        type=int,
                        choices=[16, 256],
                        default=0,
                        help='The palette size used when encoding to TMX.')
    parser.add_argument('-ui', '--userid',
                        type=int,
                        default=0,
                        help='Set the User ID of the TMX.')
    parser.add_argument('-b', '--bustup',
                        action='store_true',
                        help='Marks the TMX as being a bustup, the same as \'-ui 1\'.')
    parser.add_argument('-uc', '--usercomment',
                        type=str,
                        default=None,
                        help='Set the User Comment for the TMX. Defaults to the name of the file.')
    
    parser.add_argument('--nosolidify',
                        action='store_false',
                        help='Sets the program to NOT solidify the image before making it a TMX. (ADVANCED OPTION: Not recommended)')
    parser.add_argument('-ti', '--textureid',
                        type=int,
                        default=0,
                        help='Set the Texture ID of the TMX. (ADVANCED OPTION: Usually unnecessary)')
    parser.add_argument('-ci', '--clutid',
                        type=int,
                        default=0,
                        help='Set the CLUT ID of the TMX. (ADVANCED OPTION: Unknown Functionality)')
    parser.add_argument('--horizontalwrapmode',
                        type=str,
                        choices=['repeat', 'clamp'],
                        default=None,
                        help='The wrap mode used for horizontal in-engine effects. (ADVANCED OPTION: does not affect exported image)')
    parser.add_argument('--verticalwrapmode',
                        type=str,
                        choices=['repeat', 'clamp'],
                        default=None,
                        help='The wrap mode used for vertical in-engine effects. (ADVANCED OPTION: does not affect exported image)')
    parser.add_argument('--palettetype',
                        type=str,
                        choices=['32', '16', '16S'],
                        default='32',
                        help='The color mode used to store the palette information (ADVANCED OPTION)')
    parser.add_argument('--pixeltype',
                        type=str,
                        choices=['32', '24', '16', '16S', '8', '8H', '4', '4HH', '4HL'],
                        default=None,
                        help='The color mode used to store the pixel information (ADVANCED OPTION)')
    parser.add_argument('output',
                        metavar='output filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='TMX file or image file to output. ' +
                             'Defaults to TMX for encoding and TGA for decoding.')
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    
    in_split = os.path.splitext(args.input)
    
    if args.output is None:
        output = in_split[0]
        if in_split[1].upper() == '.TMX':
            output += '.tga'
        else:
            output += '.tmx'
    else:
        output = args.output
            
    if args.userid == 0 and args.bustup:
        user_id = args.bustup
    else:
        user_id = args.userid
            
    if args.usercomment is not None:
        user_comment = args.usercomment
    else:
        user_comment = os.path.splitext(os.path.basename(output))[0]
    
    if args.horizontalwrapmode is not None:
        horizontal_wrap_mode = (args.horizontalwrapmode.lower() == 'clamp' or args.horizontalwrapmode.lower() == 'c')
    else:
        horizontal_wrap_mode = None

    if args.verticalwrapmode is not None:
        vertical_wrap_mode = (args.verticalwrapmode.lower() == 'clamp' or args.verticalwrapmode.lower() == 'c')
    else:
        vertical_wrap_mode = None
    
    palette_size = args.palette
    if (args.pixeltype is not None and args.pixeltype.startswith('16')) or args.palettetype.startswith('16'):
        is_16_color = True
    else:
        is_16_color = False
    
    palette_override = FORMAT_STRINGS[args.palettetype.upper()]
    if args.pixeltype is not None:
        color_override = FORMAT_STRINGS[args.pixeltype.upper()]
        if args.pixeltype.startswith('8'):
            palette_size = 256
        elif args.pixeltype.startswith('4'):
            palette_size = 16
    else:
        color_override = None
        
    tmx = TMX.TMXFile(user_id, user_comment, horizontal_wrap_mode, vertical_wrap_mode, args.textureid, args.clutid, palette_override, color_override)
    out_split = os.path.splitext(output)
    if in_split[1].upper() == ".TMX":
        image = TMX.TMXFile.from_tmx(args.input)
        
        if image.shape[2] == 4 and image[:,:,3].min() == 255:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        if palette_size > 0:
            image, palette = _quantize(image, args.palette, is_16_color)
        else:
            palette = None
        
    else:
        image, palette = _image_process(args.input, palette_size, args.width, args.height, args.nosolidify, is_16_color)
        

    # Output Handling
    if os.path.splitext(output)[1].upper() == '.TMX':
        tmx = tmx.from_image(image, palette)
        tmx.to_tmx(output)
    else:
        if palette is not None:
            image = palette[image]
        Image.fromarray(image).save(output)