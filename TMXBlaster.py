import cv2
import numpy as np
import PIL.Image
from PIL import Image
import scipy.stats
import os
import argparse

import TMX

def _get_mode_color(image : Image, palette_size : int):
    temp_palettized = image.quantize(palette_size, Image.Quantize.MEDIANCUT)
    return tuple(np.array(temp_palettized.getpalette()).reshape((palette_size, -1))[scipy.stats.mode(np.array(temp_palettized).reshape((-1, 1)))[0][0]])
    
def Image2TMX(image_file : str, out_file : str, palette_size : int, width=None, height=None, HWM = None, VWM = None, user_id = 0, clut_id=0, texture_id=0, palette_override = TMX.TMXFile.PSMTC32, color_override = None, solidify=True):
    file_name = os.path.splitext(os.path.basename(out_file))[0]
    with Image.open(image_file) as image:
        w, h = image.size
        
        if not (width == w and height == h):
            if width is not None:
                w = width
            if height is not None:
                h = height
            
            o_width, o_height = image.size
        
            if width is not None and height is not None:
                print(f'Resizing image from {o_width}x{o_height} to {width}x{height}...')
                image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
            else:
                powof2 = 2 ** np.arange(16)
                if w not in powof2 or h not in powof2:
                    closest = (np.abs(powof2 - w)).argmin()
                    n_width = powof2[closest]
                    closest = (np.abs(powof2 - h)).argmin()
                    n_height = powof2[closest]
                    if round(n_width/n_height, 3) == round(w/h, 3):
                        print(f'Resizing image from {o_width}x{o_height} to {n_width}x{n_height}...')
                        image = image.resize((n_width, n_height), resample=Image.Resampling.LANCZOS)
                    elif n_width - w < n_height - h:
                        nn_height = round((h/w) * n_width)
                        print(f'Resizing image from {o_width}x{o_height} to {n_width}x{nn_height}...')
                        image = image.resize((n_width, nn_height))
                        if image.mode.endswith(('a', 'A')):
                            n_image = Image.new('RGBA', (n_width, n_height), (0, 0, 0, 0))
                        elif palette_size > 0: 
                            mode_color = _get_mode_color(image, palette_size)
                            n_image = Image.new('RGB', (n_width, n_height), mode_color)
                        else:
                            mode_color = _get_mode_color(image, 256)
                            n_image = Image.new('RGB', (n_width, n_height), mode_color)
                        print(f'Expanding image to {n_width}x{n_height}...')
                        n_image.paste(image, (0, 0))
                        image = n_image
                    else:
                        nn_width = round((w/h) * n_height)
                        print(f'Resizing image from {o_width}x{o_height} to {nn_width}x{n_height}...')
                        image = image.resize((nn_width, n_height))
                        if image.mode.endswith(('a', 'A')):
                            n_image = Image.new('RGBA', (n_width, n_height), (0, 0, 0, 0))
                        elif palette_size > 0: 
                            mode_color = _get_mode_color(image, palette_size)
                            n_image = Image.new('RGB', (n_width, n_height), mode_color)
                        else:
                            mode_color = _get_mode_color(image, 256)
                            n_image = Image.new('RGB', (n_width, n_height), mode_color)
                        print(f'Expanding image to {n_width}x{n_height}...')
                        n_image.paste(image, (0, 0))
                        image = n_image
            
            w, h = image.size
    
        palette = None
        out_npimg = None
        if image.mode.endswith(('a', 'A')) and not (np.array(image.convert("RGBA"), dtype=np.uint8)[:,:,-1] == 255).all():
            image = image.convert("RGBA")
        
            np_image = np.array(image, dtype=np.uint8)
            alpha = np_image[:, :, 3]
            np_image = np_image[:, :, :3]
            
            if solidify:
                print("Solidifying...")
                max_alpha = np.max(alpha)
                inpaint_mask = cv2.bitwise_not(np.maximum(255,  alpha + (255 - max_alpha) + (max_alpha//2)))
                np_image = cv2.inpaint(np_image, inpaint_mask, 4, cv2.INPAINT_TELEA)
        

            if palette_size > 0:
                for y in range(h):
                      for x in range(w):
                          if alpha[y, x] < 10:
                              alpha[y, x] = 0
                              np_image[y, x] = [0, 0, 0]
                np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2RGBA)
                np_image[:, :, 3] = alpha
            
                #Image.fromarray(np_image).save("imageout.tga", "tga")
            
                print("Quantizing...")
                image = Image.fromarray(np_image)
                image = image.quantize(palette_size, Image.Quantize.FASTOCTREE, kmeans=1, dither=Image.Dither.FLOYDSTEINBERG)
            
                palette = np.array(image.getpalette(None), dtype=np.uint8).reshape((palette_size, -1))
                for i in range(palette_size):
                    if palette[i, 3] >= 245:
                        palette[i, 3] = 255
                    elif palette[i, 3] <= 10:
                        palette[i, 3] = 0
                out_npimg = np.array(image, dtype=np.uint8)
            else:
                out_npimg = cv2.cvtColor(np_image, cv2.COLOR_RGB2RGBA)
                out_npimg[:, :, 3] = alpha
        else:
            image = image.convert("RGB")
            if palette_size > 0:
                print("Quantizing...")
                image = image.quantize(palette_size, Image.Quantize.MEDIANCUT, kmeans=1, dither=Image.Dither.FLOYDSTEINBERG )
            
                palette = np.array(image.getpalette(None), dtype=np.uint8).reshape((palette_size, -1))
            out_npimg = np.array(image, dtype=np.uint8)
        print("Creating TMX data...")
        tmx = TMX.TMXFile().from_image(out_npimg, user_id, file_name, HWM, VWM, texture_id=texture_id, clut_id=clut_id, palette=palette, palette_mode=palette_override, color_mode=color_override)
            
        print("Saving TMX...")
        tmx.save(out_file)
    
# image_file = "image.png"
# palette_size = 256
# palette_override = TMX.TMXFile.PSMTC16S
# color_override = None
# is_bustup = False
# HorizontalWrapMode = None # None or bool: True = Clamp, False = Repeat
# VerticallWrapMode  = None # None or bool: True = Clamp, False = Repeat

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
                        choices=2 ** np.arange(16, dtype=int),
                        default=None,
                        help='Sets the value to resize the image\'s height to.')
    parser.add_argument('--width',
                        type=int,
                        choices=2 ** np.arange(16, dtype=int),
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
    parser.add_argument('-hwm', '--horizontalwrapmode',
                        type=str,
                        choices=['repeat', 'clamp'],
                        default=None,
                        help='The wrap mode used for horizontal in-engine effects. (ADVANCED OPTION: does not affect exported image)')
    parser.add_argument('-vwm', '--verticalwrapmode',
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
    if os.path.splitext(args.input)[1].lower() == ".tmx":
        output = args.output
        if args.output is None:
            output = os.path.splitext(args.input)[0] + '.tga'
        print('Loading TMX...')
        tmx = TMX.TMXFile().from_tmx(args.input)
        print('Translating image data...')
        np_array = tmx.to_np_array()
        if np_array[:,:,3].min() == 255:
            np_array = cv2.cvtColor(np_array, cv2.COLOR_RGBA2RGB)
        print('Saving...')
        Image.fromarray(np_array).save(output)
    else:
        output = args.output
        if args.output is None:
            output = os.path.splitext(args.input)[0] + '.tmx'
            
        if args.userid == 0 and args.bustup:
            user_id = args.bustup
        else:
            user_id = args.userid
            
        horizontal_wrap_mode = None
        if args.horizontalwrapmode is not None:
            horizontal_wrap_mode = (args.horizontalwrapmode.lower() == 'clamp' or args.horizontalwrapmode.lower() == 'c')
        vertical_wrap_mode = None
        if args.verticalwrapmode is not None:
            vertical_wrap_mode = (args.verticalwrapmode.lower() == 'clamp' or args.verticalwrapmode.lower() == 'c')
         

        palette_override = TMX.TMXFile.PSMTC32
        if args.palettetype.upper() == '16':
            palette_override = TMX.TMXFile.PSMTC16
        elif args.palettetype.upper() == '16S':
            palette_override = TMX.TMXFile.PSMTC16S
        
        color_override = None
        if args.pixeltype is not None:
            if args.pixeltype.upper() == '32':
                color_override = TMX.TMXFile.PSMTC32
            elif args.pixeltype.upper() == '24':
                color_override = TMX.TMXFile.PSMTC24
            elif args.pixeltype.upper() == '16':
                color_override = TMX.TMXFile.PSMTC16
            elif args.pixeltype.upper() == '16S':
                color_override = TMX.TMXFile.PSMTC16S
            elif args.pixeltype.upper() == '8':
                color_override = TMX.TMXFile.PSMT8
            elif args.pixeltype.upper() == '8H':
                color_override = TMX.TMXFile.PSMT8H
            elif args.pixeltype.upper() == '4':
                color_override = TMX.TMXFile.PSMT4
            elif args.pixeltype.upper() == '4HH':
                color_override = TMX.TMXFile.PSMT4HH
            elif args.pixeltype.upper() == '4HL':
                color_override = TMX.TMXFile.PSMT4HL
        
        Image2TMX(args.input, output, args.palette, args.width, args.height, horizontal_wrap_mode, vertical_wrap_mode, user_id=user_id, clut_id=args.clutid, texture_id=args.textureid, palette_override=palette_override, color_override=color_override, solidify=args.nosolidify)