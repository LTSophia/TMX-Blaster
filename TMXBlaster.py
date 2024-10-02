import cv2
import numpy as np
import PIL
from PIL import Image
import os
import argparse

import TMX

def Image2TMX(image_file : str, out_file : str, palette_size : int, HWM = None, VWM = None, user_id = 0, palette_override = TMX.TMXFile.PSMTC32, color_override = None):
    file_name = os.path.splitext(os.path.basename(out_file))[0]
    with Image.open(image_file) as image:
        width, height = image.size
    
        powof2 = 2 ** np.arange(16)
        if width not in powof2 or height not in powof2:
            orig_width = width
            orig_height = height
            closest = (np.abs(powof2 - width)).argmin()
            n_width = powof2[closest]
            closest = (np.abs(powof2 - height)).argmin()
            n_height = powof2[closest]
            print(f'Resizing image from {orig_width}x{orig_height} to {n_width}x{n_height}...')
            image = image.resize((n_width, n_height), resample=Image.Resampling.LANCZOS)
            
        width, height = image.size
    
        palette = None
        out_npimg = None
        if image.mode.endswith(('a', 'A')) and not (np.array(image.convert("RGBA"), dtype=np.uint8)[:,:,-1] == 255).all():
            image = image.convert("RGBA")
        
            np_image = np.array(image, dtype=np.uint8)
            alpha = np_image[:, :, 3]
            np_image = np_image[:, :, :3]
        
            print("Inpainting...")
            max_alpha = np.max(alpha)
            inpaint_mask = cv2.bitwise_not(np.maximum(255,  alpha + (255 - max_alpha) + (max_alpha//3)))
            np_image = cv2.inpaint(np_image, inpaint_mask, 4, cv2.INPAINT_TELEA)
        

            if palette_size > 0:
                for y in range(height):
                      for x in range(width):
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
        tmx = TMX.TMXFile().from_image(out_npimg, user_id, file_name, HWM, VWM, 0, 0, palette=palette, palette_mode=palette_override, color_mode=color_override)
            
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
    parser.add_argument('-hwm', '--horizontalwrapmode',
                        type=str,
                        choices=['repeat', 'clamp'],
                        default=None,
                        help='The wrap mode used for horizontal in-engine effects. (does not affect exported image)')
    parser.add_argument('-vwm', '--verticalwrapmode',
                        type=str,
                        choices=['repeat', 'clamp'],
                        default=None,
                        help='The wrap mode used for vertical in-engine effects. (does not affect exported image)')
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
        
        Image2TMX(args.input, output, args.palette, horizontal_wrap_mode, vertical_wrap_mode, user_id=user_id, palette_override=palette_override, color_override=color_override)