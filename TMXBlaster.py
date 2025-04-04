import cv2
import numpy as np
from PIL import Image
import os
import sys
import argparse
import colour
from numba import jit

import TMX

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

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

FIFTEEN_BIT_COLORS = np.load(os.path.join(application_path, '15bitcolor.npy'))

#OkLab format + alpha: Lightness, green-to-red, blue-to-yellow, Alpha
COLOR_WEIGHT = np.array([1, 1, 1, 1], dtype=np.float32)

@jit(nopython=True)
def floyd_steinberg(image, palette):
    # image: np.array of shape (height, width, 4), dtype=float
    # works in-place!
    h, w = image.shape[:2]
    index_image = np.zeros((h, w), dtype=np.uint8)
    pal_vals = palette * COLOR_WEIGHT
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            col_vals = old * COLOR_WEIGHT
            col_vals[0] = max(0.0, min(1.0, col_vals[0]))
            col_vals[1] = max(-0.233922, min(0.276272, col_vals[1]))
            col_vals[2] = max(-0.311621, min(0.198490, col_vals[2]))
            index = np.sum((pal_vals - col_vals) ** 2, axis=1).argmin()
            new = palette[index]
            index_image[y, x] = index
            error = old - new
            # precomputing the constants helps
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375 # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625 # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125 # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h): 
                image[y + 1, x - 1] += error * 0.1875 # left, down, 3 / 16
    return index_image
    
def _image_rescale(image_file : str, width : int, height : int):
    image = None
    
    with Image.open(image_file) as in_image:
        # convert to RGBA to make image manipulation easier
        image = in_image.convert('RGBA')
        
    assert image is not None, 'Error loading image.'

    og_w, og_h = image.size
    w, h = image.size

    # if the original image is the desired width and height or the original image is an acceptable size and a custom width and height isn't given,
    #  then the image is already fine,
    if not((width == og_w and height == og_h) or (width is None and height is None and og_w in ACCEPTED_SIZES and og_h in ACCEPTED_SIZES)):
        if width is not None and height is not None:
            w = width
            h = height
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        else:
            if width is not None:
                aspect_ratio = float(og_h/og_w)
                w = width
                h = round(aspect_ratio * width)
            elif height is not None:
                aspect_ratio = float(og_w/og_h)
                h = height
                w = round(aspect_ratio * height)
            
            if og_w != w or og_h != h:
                image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
            
            closest = np.searchsorted(ACCEPTED_SIZES,[w,],side='right')[0]
            w = ACCEPTED_SIZES[closest]
            closest = np.searchsorted(ACCEPTED_SIZES,[h,],side='right')[0]
            h = ACCEPTED_SIZES[closest]

    out_image = np.array(image, dtype=np.uint8)
    
    return out_image, h, w

## Solidify image with OpenCV
def _solidify(image : np.ndarray, quiet = False):
    rgb, alpha = seperate_rgb_alpha(image)

    if not quiet:
        print("Solidifying...")
    max_alpha = np.max(alpha)
    inpaint_mask = cv2.bitwise_not(np.maximum(255,  alpha + (255 - max_alpha) + (max_alpha//2)))
    image[:,:,:3] = cv2.inpaint(rgb, inpaint_mask, 4, cv2.INPAINT_TELEA)
    
    return image

def _pad_image(image, h, w, palette = None, alpha = True):
    cur_h, cur_w = image.shape[:2]
    if h != cur_h or w != cur_w:
        if palette is not None:
            if alpha:
                background_index = palette[:,3].argmin()
            else:
                values, counts = np.unique(image.flatten(), return_counts=True)
                background_index = values[counts.argmax()]
            canvas = np.zeros((h, w), dtype=np.uint8)
            img_h, img_w = image.shape
            canvas[:,:] = background_index
            canvas[:img_h, :img_w] = image
            image = canvas
        else:
            if not alpha:
                background_color = np.array([255, 255, 255, 255], dtype=np.uint8)
            canvas = np.zeros((h, w, 4), dtype=np.uint8)
            img_h, img_w = image.shape[:2]
            if not alpha:
                canvas[:,:] = background_color
            canvas[:img_h, :img_w] = image
            image = canvas
    return image
    
def seperate_rgb_alpha(image : np.ndarray):
    original_bounds = image.shape[:-1]
    shaped_image = image.reshape((-1, 4))
    rgb = shaped_image[:, :3].reshape(tuple(original_bounds) + (3,))
    alpha = shaped_image[:, 3].reshape(tuple(original_bounds) + (1,))
    return rgb, alpha

def combine_rgb_alpha(rgb : np.ndarray, alpha : np.ndarray):
    rgb_bounds = rgb.shape[:-1]
    alpha = alpha.reshape(tuple(rgb_bounds) + (1,))
    return np.concatenate((rgb, alpha), axis=-1)

def srgba_to_work_image(srgba : np.ndarray):
    # set up image for conversion
    srgba = srgba.astype(np.float32) / 255
    srgb, alpha = seperate_rgb_alpha(srgba)

    # convert to Oklab colorspace
    xyz = colour.sRGB_to_XYZ(srgb)
    oklab = colour.XYZ_to_Oklab(xyz)
    # combine Oklab colors with alpha and reshape
    return combine_rgb_alpha(oklab, alpha)

def work_colors_to_srgba(work_colors : np.ndarray):
    work_colors, alpha = seperate_rgb_alpha(work_colors)

    work_image = work_colors.reshape(-1, 1, 3)
    xyz = colour.Oklab_to_XYZ(work_image)
    srgb = colour.XYZ_to_sRGB(xyz)

    srgb = np.round(srgb * 255, decimals=0).astype(dtype=np.uint8)
    alpha = np.round(alpha * 255, decimals=0).astype(dtype=np.uint8)
    
    return combine_rgb_alpha(srgb, alpha)

## Quantize Image
def _quantize(image : np.ndarray, palette_size : int, is_16_color = False, dither = False, quiet = False):
    # NOT NEEDED set all fully transparent colors to consistent black (helps greatly with preserving mre colors in quantized image)
    #_alpha_stack = np.dstack((image[:, :, 3], image[:, :, 3], image[:, :, 3], image[:, :, 3]))
    #image = np.where(_alpha_stack > 10, image, np.zeros_like(image))

    if not quiet:
        print("Quantizing...")

    h, w = image.shape[:2]

    work_image = srgba_to_work_image(image)
    work_colors = work_image.reshape(-1, 4)

    if is_16_color:
        # set the colors to quantize into to a precalculated list of valid 16 bit colors
        labels = FIFTEEN_BIT_COLORS
    else:
        labels = None

    # define criteria and quantize with kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _pal_img, palette = cv2.kmeans(np.float32(work_colors), palette_size, labels, criteria, 10, cv2.KMEANS_PP_CENTERS)

    if dither:
        indexes = floyd_steinberg(work_image, palette)
    else:
        indexes = _pal_img.reshape(h, w, 1)

    # convert palette to sRGB colorspace
    pal_srgba = work_colors_to_srgba(palette)

    return indexes, pal_srgba

def premultiply_alpha(image : np.ndarray):
    image = image.astype(np.float32) / 255
    rgb, alpha = seperate_rgb_alpha(image)
    rgb *= alpha
    alpha *= alpha
    return (combine_rgb_alpha(rgb, alpha) * 255).astype(np.uint8)

## Main function controlling image preparation and editing
def _image_process(image_file : str, palette_size : int, width=None, height=None, solidify=True, is_16_color = False, quiet = False):
    image, h, w = _image_rescale(image_file, width, height)
    
    alpha = False
    if image[:,:,3].min() < 255:
        alpha = True
        if solidify:
            image = _solidify(image, quiet=quiet)
        image = premultiply_alpha(image)
    
    if palette_size > 0:
        dither = min(image.shape[0], image.shape[1]) > 128
        image, palette = _quantize(image, palette_size, is_16_color, dither=dither, quiet=quiet)
    else:
        if is_16_color:
            if not quiet:
                print('Converting to 16-bit color, this can take a while...')
            image = floyd_steinberg(image, FIFTEEN_BIT_COLORS)
        palette = None

    image = _pad_image(image, h, w, palette, alpha)

    return image, palette

def main(input_file : str, output_file : str, width : int, height : int, palette_size : int, user_id : int, user_comment : str, horizontal_wrap_mode, vertical_wrap_mode, texture_id : int, clut_id : int, palette_override, color_override, is_16_color : bool, no_solidify : bool, quiet = False):
    if not quiet:
        print(f'Converting {os.path.basename(input_file)} to {os.path.basename(output_file)}...')
    in_split = os.path.splitext(input_file)
    out_split = os.path.splitext(output_file)

    if user_comment is None:
        comment = os.path.basename(out_split[0])
    else:
        comment = user_comment

    tmx = TMX.TMXFile(user_id, comment, horizontal_wrap_mode, vertical_wrap_mode, texture_id, clut_id, palette_override, color_override)
    
    if in_split[1].upper() == ".TMX":
        image = TMX.TMXFile.from_tmx(input_file)
        
        if image.shape[2] < 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            image[:,:,3] = 255
            
        if palette_size > 0:
            image, palette = _quantize(image, palette_size, is_16_color, quiet=quiet)
        else:
            palette = None
    else:
        image, palette = _image_process(input_file, palette_size, width, height, no_solidify, is_16_color, quiet=quiet)
        
    # Output Handling
    if out_split[1].upper() == '.TMX':
        tmx = tmx.from_image(image, palette)
        tmx.to_tmx(output_file)
    else:
        if palette is not None:
            image = palette[image]
        Image.fromarray(image).save(output_file)
    if not quiet:
        print(f'Conversion complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='TMX Blaster',
                    description='The only TMX tool you should ever need.')
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='Disable any command-line output.')
    parser.add_argument('--batch',
                        action='store_true',
                        help='Enable batch folder processing.')
    parser.add_argument('-r', '--recursive',
                        action='store_true',
                        help='Process all subfolders. [BATCH PROCESSING]')
    parser.add_argument('-flat', '--flattenoutput',
                        action='store_true',
                        help='When recursive, do not create extra folders to match the original file structure. [BATCH PROCESSING]')
    parser.add_argument('-if', '--infiletype',
                        type=str,
                        default='TMX',
                        help='Filetype to process in the folder. [BATCH PROCESSING]')
    parser.add_argument('-of', '--outfiletype',
                        type=str,
                        default=None,
                        help='Filetype to export too. [BATCH PROCESSING]')
    parser.add_argument('input',
                        type=str,
                        help='Can be a PNG, JPEG, etc. or TGA for encoding, ' + 
                             'or TMX for decoding. \n' + 
                             'The folder to process in Batch Processing mode.')
    parser.add_argument('output',
                        type=str,
                        nargs='?',
                        default=None,
                        help='TMX file or image file to output. ' +
                             'Defaults to TMX for encoding and TGA for decoding. \n' + 
                             'The folder to export to in Batch Processing mode.')
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
    args = parser.parse_args()
    
    
    if args.userid == 0 and args.bustup:
        user_id = args.bustup
    else:
        user_id = args.userid
            
    if args.usercomment is not None:
        user_comment = args.usercomment
    else:
        user_comment = None
    
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
        
    width = args.width
    height = args.height
    
    texture_id = args.textureid
    clut_id = args.clutid
    no_solidify = args.nosolidify
    
    quiet = args.quiet

    is_batch = args.batch or os.path.isdir(args.input)

    if not is_batch:
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
            
        main(args.input, output, width, height, palette_size, user_id, user_comment, horizontal_wrap_mode, vertical_wrap_mode, texture_id, clut_id, palette_override, color_override, is_16_color, no_solidify, quiet=quiet)
    else:
        if not os.path.isdir(args.input):
            raise FileNotFoundError(args.input)
        in_file_type = '.' + str(args.infiletype).strip().strip('.')
        
        if args.outfiletype is None:
            if in_file_type == '.TMX':
                out_file_type = '.TGA'
            else:
                out_file_type = '.TMX'
        else:
            out_file_type = '.' + str(args.outfiletype).strip().strip('.')
            
        if args.output is None:
            output = args.input
        else:
            output = args.output
            if os.path.isfile(output):
                output = os.path.dirname(output)
            elif not os.path.isdir(output):
                os.makedirs(output, exist_ok=True)
        
        if not quiet:
            print(f'Converting {in_file_type} files in {os.path.basename(args.input)} to {out_file_type} files in {os.path.basename(output)}...')
        
        if not args.recursive:
            files = [file_name for file_name in os.listdir(args.input) if file_name.upper().endswith(in_file_type)]
            all_num = len(files)
            num_chars = len(str(all_num))
            i = 1
            last_str_len = 0
            for file_name in files:
                if not quiet:
                    progress_str = f'<{str(i).zfill(num_chars)}|{all_num}> [{file_name}] -> [{os.path.splitext(file_name)[0] + out_file_type}]'
                    padding = ' ' * max(0, last_str_len - len(progress_str))
                    print(f'{progress_str}{padding}', end='\r')
                    last_str_len = len(progress_str)
                    i += 1
                in_file = os.path.join(args.input, file_name)
                out_file = os.path.join(output, os.path.splitext(file_name)[0] + out_file_type)
                main(in_file, out_file, width, height, palette_size, user_id, user_comment, horizontal_wrap_mode, vertical_wrap_mode, texture_id, clut_id, palette_override, color_override, is_16_color, no_solidify, quiet=True)
        else:
            last_str_len = 0
            input_folder_name = os.path.basename(args.input)
            output_folder_name = os.path.basename(output)
            
            out_file_path = output
            for root, dirs, files in os.walk(args.input):
                if not args.flattenoutput:
                    out_file_path = os.path.join(output, os.path.relpath(root, args.input))
                    if not os.path.isdir(out_file_path):
                        os.makedirs(out_file_path, exist_ok=True)
                for file_name in files:
                    if file_name.upper().endswith(in_file_type):
                        in_file = os.path.join(root, file_name)
                        out_file = os.path.join(out_file_path, os.path.splitext(file_name)[0] + out_file_type)
                        if not quiet:
                            in_print = str(os.path.join(input_folder_name, os.path.relpath(in_file, args.input))).strip('\\/')
                            out_print = str(os.path.join(output_folder_name, os.path.relpath(out_file, output))).strip('\\/')
                            progress_str = f'[{in_print}] -> [{out_print}]'
                            padding = ' ' * max(0, last_str_len - len(progress_str))
                            print(f'{progress_str}{padding}', end='\r')
                            last_str_len = len(progress_str)
                        main(in_file, out_file, width, height, palette_size, user_id, user_comment, horizontal_wrap_mode, vertical_wrap_mode, texture_id, clut_id, palette_override, color_override, is_16_color, no_solidify, quiet=True)
        if not quiet:
            print('\n Complete!')