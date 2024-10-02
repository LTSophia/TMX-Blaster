import numpy as np

class TMXFile:
    file_tag = b'TMX0'
    
    class PSMTC32:
        code = 0x00 #Format code
        is_palette_mode = False  #This image mode is NOT for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 32
        def __init__(self, red = 0, green = 0, blue = 0, alpha = 255):
            self.red   = np.uint8(red)
            self.green = np.uint8(green)
            self.blue  = np.uint8(blue)
            self.alpha = np.uint8(round((alpha / 255) * 128))
        def from_bytes(self, pixbytes):
            self.red   = np.uint8(pixbytes[0])
            self.green = np.uint8(pixbytes[1])
            self.blue  = np.uint8(pixbytes[2])
            self.alpha = np.uint8(pixbytes[3])
            return self
        def get_red(self):
            return self.red
        def get_green(self):
            return self.green
        def get_blue(self):
            return self.blue
        def get_alpha(self):
            return np.uint8(round((np.uint8(self.alpha) / 128) * 255))
        def to_np_array(self, np_array, i, palette=None):
            np_array[i] = [self.get_red(), self.get_green(), self.get_blue(), self.get_alpha()]
            return i+1, np_array
        def __len__(self):
            return 4  #Pixel stored as 4 bytes
        def __array__(self):
            return np.array([self.red, self.green, self.blue, self.alpha])
    class PSMTC24:
        code = 0x01 #Format code
        is_palette_mode = False  #This image mode is NOT for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 24
        def __init__(self, red = 0, green = 0, blue = 0):
            self.red   = np.uint8(red)
            self.green = np.uint8(green)
            self.blue  = np.uint8(blue)
        def from_bytes(self, pixbytes):
            self.red   = np.uint8(pixbytes[0])
            self.green = np.uint8(pixbytes[1])
            self.blue  = np.uint8(pixbytes[2])
            return self
        def get_red(self):
            return self.red
        def get_green(self):
            return self.green
        def get_blue(self):
            return self.blue
        def get_alpha(self):
            return 255
        def to_np_array(self, np_array, i, palette=None):
            np_array[i] = [self.get_red(), self.get_green(), self.get_blue(), self.get_alpha()]
            return i+1, np_array
        def __len__(self):
            return 3  #Pixel stored as 3 bytes
        def __array__(self):
            return np.array([self.red, self.green, self.blue])
    class PSMTC16:
        code = 0x02 #Format code
        is_palette_mode = False  #This image mode is NOT for palettized color
        dtype = np.dtype(np.uint16)
        bit_len = 16
        def __init__(self, red = 0, green = 0, blue = 0, alpha = 255):
            if alpha > 10:
                self.red_bits   = np.uint16(round((red / 255) * 31))
                self.green_bits = np.uint16(round((green / 255) * 31))
                self.blue_bits  = np.uint16(round((blue / 255) * 31))
                if alpha < 245:
                    self.alpha_bit = np.uint16(1)  # Semi-Transparent colors have a value of 1
                else: 
                    self.alpha_bit = np.uint16(0)  # Opaque colors have a value of 0
                if self.red_bits == self.blue_bits == self.green_bits == 0 and alpha >= 245:
                    self.red_bits = self.green_bits = self.blue_bits = np.uint16(1)
                    self.alpha_bit = np.uint16(0)
            else:  # Fully transparent colors have all values as 0
                self.red_bits = self.green_bits = self.blue_bits = self.alpha_bit = np.uint16(0)
        def from_bytes(self, pixbytes):
            raw_val = np.uint16(int.from_bytes(pixbytes, 'little'))
            self.red_bits   = (raw_val & 0x001F)
            self.green_bits = (raw_val & 0x03E0) >> 5
            self.blue_bits  = (raw_val & 0x7C00) >> 10
            self.alpha_bit  = (raw_val & 0x8000) >> 15
            return self
        def get_red(self):
            return np.uint8(round((self.red_bits / 31) * 255))
        def get_green(self):
            return np.uint8(round((self.green_bits / 31) * 255))
        def get_blue(self):
            return np.uint8(round((self.blue_bits / 31) * 255))
        def get_alpha(self):
            is_not_all_zero = min(1, (self.red_bits | self.green_bits | self.blue_bits | self.alpha_bit))
            return np.uint8((255 * is_not_all_zero) // (1 + self.alpha_bit))
        def to_np_array(self, np_array, i, palette=None):
            np_array[i] = [self.get_red(), self.get_green(), self.get_blue(), self.get_alpha()]
            return i+1, np_array
        def __len__(self):
            return 2  #Pixel stored as 2 bytes
        def __array__(self):
            return np.array([np.uint16((self.alpha_bit << 15) | (self.blue_bits << 10) | (self.green_bits << 5) | (self.red_bits))])
    class PSMTC16S:
        code = 0x0A #Format code
        is_palette_mode = False  #This image mode is NOT for palettized color
        dtype = np.dtype(np.uint16)
        bit_len = 16
        def __init__(self, red = 0, green = 0, blue = 0, alpha = 255):
            if alpha > 10:
                self.red_bits   = np.uint16(round((red / 255) * 31))
                self.green_bits = np.uint16(round((green / 255) * 31))
                self.blue_bits  = np.uint16(round((blue / 255) * 31))
                if alpha < 245:
                    self.alpha_bit = np.uint16(1)  # Semi-Transparent colors have a value of 1
                else: 
                    self.alpha_bit = np.uint16(0)  # Opaque colors have a value of 0
                if self.red_bits == self.blue_bits == self.green_bits == 0 and alpha >= 245:
                    self.red_bits = self.green_bits = self.blue_bits = np.uint16(1)
                    self.alpha_bit = np.uint16(0)
            else:  # Fully transparent colors have all values as 0
                self.red_bits = self.green_bits = self.blue_bits = self.alpha_bit = np.uint16(0)
        def from_bytes(self, pixbytes):
            raw_val = np.uint16(int.from_bytes(pixbytes, 'little'))
            self.red_bits   = (raw_val & 0x001F)
            self.green_bits = (raw_val & 0x03E0) >> 5
            self.blue_bits  = (raw_val & 0x7C00) >> 10
            self.alpha_bit  = (raw_val & 0x8000) >> 15
            return self
        def get_red(self):
            return np.uint8(self.red_bits << 3)
        def get_green(self):
            return np.uint8(self.green_bits << 3)
        def get_blue(self):
            return np.uint8(self.blue_bits << 3)
        def get_alpha(self):
            is_not_all_zero = max(1, (self.red_bits | self.green_bits | self.blue_bits | self.alpha_bit))
            return np.uint8((255 * is_not_all_zero) // (1 + self.alpha_bit))
        def to_np_array(self, np_array, i, palette=None):
            np_array[i] = [self.get_red(), self.get_green(), self.get_blue(), self.get_alpha()]
            return i+1, np_array
        def __len__(self):
            return 2  #Pixel stored as 2 bytes
        def __array__(self):
            return np.array([np.uint16((self.alpha_bit << 15) | (self.blue_bits << 10) | (self.green_bits << 5) | (self.red_bits))])
    class PSMT8:
        code = 0x13 #Format code
        is_palette_mode = True  #This image mode is for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 8
        def __init__(self, index = 0):
            self.index = np.uint8(index)
        def from_bytes(self, pixbytes):
            self.index   = np.uint8(pixbytes[0])
            return self
        def get_index(self):
            return self.index
        def to_np_array(self, np_array, i, palette):
            return palette[self.get_index()].to_np_array(np_array, i)
        def __len__(self):
            return 1  #Pixel stored as 1 byte
        def __array__(self):
            return np.array([np.uint8(self.index)])
    class PSMT4:
        code = 0x14 #Format code
        is_palette_mode = True  #This image mode is for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 4
        def __init__(self, index1 = 0, index2 = 0):
            assert index1 < 16 and index2 < 16, "Pallete indexes must be less than 16."
            self.index1 = np.uint8(index1)
            self.index2 = np.uint8(index2)
        def from_bytes(self, pixbytes):
            raw_val = np.uint8(pixbytes[0])
            self.index1 = (raw_val & 0x0F)
            self.index2 = (raw_val & 0xF0) >> 4
            return self
        def get_index1(self):
            return self.index1
        def get_index2(self):
            return self.index2
        def to_np_array(self, np_array, i, palette):
            i, np_array = palette[self.get_index1()].to_np_array(np_array, i)
            return palette[self.get_index2()].to_np_array(np_array, i)
        def __len__(self):
            return 1  #Pixel stored as 4 bits, Container holds 2
        def __array__(self):
            return np.array([np.uint8(self.index1 | self.index2 << 4)])
    class PSMT8H:
        code = 0x1B #Format code
        is_palette_mode = True  #This image mode is for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 8
        def __init__(self, index = 0):
            self.index = np.uint8(index)
        def from_bytes(self, pixbytes):
            self.index   = np.uint8(pixbytes[0])
            return self
        def get_index(self):
            return self.index
        def to_np_array(self, np_array, i, palette):
            return palette[self.get_index()].to_np_array(np_array, i)
        def __len__(self):
            return 1  #Pixel stored as 1 byte
        def __array__(self):
            return np.array([np.uint8(self.index)])
    class PSMT4HL:
        code = 0x24 #Format code
        is_palette_mode = True  #This image mode is for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 4
        def __init__(self, index1 = 0, index2 = 0):
            assert index1 < 16 and index2 < 16, "Pallete indexes must be less than 16."
            self.index1 = np.uint8(index1)
            self.index2 = np.uint8(index2)
        def from_bytes(self, pixbytes):
            raw_val = np.uint8(pixbytes[0])
            self.index1 = (raw_val & 0x0F)
            self.index2 = (raw_val & 0xF0) >> 4
            return self
        def get_index1(self):
            return self.index1
        def get_index2(self):
            return self.index2
        def to_np_array(self, np_array, i, palette):
            i, np_array = palette[self.get_index1()].to_np_array(np_array, i)
            return palette[self.get_index2()].to_np_array(np_array, i)
        def __len__(self):
            return 1  #Pixel stored as 4 bits, Container holds 2
        def __array__(self):
            return np.array([np.uint8(self.index1 | self.index2 << 4)])
    class PSMT4HH:
        code = 0x2C #Format code
        is_palette_mode = True  #This image mode is for palettized color
        dtype = np.dtype(np.uint8)
        bit_len = 4
        def __init__(self, index1 = 0, index2 = 0):
            assert index1 < 16 and index2 < 16, "Pallete indexes must be less than 16."
            self.index1 = np.uint8(index1)
            self.index2 = np.uint8(index2)
        def from_bytes(self, pixbytes):
            raw_val = np.uint8(pixbytes[0])
            self.index1 = (raw_val & 0x0F)
            self.index2 = (raw_val & 0xF0) >> 4
            return self
        def get_index1(self):
            return self.index1
        def get_index2(self):
            return self.index2
        def to_np_array(self, np_array, i, palette):
            i, np_array = palette[self.get_index1()].to_np_array(np_array, i)
            return palette[self.get_index2()].to_np_array(np_array, i)
        def __len__(self):
            return 1  #Pixel stored as 4 bits, Container holds 2
        def __array__(self):
            return np.array([np.uint8(self.index1 | self.index2 << 4)])
        
    def __init__(self):
        pass
    def from_image(self, img, user_id : int, comment : str, HorizontalWrapMode, VerticalWrapMode, texture_id, clut_id, palette = None, palette_mode = PSMTC32, color_mode = None):
        self.palette_mode = palette_mode
        self.palette = []
        self.user_id = user_id
        self.texture_id = texture_id
        self.clut_id = clut_id
        self.comment = comment
        self.HorizontalWrapMode = HorizontalWrapMode
        self.VerticalWrapMode = VerticalWrapMode
        if palette is not None:
            palette_size, color_size = palette.shape
            for i in range(palette_size):
                if color_size == 4:
                    self.palette.append(self.palette_mode(red=palette[i,0], green=palette[i,1], blue=palette[i,2], alpha=palette[i,3]))
                elif color_size == 3:
                    self.palette.append(self.palette_mode(red=palette[i,0], green=palette[i,1], blue=palette[i,2]))
            if palette_size == 256:
                i = 0
                while i < 256:
                    temp = self.palette[i+8:i+16]
                    self.palette[i+8:i+16] = self.palette[i+16:i+24]
                    self.palette[i+16:i+24] = temp
                    i += 32
        self.palette_size = len(self.palette_mode()) * len(self.palette)
        if len(img.shape) > 2:
            self.img_h, self.img_w, img_colors = img.shape
        else:
            self.img_h, self.img_w = img.shape
            img_colors = 1
        if color_mode is not None:
            if len(self.palette) > 0 and not color_mode.is_palette_mode:
                raise ValueError("Cannot create palettized image with non-palette color mode!")
            else:
                self.color_mode = color_mode
        else:
            if img_colors == 4:
                self.color_mode = TMXFile.PSMTC32
            elif img_colors == 3:
                self.color_mode =  TMXFile.PSMTC24
            elif img_colors == 1:
                assert len(self.palette) > 0
                if len(self.palette) == 16:
                    self.color_mode =  TMXFile.PSMT4
                elif len(self.palette) == 256:
                    self.color_mode =  TMXFile.PSMT8
        self.pixel_data = []
        for y in range(self.img_h):
            x = 0
            while x < self.img_w:
                if img_colors == 4:
                    self.pixel_data.append(self.color_mode(red=img[y, x, 0], green=img[y, x, 1], blue=img[y, x, 2], alpha=img[y, x, 3]))
                elif img_colors == 3:
                    self.pixel_data.append(self.color_mode(red=img[y, x, 0], green=img[y, x, 1], blue=img[y, x, 2]))
                elif img_colors == 1 and (self.color_mode is  TMXFile.PSMT8 or self.color_mode is  TMXFile.PSMT8H):
                    self.pixel_data.append(self.color_mode(img[y, x]))
                elif img_colors == 1 and (self.color_mode is  TMXFile.PSMT4 or self.color_mode is  TMXFile.PSMT4HH or self.color_mode is  TMXFile.PSMT4HL):
                    self.pixel_data.append(self.color_mode(img[y, x], img[y, x + 1]))
                    x += 1
                x += 1
        self.pixdata_size = len(self.color_mode()) * len(self.pixel_data)
        return self
    def from_tmx(self, file_path):
        with open(file_path, "rb") as tmxfile:
            tmxfile.seek(8, 0)
            if self.file_tag != tmxfile.peek(4)[:4]:
                raise RuntimeError(f'Not a valid TMX File:{file_path}')
            tmxfile.seek(2, 0)
            self.userid = np.uint16(int.from_bytes(tmxfile.read(2), 'little'))
            tmxfile.seek(16, 0)
            is_palette = bool(np.uint8(int.from_bytes(tmxfile.read(1), 'little')))
            mode_byte = int.from_bytes(tmxfile.read(1))
            if is_palette:
                if mode_byte == TMXFile.PSMTC32.code:
                    self.palette_mode = TMXFile.PSMTC32
                elif mode_byte == TMXFile.PSMTC24.code:
                    self.palette_mode = TMXFile.PSMTC24
                elif mode_byte == TMXFile.PSMTC16.code:
                    self.palette_mode = TMXFile.PSMTC16
                elif mode_byte == TMXFile.PSMTC16S.code:
                    self.palette_mode = TMXFile.PSMTC16S
            else:
                self.palette_mode = TMXFile.PSMTC32
            self.img_w = int.from_bytes(tmxfile.read(2), 'little')
            self.img_h = int.from_bytes(tmxfile.read(2), 'little')
            mode_byte = int.from_bytes(tmxfile.read(1))
            if mode_byte == TMXFile.PSMTC32.code:
                self.color_mode = TMXFile.PSMTC32
                self.palette_size = 0
            elif mode_byte == TMXFile.PSMTC24.code:
                self.color_mode = TMXFile.PSMTC24
                self.palette_size = 0
            elif mode_byte == TMXFile.PSMTC16.code:
                self.color_mode = TMXFile.PSMTC16
                self.palette_size = 0
            elif mode_byte == TMXFile.PSMTC16S.code:
                self.color_mode = TMXFile.PSMTC16S
                self.palette_size = 0
            elif mode_byte == TMXFile.PSMT8.code:
                self.color_mode = TMXFile.PSMT8
                self.palette_size = 256 * len(self.palette_mode())
            elif mode_byte == TMXFile.PSMT4.code:
                self.color_mode = TMXFile.PSMT4
                self.palette_size = 16 * len(self.palette_mode())
            elif mode_byte == TMXFile.PSMT8H.code:
                self.color_mode = TMXFile.PSMT8H
                self.palette_size = 256 * len(self.palette_mode())
                palette_size = 256
            elif mode_byte == TMXFile.PSMT4HH.code:
                self.color_mode = TMXFile.PSMT4HH
            elif mode_byte == TMXFile.PSMT4HL.code:
                self.color_mode = TMXFile.PSMT4HL
                self.palette_size = 16 * len(self.palette_mode())
            self.pixdata_size = ((int(self.img_h) * int(self.img_w)) * self.color_mode().bit_len) // 8
            tmxfile.seek(4, 1)
            wrapmodebyte = np.uint8(int.from_bytes(tmxfile.read(1)))
            if wrapmodebyte == 0xFF:
                self.HorizontalWrapMode = self.VerticalWrapMode = None
            elif wrapmodebyte == 0x00:
                self.HorizontalWrapMode = self.VerticalWrapMode = False
            elif wrapmodebyte == 0x01:
                self.HorizontalWrapMode = True
                self.VerticalWrapMode   = False
            elif wrapmodebyte == 0x04:
                self.HorizontalWrapMode = False
                self.VerticalWrapMode   = True
            elif wrapmodebyte == 0x05:
                self.HorizontalWrapMode = self.VerticalWrapMode = True
            self.texture_id = np.uint32(int.from_bytes(tmxfile.read(4), 'little'))
            self.clut_id = np.uint32(int.from_bytes(tmxfile.read(4), 'little'))
            self.comment = str(tmxfile.read(28), 'shift_jis')
            #buffer_np = np.frombuffer(tmxfile.read(self.palette_size), dtype=self.palette_mode, count=-1)
            pal_bytes = tmxfile.read(self.palette_size)[:self.palette_size]
            self.palette = []
            i = 0
            b_s = len(self.palette_mode())
            while i + b_s <= self.palette_size:
                self.palette.append(self.palette_mode().from_bytes(pal_bytes[i:i+b_s]))
                i += b_s
            if len(self.palette) == 256:
                i = 0
                while i < 256:
                    temp = self.palette[i+8:i+16]
                    self.palette[i+8:i+16] = self.palette[i+16:i+24]
                    self.palette[i+16:i+24] = temp
                    i += 32
            
            pix_bytes = tmxfile.read(self.pixdata_size)[:self.pixdata_size]
            self.pixel_data = []
            i = 0
            b_s = len(self.color_mode())
            while i + b_s < self.pixdata_size:
                self.pixel_data.append(self.color_mode().from_bytes(pix_bytes[i:i+b_s]))
                i += b_s
            
        return self

    def to_np_array(self):
        np_array = np.zeros((self.img_h*self.img_w, 4), dtype = np.uint8)
        i = 0
        for pix in self.pixel_data:
            i, np_array = pix.to_np_array(np_array, i, self.palette)
        np_array = np_array.reshape((self.img_h, self.img_w, 4))
        return np_array
    def __len__(self):
        data_size = 16 + 48 + self.palette_size + self.pixdata_size
        padding = data_size % 16
        return data_size + padding
    def save(self, file_path):
        with open(file_path, "wb") as tmxfile:
            # File Header
            tmxfile.write(np.uint16(2).tobytes())
            tmxfile.write(np.uint16(self.user_id).tobytes())
            tmxfile.write(np.uint32(len(self)).tobytes())
            tmxfile.write(self.file_tag)
            tmxfile.write(np.uint32(0).tobytes()) # Padding
            # Picture Header
            tmxfile.write(np.uint8(len(self.palette) > 0).tobytes())
            tmxfile.write(np.uint8(self.palette_mode.code).tobytes())
            tmxfile.write(np.uint16(self.img_w).tobytes())
            tmxfile.write(np.uint16(self.img_h).tobytes())
            tmxfile.write(np.uint8(self.color_mode.code).tobytes())
            tmxfile.write(np.uint8(0).tobytes()) # MipmapCount (Unsupported)
            tmxfile.write(np.uint8(0).tobytes()) # MipK (Unsupported)
            tmxfile.write(np.uint8(0).tobytes()) # MipL (Unsupported)
            tmxfile.write(np.uint8(0).tobytes()) # Padding
            if self.HorizontalWrapMode is None or self.VerticalWrapMode is None:
                tmxfile.write(np.uint8(0xFF).tobytes())
            else:
                tmxfile.write(np.uint8(self.HorizontalWrapMode + (self.VerticalWrapMode * 0x4)).tobytes())
            tmxfile.write(np.uint32(self.texture_id).tobytes())
            tmxfile.write(np.uint32(self.clut_id).tobytes())
            if len(self.comment) <= 28:
                tmxfile.write(self.comment.encode("shift_jis", "replace") + bytes((28 - len(self.comment))))
            else:
                tmxfile.write(self.comment[:28].encode("shift_jis", "replace"))
            # Palette Data
            tmxfile.write(np.array(self.palette).tobytes())
            # Image Data
            tmxfile.write(np.array(self.pixel_data).tobytes())
            # End Padding
            padding = tmxfile.tell() % 16
            if padding > 0:
                tmxfile.write(bytes(padding))