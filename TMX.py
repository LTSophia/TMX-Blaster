import numpy as np

class TMXFile:
    class PSMTC32:
        code = 0x00  #Format code
        is_color = True #This format stores color
        dtype = np.dtype(np.uint8)
        def to_bytes(np_array : np.ndarray):
            np_array = np_array.reshape((-1, 4))
            np_array[:,3] = np.round((np_array[:,3] / 255) * 128, decimals=0).astype(dtype=np.uint8)
            return np_array.flatten().tobytes()
        def from_bytes(np_array : np.ndarray, palette = None):
            np_array = np_array.reshape((-1, 4))
            np_array[:,3] = np.round((np_array[:,3] / 128) * 255, decimals=0).astype(dtype=np.uint8)
            return np_array
        def get_length():
            return 32
    class PSMTC24:
        code = 0x01 #Format code
        is_color = True #This format stores color
        dtype = np.dtype(np.uint8)
        def to_bytes(np_array : np.ndarray):
            np_array = np_array.reshape((-1, 4))
            return np_array[:, :3].flatten().tobytes()
        def from_bytes(np_array : np.ndarray, palette = None):
            return np_array.reshape((-1, 3))
        def get_length():
            return 24
    class PSMTC16:
        code = 0x02 #Format code
        is_color = True #This format stores color
        dtype = np.dtype(np.uint16)
        def to_bytes(np_array : np.ndarray):
            np_array = np_array.reshape((-1, 4))
            np_array = np_array.astype(np.uint16)
            array_colors = np.round((np_array[:,:3] / 255) * 31, decimals=0).astype(dtype=np.uint16)
            new_array = (array_colors[:,2] << 10) | (array_colors[:,1] << 5) | array_colors[:,0]
            alpha = np_array[:,3]
            default = np.uint16(0x0421) #  Equivalent to '0000 0100 0010 0001', represents a color with a value of 1 for each color
            new_array = np.where((new_array == 0).astype(np.uint16) & (alpha > 245).astype(np.uint16), default, new_array)
            new_array = np.where((alpha < 245).astype(np.uint16) & (alpha[:] > 10).astype(np.uint16), new_array | 0x8000, new_array)
            new_array = np.where(alpha < 10, np.uint16(0x0000), new_array)  # Set all transparent colors to 0
            return new_array.tobytes()
        def from_bytes(np_array : np.ndarray, palette = None):
            np_array = np_array.reshape((-1, 1))
            new_array = np.zeros((np_array.shape[0], 4), dtype=np.uint8)
            new_array[:,0] = np.round(( (np_array & 0x001F)         / 31) * 255, decimals=0).astype(np.uint8)
            new_array[:,1] = np.round((((np_array & 0x03E0) >>  5)  / 31) * 255, decimals=0).astype(np.uint8)
            new_array[:,2] = np.round((((np_array & 0x7C00) >> 10)  / 31) * 255, decimals=0).astype(np.uint8)
            alpha = (((np_array & 0x8000) >> 15) * 127).astype(np.uint8)
            new_array[:,3] = np.where(alpha[:] == 0 & (new_array[:,0] | new_array[:,1] | new_array[:,0]), 255, alpha).astype(np.uint8)
            return new_array
        def get_length():
            return 16
    class PSMTC16S(PSMTC16):
        code = 0x0A #Format code
        is_color = True #This format stores color
        dtype = np.dtype(np.uint16)
        def to_bytes(np_array : np.ndarray):
            return TMXFile.PSMTC16.to_bytes(np_array)
        def from_bytes(np_array : np.ndarray, palette = None):
            return TMXFile.PSMTC16.from_bytes(np_array)
        def get_length():
            return TMXFile.PSMTC16.get_length()
    class PSMT8:
        code = 0x13 #Format code
        is_color = False #This format does NOT store color
        dtype = np.dtype(np.uint8)
        palette_size = 256
        def to_bytes(np_array : np.ndarray):
            return np_array.reshape((-1, 1)).tobytes()
        def from_bytes(np_array : np.ndarray, palette : np.ndarray):
            np_array = np_array.flatten()
            return palette[np_array]
        def get_length():
            return 8
    class PSMT4:
        code = 0x14 #Format code
        is_color = False #This format does NOT store color
        dtype = np.dtype(np.uint8)
        palette_size = 16
        def to_bytes(np_array : np.ndarray):
            np_array = np_array.reshape((-1, 2))
            new_array = np.zeros((np_array.shape[0], 1), dtype=np.uint8)
            new_array = np_array[:,0] | (np_array[:,1] << 4)
            return new_array.tobytes()
        def from_bytes(np_array : np.ndarray, palette  : np.ndarray):
            new_array = np.zeros((np_array.shape[0], 2), dtype=np.uint8)
            new_array[:,0] =  np_array & 0x0F
            new_array[:,1] = (np_array & 0xF0) >> 4
            new_array = new_array.flatten()
            return palette[new_array]
        def get_length():
            return 4
    class PSMT8H:
        code = 0x1B #Format code
        is_color = False #This format does NOT store color
        dtype = np.dtype(np.uint8)
        palette_size = 256
        def to_bytes(np_array : np.ndarray):
            return TMXFile.PSMT8.to_bytes(np_array)
        def from_bytes(np_array : np.ndarray, palette  : np.ndarray):
            return TMXFile.PSMT8.from_bytes(np_array, palette)
        def get_length():
            return TMXFile.PSMT8.get_length()
    class PSMT4HL:
        code = 0x24 #Format code
        is_color = False #This format does NOT store color
        dtype = np.dtype(np.uint8)
        palette_size = 16
        def to_bytes(np_array : np.ndarray):
            return TMXFile.PSMT4.to_bytes(np_array)
        def from_bytes(np_array : np.ndarray, palette  : np.ndarray):
            return TMXFile.PSMT4.from_bytes(np_array, palette)
        def get_length():
            return TMXFile.PSMT4.get_length()
    class PSMT4HH:
        code = 0x2C #Format code
        is_color = False #This format does NOT store color
        dtype = np.dtype(np.uint8)
        palette_size = 16
        def to_bytes(np_array : np.ndarray):
            return TMXFile.PSMT4.to_bytes(np_array)
        def from_bytes(np_array : np.ndarray, palette  : np.ndarray):
            return TMXFile.PSMT4.from_bytes(np_array, palette)
        def get_length():
            return TMXFile.PSMT4.get_length()
        
    file_tag = b'TMX0'
    format_codes = {
        PSMTC32.code : PSMTC32, 
        PSMTC24.code : PSMTC24, 
        PSMTC16.code : PSMTC16, PSMTC16S.code : PSMTC16S, 
        PSMT8.code : PSMT8, PSMT8H.code : PSMT8H,
        PSMT4.code : PSMT4, PSMT4HL.code : PSMT4HL, PSMT4HH.code : PSMT4HH
                    }
    def __init__(self, user_id : int, comment : str, HorizontalWrapMode = None, VerticalWrapMode = None, texture_id = 0, clut_id = 0, palette_mode = PSMTC32, color_mode = None):
        self.user_id            = user_id
        self.texture_id         = texture_id
        self.clut_id            = clut_id
        self.comment            = comment
        self.HorizontalWrapMode = HorizontalWrapMode
        self.VerticalWrapMode   = VerticalWrapMode
        self.palette_mode       = palette_mode
        self.color_mode         = color_mode
    
    def from_image(self, image : np.ndarray, palette = None):
        if palette is not None and palette.shape[-1] == 3:
            #palette = np.insert(palette, -1, 255, axis=1)
            palette = np.hstack((palette, np.full((palette.shape[0], 1), 255, dtype=np.uint8)))
        self.palette_data = palette
        
        if len(image.shape) > 2:
            self.height, self.width, channels = image.shape
            if channels == 3:
                #image = np.insert(image, -1, 255, axis=2)
                image = np.dstack((image, np.full(image.shape[:2], 255, dtype=np.uint8)))
        else:
            self.height, self.width = image.shape
            channels = 1
        
        self.image_data = image
        
        if self.color_mode is not None and palette is not None and self.color_mode.is_color:
                raise ValueError("Cannot create palettized image with non-palette color mode!")
        elif self.color_mode is None:
            if channels == 4:
                self.color_mode = TMXFile.PSMTC32
            elif channels == 3:
                self.color_mode =  TMXFile.PSMTC24
            elif channels == 1:
                assert palette is not None
                if palette.shape[0] == 256:
                    self.color_mode =  TMXFile.PSMT8
                elif palette.shape[0] == 16:
                    self.color_mode =  TMXFile.PSMT4
        return self
    def from_tmx(file_path):
        with open(file_path, "rb") as tmxfile:
            tmxfile.seek(8, 0)
            if TMXFile.file_tag != tmxfile.peek(4)[:4]:
                raise RuntimeError(f'Not a valid TMX File:{file_path}')
            tmxfile.seek(17, 0)
            palette_mode = TMXFile.format_codes[int.from_bytes(tmxfile.read(1))]
            width  = int.from_bytes(tmxfile.read(2), 'little')
            height = int.from_bytes(tmxfile.read(2), 'little')
            color_mode = TMXFile.format_codes[int.from_bytes(tmxfile.read(1))]
            tmxfile.seek(64, 0)
            palette_data = None
            if not color_mode.is_color:
                palette_length = (color_mode.palette_size * palette_mode.get_length()) // 8
                palette_data = palette_mode.from_bytes(np.fromfile(tmxfile, palette_mode.dtype, count=palette_length))
            image_length = ((width * height) * color_mode.get_length()) // 8
            image_data = color_mode.from_bytes(np.fromfile(tmxfile, color_mode.dtype, count=image_length), palette_data)
        return image_data.reshape((height, width, -1)) 
    
    def to_tmx(self, file_path):
        palette_bytes = []
        if self.palette_data is not None:
            if self.palette_data.shape[0] == 256:
                palette_swap = self.palette_data.reshape((8, 4, 8, -1))
                _temp = palette_swap[:,1,:,:].copy()
                palette_swap[:,1,:,:] = palette_swap[:,2,:,:]
                palette_swap[:,2,:,:] = _temp
                out_palette = palette_swap.reshape((256, -1))
            else:
                out_palette = self.palette_data
            palette_bytes = self.palette_mode.to_bytes(out_palette)
        
        image_bytes = self.color_mode.to_bytes(self.image_data)
        file_len = 16 + 48 + len(palette_bytes) + len(image_bytes)
        
        with open(file_path, "wb") as tmxfile:
            # File Header
            tmxfile.write(np.uint16(2).tobytes())
            tmxfile.write(np.uint16(self.user_id).tobytes())
            tmxfile.write(np.uint32(file_len).tobytes())
            tmxfile.write(self.file_tag)
            tmxfile.write(np.uint32(0).tobytes()) # Padding
            # Picture Header
            tmxfile.write(np.uint8((self.palette_data is not None)).tobytes())
            tmxfile.write(np.uint8(self.palette_mode.code).tobytes())
            tmxfile.write(np.uint16(self.width).tobytes())
            tmxfile.write(np.uint16(self.height).tobytes())
            tmxfile.write(np.uint8(self.color_mode.code).tobytes())
            tmxfile.write(np.uint32(0).tobytes()) #Mipmap Info (Unsupported)
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
            if len(palette_bytes) > 0:   
                tmxfile.write(palette_bytes)
            # Image Data
            tmxfile.write(image_bytes)
            # End Padding
            padding = tmxfile.tell() % 16
            if padding > 0:
                tmxfile.write(bytes(padding))