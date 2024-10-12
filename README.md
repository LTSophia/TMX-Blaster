# TMX Blaster
>The only TMX tool you should ever need.

A (for now) command-line tool to convert TMX files to regular image files (such as PNG, TGA, etc.), regular image files to TMX files, TMX to TMX, with all available color/palette formats, all available options and settings. 

Made to be feature-complete with the [Atlus TMX Editor](https://tcrf.net/Shin_Megami_Tensei:_Nocturne#Leftover_TMX_Editor) that was left over in a release of Nocturne, but with less jank and easier to use with batch scripting or other applications.

## Command-Line Options:
```
usage: TMX Blaster [-h] [-q] [--batch] [-r] [-flat] [-if INFILETYPE] [-of OUTFILETYPE]
                   [--height {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768}]
                   [--width {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768}] [-p {16,256}] [-ui USERID]
                   [-b] [-uc USERCOMMENT] [--nosolidify] [-ti TEXTUREID] [-ci CLUTID]
                   [--horizontalwrapmode {repeat,clamp}] [--verticalwrapmode {repeat,clamp}]
                   [--palettetype {32,16,16S}] [--pixeltype {32,24,16,16S,8,8H,4,4HH,4HL}]
                   input [output]

The only TMX tool you should ever need.

positional arguments:
  input                 Can be a PNG, JPEG, etc. or TGA for encoding, or TMX for decoding. The folder to process in
                        Batch Processing mode.
  output                TMX file or image file to output. Defaults to TMX for encoding and TGA for decoding. The
                        folder to export to in Batch Processing mode.

options:
  -h, --help            show this help message and exit
  -q, --quiet           Disable any command-line output.
  --batch               Enable batch folder processing.
  -r, --recursive       Process all subfolders. [BATCH PROCESSING]
  -flat, --flattenoutput
                        When recursive, do not create extra folders to match the original file structure. [BATCH
                        PROCESSING]
  -if INFILETYPE, --infiletype INFILETYPE
                        Filetype to process in the folder. [BATCH PROCESSING]
  -of OUTFILETYPE, --outfiletype OUTFILETYPE
                        Filetype to export too. [BATCH PROCESSING]
  --height {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768}
                        Sets the value to resize the image's height to.
  --width {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768}
                        Sets the value to resize the image's width to.
  -p {16,256}, --palette {16,256}
                        The palette size used when encoding to TMX.
  -ui USERID, --userid USERID
                        Set the User ID of the TMX.
  -b, --bustup          Marks the TMX as being a bustup, the same as '-ui 1'.
  -uc USERCOMMENT, --usercomment USERCOMMENT
                        Set the User Comment for the TMX. Defaults to the name of the file.
  --nosolidify          Sets the program to NOT solidify the image before making it a TMX. (ADVANCED OPTION: Not
                        recommended)
  -ti TEXTUREID, --textureid TEXTUREID
                        Set the Texture ID of the TMX. (ADVANCED OPTION: Usually unnecessary)
  -ci CLUTID, --clutid CLUTID
                        Set the CLUT ID of the TMX. (ADVANCED OPTION: Unknown Functionality)
  --horizontalwrapmode {repeat,clamp}
                        The wrap mode used for horizontal in-engine effects. (ADVANCED OPTION: does not affect
                        exported image)
  --verticalwrapmode {repeat,clamp}
                        The wrap mode used for vertical in-engine effects. (ADVANCED OPTION: does not affect exported
                        image)
  --palettetype {32,16,16S}
                        The color mode used to store the palette information (ADVANCED OPTION)
  --pixeltype {32,24,16,16S,8,8H,4,4HH,4HL}
                        The color mode used to store the pixel information (ADVANCED OPTION)
```
