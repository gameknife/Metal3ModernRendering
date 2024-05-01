# replace .mtl file
# map_Pm -> map_metallic
# map_Bump -> map_bump
# map_Pr -> map_roughness
# remove -bm x
# remove file extension from texture file names

import os
import sys

def convertMtl(mtlPath):
    with open(mtlPath, 'r') as f:
        lines = f.readlines()
    with open(mtlPath, 'w') as f:
        for line in lines:
            if 'map_Pm' in line:
                line = line.replace('map_Pm', 'map_metallic')
            if 'map_Bump' in line:
                line = line.replace('map_Bump', 'map_bump')
            if 'map_Pr' in line:
                line = line.replace('map_Pr', 'map_roughness')
            if 'Ke' in line:
                line = line.replace('Ke', 'Ka') 
            if '-bm' in line:
                line = line.replace('-bm 1.000000', '')
            if '.png' in line:
                line = line.replace('.png', '')
            if '.jpg' in line:
                line = line.replace('.jpg', '')
            f.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python convertMtl.py <mtlPath>')
        sys.exit(1)
    mtlPath = sys.argv[1]
    if not os.path.exists(mtlPath):
        print('Error: mtl file not found')
        sys.exit(1)
    convertMtl(mtlPath)
    print('Success: mtl file converted')