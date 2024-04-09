from rembg import remove
from PIL import Image

input_path = 'C:/Users/venus/OneDrive/Desktop/deep learning/dataset/Alstonia Scholaris (P2)/0003_0001.JPG'
output_path = 'output.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)