import cv2
import numpy as np
import glob


print("Starting...")
img_array = []
network = "PINN"
size = ()
file_list = glob.glob(f'./plots/*_{network}_Epochs.png')
file_list.sort()
for filename in file_list:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# fps defined within pinn.py fit function
fps = 24
# cv2.VideoWriter(Filename, codec, fps, size, color=True)
out = cv2.VideoWriter(f'./plots/{network}_result_vid.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Finished!")
