# import numpy as np
#
# # K = channels, H = height, W = width
# K = 3
# H = 15
# W = 15
# meanpixel = np.ones((H, W, K), dtype=np.float32)
# meanpixel[:, :, 2] *= 104
# meanpixel[:, :, 1] *= 117
# meanpixel[:, :, 0] *= 123
#
# np.save('meanfile_small_to_big_15.npy', meanpixel)
#
# x = np.load('meanfile_small_to_big_15.npy').mean(1).mean(0)
# print x

import numpy as np
K = 3
H = 15
W = 15
zeros = np.zeros((H, W, K), dtype=np.float32) # K = channels, H = height, W = width
np.save('meanfile.npy', zeros)
