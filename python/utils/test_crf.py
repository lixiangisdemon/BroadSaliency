from supercrf import SuperCRF
from extrafeature import ExtraFeature
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
#img = np.array(np.arange(40*50*3).reshape((40,50,3)), dtype=np.uint8)
img = np.array(Image.open('../../images/0037.jpg'))
gt = np.array(Image.open('../../images/0037.png'))
gt = np.concatenate((gt[:,:,np.newaxis], gt[:,:,np.newaxis],gt[:,:,np.newaxis]), axis=-1)
E = ExtraFeature()
[seg, feat, lbl] = E.extract_feature(img, gt)
print ('details1:')
print (feat[:,27].max(), feat[:,28].max())
print( feat.max(), ' ', feat.min() )
print( max(lbl), ' ', min(lbl) )
res = E.cvt_gridfeature(feat, lbl)
print ('details2:')
print( res[0].max(), ' ', res[0].min() )
print( max(res[1]), ' ', min(res[1]) )
res_gt = np.array(res[1]).reshape(32, 32)
mask = np.random.randint(0, 256, img.shape)
K = np.max(seg[:]) + 1
kernel = np.ones((K, K), dtype=np.float32) - 0.5
res = SuperCRF(img, mask, kernel, seg)
print( res )
plt.imshow(res_gt)
plt.show()