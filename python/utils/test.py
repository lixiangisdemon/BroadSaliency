import numpy as np
from extrafeature import ExtraFeature
img = np.array(np.arange(40*50*3).reshape((40,50,3)), dtype=np.uint8)
gt = np.ones((40,50,3), dtype=np.uint8) * 255
E = ExtraFeature()
res = E.extract_feature(img, gt)
print( res[1].shape)
res1 = E.cvt_gridfeature(res[1], res[2])
print( res1[0].shape)
#res = E.extract_dists(img)
#print res
