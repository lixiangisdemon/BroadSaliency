import numpy as np 
from opencv_mat import PyMat8U as Mat8U
from opencv_mat import PyMat as Mat
import opencv_mat
a = np.array(np.arange(15).reshape(5,1,3), dtype=np.uint8)
print( a )
mat = Mat8U(a)
print( 'mat: ' )
mat.printValues()
print( 'matnumpy: ', mat.rows, ', ', mat.cols, ', ', mat.channels() )
mat.printNumpy()
a_mat = opencv_mat.np2Mat2np(a)
print( 'np2mat2np: ' )
print( a_mat )

b = np.array(np.arange(15).reshape(1, 5,3), dtype=np.uint8)
print( 'print b ' )
print( b )
mat = Mat(b) 
print( 'print bmat' )
print( mat.get_mat() )
