#ifndef __PCG___
#define __PCG___

#include <cv.h>
#include <stdio.h>

// Preconditioned Conjugate Gradient
// 前処理付き共役勾配法
// Ax = b の方程式をxについて解く手法
// 前処理付き行列として Pinv * A ~ I
// になるような行列を渡す
void PCG( CvMat* A, CvMat* b, CvMat* x, CvMat* Pinv, double th);

#endif
