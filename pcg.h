#ifndef __PCG___
#define __PCG___

#include <cv.h>
#include <stdio.h>

#include "compressiveSensing.h"

// Preconditioned Conjugate Gradient
// 前処理付き共役勾配法
// Ax = b の方程式をxについて解く手法
// 前処理付き行列として Pinv * A ~ I
// になるような行列を渡す
void PCG( CvMat* A, CvMat* b, CvMat* x, CvMat* Pinv, double th);


/************************************************************
  前処理付き共役勾配法(PCG)
  の行列の積の演算をオペレータで行うバージョン
  疎な行列に対して、A'Aを計算するのは計算量が
  大きくなるが、行列×ベクトルを２回演算する
  ことで計算量を削減する
  mulA は y = Ax を、
  mulPinv は y = Pinv*x 
　をそれぞれ計算する. 
************************************************************/
void PCG_MatMulOperator( void (*mulA)( CSstruct* cs, CvMat* u, CvMat* y), 
			 CvMat* b, CvMat* x, 
			 void (*mulPinv)( CSstruct* cs, CvMat* u, CvMat* y), 
			 double th, CSstruct* cs);


#endif
