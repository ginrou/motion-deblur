#ifndef __COMPRESSIVE__
#define __COMPRESSIVE__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>


#include "pcg.h"

#define MAT( mat, r, c )  (CV_MAT_ELEM( mat, double, r, c ))
#define SQUARE( a )  ((a)*(a))


//|| Ax - y ||2 をとくための構造体
typedef struct _CSstruct{
  
  // size
  int xSize; // number of unknown variables
  int ySize; // number of equations

  // input and output data
  CvMat* A;
  CvMat* x;
  CvMat* y;
  
  // computing data
  CvMat* u; // 補助変数みたいなもの
  CvMat* dx; // 探索方向
  CvMat* du; // 探索方向
  double t; // 探索ステップ 
  CvMat* nu; // 双対変数
  double eta; // 双対ギャップ
  
  //For PCG : P*var = -b , solve for var
  CvMat* subMat[4];
  CvMat* P;
  CvMat* b;
  CvMat* Pinv;
  CvMat* var;

  // buffers
  CvMat* buf; // for convex function ( ySize * 1)
  CvMat* bufX; // for duality gap scaling parameter ( xSize * 1);
  CvMat* tmpX;
  CvMat* tmpU;
  



  // parameters
  double lambda; // normalize
  double epRel;  // 許容誤差
  double alpha;  // x,u を更新する時
  double beta;  // x,uを更新する時 
  double mu;  // t を更新する時
  double sMin; // minimum of scaling constant
  double nabla;

}CSstruct;


CSstruct* createCPStructure( IplImage* input, IplImage* output , int filterSize, CvSize imgSize);

void releaseCPStructure( CSstruct *cp );

IplImage *solveCPImage( IplImage* src, IplImage *filter);

void solveCompressiveSensing( CSstruct *cs );

#endif
