#ifndef __COMPRESSIVE__
#define __COMPRESSIVE__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>



#define MAT( mat, r, c )  (CV_MAT_ELEM( mat, double, r, c ))
#define SQUARE( a )  ((a)*(a))
#define cvSetValue( arr, val ) cvSet( arr, cvScalarAll( val ), NULL )

//|| Ax - y ||2 + |x|1をとくための構造体
typedef struct _CSstruct{
  
  // size
  int xSize; // 求める解の次元
  int ySize; // データ点数

  // data
  CvMat* A; // sparse matrix
  CvMat* x; // argument to minimize
  CvMat* y; // captured image 

  // arguments for newton method
  CvMat* u; // バリア変数
  CvMat* z; // z = Ax-y
  CvMat* f;
  CvMat* nu; // 双対変数
  double pobj; // 主問題の値
  double dobj; // 双対問題の値
  double gap; // 双対ギャップ
  double t;

  // arguments for PCG
  CvMat *d1, *d2, *q1, *q2;
  CvMat* gradPhi;
  CvMat *prb, *prs;
  CvMat* dxu;
  CvMat* diagxtx;
  CvMat* P;
  CvMat* Pinv;
  CvMat *x1, *x2, *y1, *y2;

  // argumetns for backtrack line search
  CvMat* dx; 
  CvMat* du;
  CvMat* newX;
  CvMat* newU;
  CvMat* newF;
  CvMat* newZ;


  // パラメータ
  double mu; // tの更新パラメータ
  int MAX_NT_ITER;
  double alpha;
  double beta;
  int MAX_LS_ITER;
  double lambda; // 更新幅
  double retol; // 相対許容誤差
  double eta;
  double s; // 双対変数のスケーリングパラメータ

}CSstruct;
#include "pcg.h"

CSstruct* createCPStructure( int filterSize, int imgSize);
void releaseCPStructure( CSstruct *cp );
IplImage *solveCPImage( IplImage* src, IplImage *filter);
void solveCompressiveSensing( CSstruct *cs );

void packImageToCSStruct( IplImage* input, IplImage* output, CvSize imgSize, CvSize psfSize, CSstruct *cs);
void packImageToCSStructVarianceAligned( IplImage* input, IplImage* output, CvSize imgSize, CvSize psfSize, CSstruct *cs);


#endif
