#ifndef __MOTION_DEBLUR__
#define __MOTION_DEBLUR__
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>

#include "compressiveSensing.h"

// motion deblur の構造体
// y = f*x の形で現れるぶれ写真を取り除く
typedef struct _MDBL{

  // image size
  int width;
  int height;
  CvSize imgSize;

  // psf size
  int psfWidth;
  int psfHeight;
  CvSize psfSize;

  // latent, captured, psf images
  CvMat *captured;
  CvMat *psf;
  CvMat *original;

  // variables for PSI step
  CvMat *psi_x;
  CvMat *psi_y;
  CvMat *grad_i_x; // horizontal gradient of captured image I
  CvMat *grad_i_y; // vertical gradient of captured image I
  CvMat *grad_l_x;
  CvMat *grad_l_y;
  CvMat *SmoothRegion; // Omega in paper which is for hogehoge
  
  // variables for L step
  CvMat* GradSum[2]; // gradSum = sum_of( weight(k) * |F( diff_operator)|
  CvMat* Grad_X[2];  // x方向の微分フィルタのフーリエ変換
  CvMat* Grad_Y[2];  // y方向の微分フィルタのフーリエ変換
  CvMat* Captured[2]; // 撮影画像のフーリエ変換
  CvMat* Filter[2];
  CvMat* Psi_X[2];
  CvMat* Psi_Y[2];
  CvMat* FFTRegion;


  // variablse for f step which update PSF 
  int CSRows; // pixels used to solve Compressive Sampling
  CSstruct *cs;


  // parameters 
  double lambda[2];
  double gamma;

}MotionDBL;

MotionDBL* createMotionDBLStruct( IplImage* captured, CvSize psfSize);
void solveMotionDeblurring( MotionDBL* mdbl );

void myFFT( CvMat* real, CvMat* imag, CvMat *tmpRegion);
void myIFFT( CvMat* real, CvMat* imag, CvMat *tmpRegion);

typedef enum _gradDirection{
  kXdirection,
  kYdirection
}gradDirection;
void gradient( CvMat* src, CvMat* dst, gradDirection dir);

// parameters
#define INIT_WEIGHT 50.0
#define LAMBDA_0 0.01
#define LAMBDA_1 20.0
#define INIT_GAMMA 2.0
#define CS_ROWS_RATIO 100.0 // %


#endif
