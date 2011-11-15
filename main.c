#include <stdio.h>

#include <cv.h>
#include <highgui.h>

#include "pcg.h"
#include "compressiveSensing.h"
#include "motionDeblur.h"

/*
  argv[1] : original image
  argv[2] : blurred image
  argv[3] : result
 */
int main(int argc, char* argv[] ){

  IplImage *original = cvLoadImage( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  IplImage *buf = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  IplImage *captured;

  

  CvSize imgSize = cvSize( 320, 320 );
  int psfSize = 16;
  int psfSizeSquare = psfSize*psfSize;

  if( original->height == buf->height && original->width == buf->width ){
    captured = buf;
  }
  else{
    CvMat* psf = cvCreateMat( psfSize, psfSize , CV_32FC1);
    IplImage* tmp = cvCreateImage( cvSize( psfSize, psfSize), IPL_DEPTH_8U, 1);
    cvResize( buf, tmp, CV_INTER_AREA);
    cvConvert( tmp, psf );
    cvNormalize( psf, psf, 1.0, 0.0, CV_L1, NULL);
    captured = cvCreateImage( cvGetSize(original), IPL_DEPTH_8U, 1);    
    cvFilter2D( original, captured, psf, cvPoint( -1, -1) );
    int piyo;
    cvSaveImage( "blurredLenna.png", captured, &piyo);
  }


  MotionDBL *mdbl = createMotionDBLStruct( captured, cvSize( psfSize, psfSize ));
  solveMotionDeblurring(mdbl);
  IplImage *dst = cvCreateImage( cvGetSize(captured), IPL_DEPTH_8U, 1);
  cvConvertScale( mdbl->original, dst, 256.0, 0.0);
  cvSaveImage( argv[3], dst, 0);
  return 0;

  // debug for compressive sensing

  CSstruct* cs = createCPStructure( psfSizeSquare, imgSize.width*imgSize.height);
  printf("create struct done\n");
  packImageToCSStruct( original, captured, imgSize, cvSize(psfSize, psfSize), cs);
  
  solveCompressiveSensing( cs );

  IplImage* img = cvCreateImage( cvSize( psfSize, psfSize), IPL_DEPTH_8U, 1);
  double norm = cvNorm( cs->x, NULL, CV_L1, NULL);
  printf("norm = %e \n", norm);
  for( int h = 0; h <  psfSize; ++h){
    for( int w = 0; w <  psfSize; ++w){
      int idx = h * psfSize + w;
      CV_IMAGE_ELEM( img, uchar, h, w) 
	= fabs ( CV_MAT_ELEM( *cs->x, double, idx, 0 ) * 25000.0 / norm);
    }
  }
  cvSaveImage( argv[3], img, 0);
  

  return 0;
}



