#include <stdio.h>

#include <cv.h>
#include <highgui.h>

#include "pcg.h"
#include "compressiveSensing.h"

/*
  argv[1] : original image
  argv[2] : captured image
  argv[3] : result
 */
int main(int argc, char* argv[] ){

  IplImage *original = cvLoadImage( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  IplImage *captured = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  int psfSize = 32;
  int psfSizeSquare = psfSize*psfSize;
  CSstruct* cs = createCPStructure( NULL, NULL, psfSizeSquare, cvGetSize( original ));
  printf("create struct done\n");

  // packing captured
  for( int h = 0; h < captured->height; ++h){
    for(int w = 0; w < captured->width; ++w){
      int idx = h * captured->width + w;
      CV_MAT_ELEM( *cs->y, double, idx, 0 ) 
	= (double)CV_IMAGE_ELEM( captured, uchar, h, w) / 256.0;
    }
  }

  // packing original
  cvSetZero( cs->A );
  printf("size = %d, %d\n", cs->A->rows, cs->A->cols);
  for(int h = 0; h < captured->height; ++h ){
    for( int w = 0; w < captured->width; ++w){
      
      for(int y = 0; y < psfSize; ++y){
	for(int x = 0; x < psfSize; ++x){
	  int col = y * psfSize + x;
	  int row = h * captured->width + w;
	  if( h+y > original->height || w+x > original->width ) continue;
	  CV_MAT_ELEM( *cs->A, double, row, col ) 
	    = CV_IMAGE_ELEM( original, uchar, h+y, w+x ) / 256.0;
	}
      }

    }
  }

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



