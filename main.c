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
  IplImage *buf = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE);
  IplImage *captured;

  CvSize imgSize = cvSize( 360, 360 );
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

  CSstruct* cs = createCPStructure( NULL, NULL, psfSizeSquare, imgSize);
  printf("create struct done\n");

  // packing captured
  for( int h = 0; h < imgSize.height; ++h){
    for( int w = 0; w < imgSize.width; ++w){
      int y_margin = captured->height/2 - imgSize.height/2;
      int x_margin = captured->width/2 - imgSize.width/2;
      int y = h + y_margin;
      int x = w + x_margin;
      int idx = h * imgSize.width + w;
      if( y < 0 || y >= captured->height || x < 0 || x >= captured->width ) continue;
      else CV_MAT_ELEM( *cs->y, double, idx, 0) = (double)CV_IMAGE_ELEM( captured, uchar, y, x) / 256.0;
    }
  }

  // packing original
  cvSetZero( cs->A );
  for( int h = 0; h < imgSize.height; ++h){
    for( int w = 0; w < imgSize.width; ++w){
      int y_margin = original->height/2 - imgSize.height/2;
      int x_margin = original->width/2 - imgSize.width/2;

      for( int y = 0;y < psfSize; ++y){
	for( int x = 0; x < psfSize; ++x){
	  int row = h * imgSize.width + w;
	  int col = y * psfSize + x;
	  int Xidx = w + x - psfSize/2;
	  int Yidx = h + y - psfSize/2;
	  if( Yidx < 0 || Yidx >= original->height || Xidx < 0 || Xidx >= original->width ) continue;
	  else CV_MAT_ELEM( *cs->A, double, row, col ) = (double)CV_IMAGE_ELEM( original, uchar, Yidx, Xidx) / 256.0;
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



