#include "motionDeblur.h"
double min(double a, double b){ return (a<b)?a:b; }
double max(double a, double b){ return (a>b)?a:b; }


void myFFT( CvMat* real, CvMat* imag, CvMat *tmpRegion){
  cvSetZero( imag );
  cvMerge( real, imag, NULL, NULL , tmpRegion);
  cvDFT( tmpRegion, tmpRegion, CV_DXT_FORWARD, 0);
  cvSplit( tmpRegion, real, imag , NULL, NULL);
}

void myIFFT( CvMat* real, CvMat* imag, CvMat *tmpRegion){
  cvMerge( real, imag, NULL, NULL , tmpRegion); // merge : (real, imag) -> tmp
  cvDFT( tmpRegion, tmpRegion, CV_DXT_INV_SCALE, 0); // IDFT
  cvSplit( tmpRegion, real, imag , NULL, NULL);// split : tmp -> (real, imag)
  cvPow( real, real, 2.0 ); // real = real^2
  cvPow( imag, imag, 2.0 ); // imag = imag^2
  cvAdd( real, imag, real, NULL); // real = real + imag
  cvPow( real, real, 0.5 ); // real = sqrt(real)
}




MotionDBL* createMotionDBLStruct( IplImage* captured, CvSize psfSize){

  MotionDBL *mdbl = (MotionDBL*)malloc( sizeof(MotionDBL));

  mdbl->width     = captured->width;
  mdbl->height    = captured->height;
  mdbl->imgSize   = cvGetSize( captured );
  mdbl->psfWidth  = psfSize.width;
  mdbl->psfHeight = psfSize.height;
  mdbl->psfSize   = psfSize;


  /*****************************************/
  /*        memory allocation              */
  /*****************************************/

  // variables for basic image structure
  mdbl->captured = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->original = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->psf = cvCreateMat( mdbl->psfHeight, mdbl->psfWidth,CV_64FC1);

  // variables for PSI step
  mdbl->psi_x = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->psi_y = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->grad_i_x = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->grad_i_y = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->grad_l_x = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->grad_l_y = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->SmoothRegion = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);

  
  // variables for L step
  for( int i = 0; i < 2; ++i){
    mdbl->GradSum[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
    mdbl->Grad_X[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
    mdbl->Grad_Y[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
    mdbl->Captured[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
    mdbl->Filter[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
    mdbl->Psi_X[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
    mdbl->Psi_Y[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  }
  mdbl->FFTRegion = cvCreateMat( mdbl->height, mdbl->width, CV_64FC2);

  // variables for f step
  mdbl->CSRows = mdbl->width * mdbl->height;
  mdbl->cs = createCPStructure( psfSize.width * psfSize.height, mdbl->CSRows );
  
  /*****************************************/
  /*             convert data              */
  /*****************************************/
  cvConvertScale( captured, mdbl->captured, 1.0/256.0, 1.0);


  /*****************************************/
  /*    compute constants at PSI step      */
  /*****************************************/
  
  // gradient of captured image
  gradient( mdbl->captured, mdbl->grad_i_x, kXdirection);
  gradient( mdbl->captured, mdbl->grad_i_y, kYdirection);
  
  // smooth region Omega
  double varianceTh = 5.0 / 256.0;

  // compute mean 
  CvMat* meanMat = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  for( int h = 0; h < mdbl->height; ++h){    
    for(int w = 0; w < mdbl->width; ++w){
      MAT( *meanMat, h, w) = 0.0;
      for( int y = 0; y < mdbl->psfHeight; ++y){	
	for( int x = 0; x < mdbl->psfWidth; ++x){
	  int py = h + y - mdbl->psfHeight/2;	  
	  int px = w + x - mdbl->psfWidth/2;
	  if( py < 0 || py >= mdbl->height || px < 0 || px >= mdbl->width ){
	    continue;
	  }else{
	    MAT(*meanMat, h, w) += MAT(*mdbl->captured, py, px );
	  }// else
	}// x
      }// y
      MAT(*meanMat, h, w) /= (double)(mdbl->psfHeight * mdbl->psfWidth );
    }// w
  }// h

  // compute variance
  for( int h = 0; h < mdbl->height; ++h){    
    for(int w = 0; w < mdbl->width; ++w){
      double var = 0.0;
      for( int y = 0; y < mdbl->psfHeight; ++y){	
	for( int x = 0; x < mdbl->psfWidth; ++x){
	  int py = h + y - mdbl->psfHeight/2;	  
	  int px = w + x - mdbl->psfWidth/2;
	  if( py < 0 || py >= mdbl->height || px < 0 || px >= mdbl->width ){
	    continue;
	  }else{
	    var += SQUARE( MAT(*meanMat, h, w) - MAT(*mdbl->captured, py, px) );
	  }// else
	}// x
      }// y
      var /= (double)(mdbl->psfHeight * mdbl->psfWidth );
      if( var < varianceTh ){
	MAT(*mdbl->SmoothRegion, h, w) = 1.0;
      }else{
	MAT(*mdbl->SmoothRegion, h, w) = 0.0;
      }
    }// w
  }// h
  cvReleaseMat( &meanMat );

  /*****************************************/
  /*    compute constants at  L step       */
  /*****************************************/
  
  // fourier transform of grad x 
  cvSetZero( mdbl->Grad_X[0] );
  MAT( *mdbl->Grad_X[0], 0, 0 ) = -1.0;
  MAT( *mdbl->Grad_X[0], 0, 1 ) =  1.0;
  myFFT( mdbl->Grad_X[0], mdbl->Grad_X[1], mdbl->FFTRegion );

  // fourier transform of grad y
  cvSetZero( mdbl->Grad_Y[0] );
  MAT( *mdbl->Grad_Y[0], 0, 0 ) = -1.0;
  MAT( *mdbl->Grad_Y[0], 1, 0 ) =  1.0;
  myFFT( mdbl->Grad_Y[0], mdbl->Grad_Y[1], mdbl->FFTRegion );

  // fourier transform of captured image
  cvConvert( mdbl->captured, mdbl->Captured[0] );
  myFFT( mdbl->Captured[0], mdbl->Captured[1] , mdbl->FFTRegion);

  //----------------------------------------//
  // compute grad sum
  //----------------------------------------//
  CvMat *gradRe = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  CvMat *gradIm = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  CvScalar weight = cvScalarAll( INIT_WEIGHT );

  cvSetZero( mdbl->GradSum[0] );
  cvSetZero( mdbl->GradSum[1] );
  
  // grad 0 -> そのまんま
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) = 0.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1] );


  weight.val[0] /= 2.0;
  // grad x -> [ -1  1]
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) = -1.0;
  MAT( *gradRe, 0, 1 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1] );
  
  // grad y -> [-1  1]'
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) = -1.0;
  MAT( *gradRe, 0, 1 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1] );

  
  weight.val[0] /= 2.0;
  // grad xx -> [ 1 -2  1]
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) =  1.0;
  MAT( *gradRe, 0, 1 ) = -2.0;
  MAT( *gradRe, 0, 2 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1] );

  // grad yy -> [ 1 -2  1]'
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) =  1.0;
  MAT( *gradRe, 1, 0 ) = -2.0;
  MAT( *gradRe, 2, 0 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1] );

  // grad xy -> [ 1 -1]
  //            [-1  1]
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) =  1.0;
  MAT( *gradRe, 0, 1 ) = -1.0;
  MAT( *gradRe, 1, 0 ) = -1.0;
  MAT( *gradRe, 1, 1 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1] );
  
  cvReleaseMat( &gradRe);
  cvReleaseMat( &gradIm);
  
  /*****************************************/
  /*   settings of Compressive sampling    */
  /*****************************************/
  mdbl->cs->lambda = 1.0;



  /*****************************************/
  /*               paramteres              */
  /*****************************************/
  mdbl->lambda[0] = LAMBDA_0;
  mdbl->lambda[1] = LAMBDA_1;
  mdbl->gamma = INIT_GAMMA;

  return;
}




void solveMotionDeblurring( MotionDBL* mdbl )
{
  int h, w;
  int height = mdbl->height;
  int width  = mdbl->width;

  /*****************************************/
  /*           INITIALIZE                  */
  /*****************************************/
  cvConvert( mdbl->captured, mdbl->original );
  cvSetValue( mdbl->psf, 1.0 / (double)( mdbl->psfWidth * mdbl->psfHeight ));

  CvMat *captured = mdbl->captured;
  CvMat *psf = mdbl->psf;
  CvMat *original = mdbl->original;

  CvMat *smooth = mdbl->SmoothRegion;  
  CvMat *psi[2];
  CvMat *grad_i[2];
  CvMat *grad_l[2];
  psi[0] = mdbl->psi_x;  psi[1] = mdbl->psi_y;
  grad_i[0] = mdbl->grad_i_x; grad_i[1] = mdbl->grad_i_y;
  grad_l[0] = mdbl->grad_l_x; grad_l[1] = mdbl->grad_l_y;


  CvMat *Grad[2][2];
  CvMat *Psi[2][2];
  Grad[0][0] = mdbl->Grad_X[0]; Grad[0][1] = mdbl->Grad_X[1];
  Grad[1][0] = mdbl->Grad_Y[0]; Grad[1][1] = mdbl->Grad_Y[1];
  Psi[0][0]  = mdbl->Psi_X[0];   Psi[0][1] = mdbl->Psi_X[1]; 
  Psi[1][0]  = mdbl->Psi_Y[0];   Psi[1][1] = mdbl->Psi_Y[1]; 

  double lambda[2] = { mdbl->lambda[0], mdbl->lambda[1] };


  
  /*****************************************/
  /*              MAIN LOOP                */
  /*****************************************/
  int main_itr;
  for( main_itr = 0; main_itr < 100; ++main_itr){


    //----------------------------------------//
    // L-STEP iteration
    //----------------------------------------//
    int l_itr;
    double gamma = INIT_GAMMA;
    for(l_itr = 0; l_itr < 100; ++l_itr ){
      /*****************************************/
      /*              PSI STEP                 */
      /*****************************************/
      double a = 6.1 * pow( 10.0, -4.0 );
      double b = 5.0;
      double k = 2.7/2.0;
      double lt = 1.8526;
      gradient( original, grad_l[0], kXdirection);
      gradient( original, grad_l[1], kYdirection);

      for( int dir = 0; dir < 2; ++dir ){ // dir 0 -> x , dir 1 -> y
	double newPsi[3];
	for( h = 0; h < height; ++h){
	  for( w = 0; w < width; ++w){
	    
	    newPsi[0] = 
	      (lambda[1] * MAT(*smooth, h, w) * MAT( *grad_i[dir], h, w) 
	       + gamma * MAT(*grad_l[dir], h, w) ) / 
	      ( a * lambda[0] + lambda[2] * MAT(*smooth, h, w) + gamma );

	    newPsi[1] = 
	      (lambda[1] * MAT(*smooth, h, w) * MAT( *grad_i[dir], h, w) 
	       + gamma * MAT(*grad_l[dir], h, w)  + lambda[0] * k) / 
	      ( lambda[2] * MAT(*smooth, h, w) + gamma );

	    newPsi[2] = 
	      (lambda[1] * MAT(*smooth, h, w) * MAT( *grad_i[dir], h, w) 
	       + gamma * MAT(*grad_l[dir], h, w)  - lambda[0] * k) / 
	      ( lambda[2] * MAT(*smooth, h, w) + gamma );

	    double newE[3];
	    if( -lt > newPsi[0] || newPsi[0] > lt ) {
	      newE[0] = lambda[0] * fabs( a*SQUARE( newPsi[0] ) + b );
	    }else{ newE[0] = DBL_MAX; }

	    if( newPsi[1] > -lt && newPsi[1] < 0  ) {
	      newE[1] = lambda[0] * fabs( -k * newPsi[1] );
	    }else{ newE[1] = DBL_MAX; }

	    if( newPsi[1] > -lt && newPsi[1] < 0  ) {
	      newE[2] = lambda[0] * fabs( k * newPsi[1] );
	    }else{ newE[1] = DBL_MAX; }

	    for( int i = 0; i< 3; ++i){
	      newE[i] += 
		lambda[1] * MAT(*smooth, h, w) *SQUARE( newPsi[i] - MAT( grad_i[dir], h, w))
		+ gamma * SQUARE( newPsi[i] - MAT( grad_l[dir], h, w));
	    }
	    
	    double minVal =  min( min( newE[0], new[1] ), new[2] );
	    for( int i = 0; i< 3; ++i)
	      if( minVal == newE[i] ) MAT( *psi[dir], h, w) = newPsi[i];

	  }//w 
	}// h
      }// dir


      /*****************************************/
      /*               L-STEP                  */
      /*****************************************/
      





      // stopping creterion


    }


    /*****************************************/
    /*               F-STEP                  */
    /*****************************************/


  

    /*****************************************/
    /*            UPDATE LAMBDA              */
    /*****************************************/
  

  }

}




void gradient( CvMat* src, CvMat* dst, gradDirection dir)
{
  int dx, dy;
  switch(dir){
  case kXdirection:
    dx = 1;    dy = 0;
    break;
  case kYdirection:
    dx = 0;    dy = 1;
    break;
  default:
    break;
  }

  for( int h = 0; h < src->rows - dy; ++h){
    for( int w = 0; w < src->cols - dx; ++w){
      MAT( *dst, h, w) = MAT( *src, h+dy, w+dx) - MAT( *dst, h, w);
    }
  }
  return;
}
