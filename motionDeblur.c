#include "motionDeblur.h"


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

void PSFCopyForFFT( const CvMat *src, CvMat *dst){
  cvSetZero(dst);
  for( int h = 0; h < src->rows; ++h){
    for(int w = 0; w < src->cols; ++w){
      int y = h - src->rows/2;
      int x = w - src->cols/2;
      y += ( y<0 ) ? dst->rows : 0;
      x += ( x<0 ) ? dst->cols : 0;
      MAT( *dst, y, x) = MAT( *src, h, w);
    }
  }
}

void updatePsi(CvMat* Psi, CvMat *smooth, CvMat* gradI, CvMat * gradL, double lambda[2], double gamma);


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
    mdbl->Latent[i] = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
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

  // variables for stopping criterion
  mdbl->prevOriginal = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->prevPsiX = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->prevPsiY = cvCreateMat( mdbl->height, mdbl->width, CV_64FC1);
  mdbl->prevPSF = cvCreateMat( mdbl->psfHeight, mdbl->psfWidth,CV_64FC1);
  
  /*****************************************/
  /*             convert data              */
  /*****************************************/
  cvConvertScale( captured, mdbl->captured, 1.0/256.0, 0.0);


  /*****************************************/
  /*    compute constants at PSI step      */
  /*****************************************/
  
  // gradient of captured image
  gradient( mdbl->captured, mdbl->grad_i_x, kXdirection);
  gradient( mdbl->captured, mdbl->grad_i_y, kYdirection);
  
  // smooth region Omega
  double varianceTh = 0.5 / 256.0;

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
  MAT( *gradRe, 0, 0 ) = 1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0], mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1], mdbl->GradSum[1] );


  weight.val[0] /= 2.0;
  // grad x -> [ -1  1]
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) = -1.0;
  MAT( *gradRe, 0, 1 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0], mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1], mdbl->GradSum[1] );

  // grad y -> [-1  1]'
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) = -1.0;
  MAT( *gradRe, 0, 1 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0], mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1], mdbl->GradSum[1] );


  weight.val[0] /= 2.0;
  // grad xx -> [ 1 -2  1]
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) =  1.0;
  MAT( *gradRe, 0, 1 ) = -2.0;
  MAT( *gradRe, 0, 2 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0], mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1], mdbl->GradSum[1] );

  // grad yy -> [ 1 -2  1]'
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) =  1.0;
  MAT( *gradRe, 1, 0 ) = -2.0;
  MAT( *gradRe, 2, 0 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0], mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1], mdbl->GradSum[1] );


  // grad xy -> [ 1 -1]
  //            [-1  1]
  cvSetZero( gradRe );  cvSetZero( gradIm );
  MAT( *gradRe, 0, 0 ) =  1.0;
  MAT( *gradRe, 0, 1 ) = -1.0;
  MAT( *gradRe, 1, 0 ) = -1.0;
  MAT( *gradRe, 1, 1 ) =  1.0;
  myFFT( gradRe, gradIm, mdbl->FFTRegion );
  cvScaleAdd( gradRe, weight, mdbl->GradSum[0], mdbl->GradSum[0] );
  cvScaleAdd( gradIm, weight, mdbl->GradSum[1], mdbl->GradSum[1] );

  
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
  mdbl->L_TH = L_STEP_TH;

  printf("create motion deblurring structure done\n");


  return mdbl;
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

  double lambda[2] = { mdbl->lambda[0], mdbl->lambda[1] };


  /*****************************************/
  /*              MAIN LOOP                */
  /*****************************************/
  int main_itr;
  for( main_itr = 0; main_itr < 10; ++main_itr){


    //----------------------------------------//
    // L-STEP iteration
    //----------------------------------------//
    int l_itr;
    double gamma = INIT_GAMMA;
    for(l_itr = 0; l_itr < 10; ++l_itr ){

      // move psi and original to previous
      cvConvert( mdbl->original, mdbl->prevOriginal);
      cvConvert( mdbl->psi_x, mdbl->prevPsiX );
      cvConvert( mdbl->psi_y, mdbl->prevPsiY );


      /*****************************************/
      /*              PSI STEP                 */
      /*****************************************/
      gradient( mdbl->original, mdbl->grad_l_x, kXdirection);
      gradient( mdbl->original, mdbl->grad_l_y, kYdirection);

      updatePsi(mdbl->psi_x, mdbl->SmoothRegion, mdbl->grad_i_x, mdbl->grad_l_x,
		lambda, gamma);
      updatePsi(mdbl->psi_y, mdbl->SmoothRegion, mdbl->grad_i_y, mdbl->grad_l_y,
		lambda, gamma);

      /*****************************************/
      /*               L-STEP                  */
      /*****************************************/
      // fourier transform of psi_x and psi_y, psf
      cvConvert( mdbl->psi_x, mdbl->Psi_X[0] ); // psi_x -> PsiX (real part)
      cvConvert( mdbl->psi_y, mdbl->Psi_Y[0] ); // psi_y -> PsiY (real part)
      PSFCopyForFFT( mdbl->psf, mdbl->Filter[0] ); // psf  -> PSF

      myFFT( mdbl->Psi_X[0], mdbl->Psi_X[1], mdbl->FFTRegion ); // FFT of PsiX
      myFFT( mdbl->Psi_Y[0], mdbl->Psi_Y[1], mdbl->FFTRegion ); // FFT of PsiX
      myFFT( mdbl->Filter[0],  mdbl->Filter[1],  mdbl->FFTRegion ); // FFT of PSF

      // compute latent image in Frequency domain
      for( h = 0; h < height; ++h){
	for( w = 0; w < width; ++w){

	  double Ir  = MAT( *mdbl->Captured[0], h, w);
	  double Ii  = MAT( *mdbl->Captured[1], h, w);
	  double Fr  = MAT( *mdbl->Filter[0], h, w);
	  double Fi  = MAT( *mdbl->Filter[1], h, w);
	  double GSr = MAT( *mdbl->GradSum[0], h, w);
	  double GSi = MAT( *mdbl->GradSum[1], h, w);
	  double GXr = MAT( *mdbl->Grad_X[0], h, w);
	  double GXi = MAT( *mdbl->Grad_X[1], h, w);
	  double GYr = MAT( *mdbl->Grad_Y[0], h, w);
	  double GYi = MAT( *mdbl->Grad_Y[1], h, w);
	  double PXr = MAT( *mdbl->Psi_X[0], h, w);
	  double PXi = MAT( *mdbl->Psi_X[1], h, w);
	  double PYr = MAT( *mdbl->Psi_Y[0], h, w);
	  double PYi = MAT( *mdbl->Psi_Y[1], h, w);

	  double denom_r, denom_i, nume_r, nume_i; // denom : 分母   nume : 分子
	  denom_r = ( Fr*Fr + Fi*Fi ) * GSr + gamma * ( GXr*GXr + GXi*GXi + GYr*GYr + GYi*GYi);
	  denom_i = ( Fr*Fr + Fi*Fi ) * GSi;

	  nume_r  = GSr * ( Fr*Ir + Fi*Ii ) - GSi * ( Fr*Ii - Fi*Ir );
	  nume_i  = GSr * ( Fr*Ii - Fi*Ir ) + GSi * ( Fr*Ir + Fi*Ii );
	  nume_r += gamma * ( GXr*PXr + GXi*PXi + GYr*PYr + GYi*PYi );
	  nume_i += gamma * ( GXr*PXi - GXi*PXr + GYr*PXi - GYi*PXr );

	  MAT( *mdbl->Latent[0], h, w) = 
	    ( nume_r*denom_r + nume_i*denom_i) / ( nume_r*nume_r + nume_i*nume_i );
	  MAT( *mdbl->Latent[1], h, w) = 
	    ( nume_i*denom_r - nume_r*denom_i) / ( nume_r*nume_r + nume_i*nume_i );
	}
      }

      // IFFT of estimated latent image
      myIFFT( mdbl->Latent[0], mdbl->Latent[1], mdbl->FFTRegion );

      cvConvert( mdbl->Latent[0], mdbl->original ); // copy Latent to orignal

      // stopping creterion
      double normOriginal = cvNorm( mdbl->prevOriginal, mdbl->original, CV_L2, NULL);
      double normPsiX = cvNorm( mdbl->prevPsiX, mdbl->psi_x, CV_L2, NULL);
      double normPsiY = cvNorm( mdbl->prevPsiY, mdbl->psi_y, CV_L2, NULL);
      double normPsi = sqrt( normPsiX*normPsiX + normPsiY*normPsiY );
      if( normOriginal < mdbl->L_TH && normPsi < mdbl->L_TH){ 
	break;
      }else{
	gamma *= 2.0;
      }
      
      printf("iteration of L-step : %d, ", l_itr);
      printf("orignal norm : %lf, psi norm %lf\n", normOriginal, normPsi);

    }


    /*****************************************/
    /*               F-STEP                  */
    /*****************************************/
    cvConvert( mdbl->psf, mdbl->prevPSF ); // convert previous result
    packImageToCSStructMat( mdbl->original, mdbl->captured, mdbl->imgSize, mdbl->psfSize, mdbl->cs );
    solveCompressiveSensing( mdbl->cs );
    unpackCSStrutct2Mat( mdbl->cs, mdbl->psf );



    /*****************************************/
    /*          STOPPING CRITERION           */
    /*          AND UPDATE LAMBDA            */
    /*****************************************/
    double psfNorm = cvNorm( mdbl->prevPSF, mdbl->psf, CV_L1, NULL );
    if( psfNorm < 0.01 ){
      printf("iteration done\n");
    }else{
      lambda[0] /= KAPPA_0;
      lambda[1] /= KAPPA_1;
    }

    char filename[256];
    IplImage* img = cvCreateImage( mdbl->imgSize, IPL_DEPTH_8U, 1);
    sprintf(filename,"test/original%02d.png", main_itr);
    cvConvertScale( mdbl->original, img, 256.0 , 0.0);
    cvSaveImage( filename, img, 0);
    IplImage* fil = cvCreateImage( mdbl->psfSize, IPL_DEPTH_8U, 1);
    sprintf(filename,"test/psf%02d.png", main_itr);
    cvConvertScale( mdbl->psf, fil, 256.0 , 0.0);
    cvSaveImage( filename, fil, 0);

  }// main loop

  printf("solving image deblurring done\n");
  return;

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
      MAT( *dst, h, w) = MAT( *src, h+dy, w+dx) - MAT( *src, h, w);
    }
  }
  return;
}


/*****************************************
  batch function to update Psi
 *****************************************/
void updatePsi( CvMat* Psi, CvMat *smooth, CvMat* gradI, CvMat * gradL, double lambda[2], double gamma)
{
  double a = 6.1 * pow( 10.0, -4.0 );
  double b = 5.0;
  double k = 2.7 / 2.0;
  double lt = 1.8526;

  double newPsi[3];
  double E[3]; // value of energy function

  for( int h = 0; h < Psi->rows; ++h){
    for( int w = 0; w < Psi->cols; ++w ){

      double m = MAT( *smooth, h, w);
      double gi = MAT( *gradI, h, w);
      double gl = MAT( *gradL, h, w);
      
      newPsi[0] = (lambda[1]*m*gi + gamma*gl - lambda[0]*k )/( lambda[1]*m + gamma);
      newPsi[1] = (lambda[1]*m*gi + gamma*gl + lambda[0]*k )/( lambda[1]*m + gamma);
      newPsi[2] = (lambda[1]*m*gi + gamma*gl )/( a*lambda[0] + lambda[1]*m + gamma);

      E[0] = E[1] = E[2] = DBL_MAX / 2.0;

      if( 0 <= newPsi[0] && newPsi[0] <= lt)
	E[0] = lambda[0] * k * fabs(newPsi[0]);

      if( -lt <= newPsi[1] && newPsi[1] <= lt )
	E[1] = lambda[0] * k * fabs(newPsi[1]);

      if( newPsi[2] <= -lt && lt <= newPsi[2] )
	E[2] = lambda[0] * fabs( a*newPsi[2]*newPsi[2] + b );

      for(int i = 0; i < 3; ++i){
	E[i] += lambda[1] * m * SQUARE( newPsi[i] - gi );
	E[i] += gamma * SQUARE( newPsi[i] - gl );
      }

      double newE = min( E[0], min( E[1], E[2] ));
      for(int i = 0; i < 3; ++i){
	if( newE == E[i] ) MAT( *Psi, h, w) = newPsi[i];
      }


    }
  }

}
