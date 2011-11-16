#include "compressiveSensing.h"

// private functions
void mulA( CSstruct *cs, CvMat* x, CvMat* y);
void mulPinv( CSstruct *cs, CvMat* x, CvMat* y);

void saveMat( CvMat *mat, char filename[] );

double min( double a, double b ){
  return  ( a<b )? a : b;
}

double max( double a, double b ){
  return  ( a>b )? a : b;
}


IplImage *solveCPImage( IplImage* src, IplImage *filter)
{
  return NULL;
}


CSstruct* createCPStructure( int filterSize, int  imgSize)
{
  CSstruct* cs = (CSstruct*)malloc(sizeof(CSstruct));

  cs->xSize = filterSize;
  cs->ySize = imgSize;
 
  int N = cs->xSize;
  int M = cs->ySize;


  // data arguments
  cs->A = cvCreateMat( M, N , CV_64FC1);
  cs->x = cvCreateMat( N, 1 , CV_64FC1);
  cs->y = cvCreateMat( M, 1 , CV_64FC1);

  /// arguments for bariar and dualtiy
  cs->u = cvCreateMat( N, 1 , CV_64FC1);
  cs->z = cvCreateMat( M, 1 , CV_64FC1);
  cs->f = cvCreateMat( 2*N, 1 , CV_64FC1);
  cs->nu = cvCreateMat( M, 1 , CV_64FC1);
  
  // inital parameters
  cs->pobj = DBL_MAX;
  cs->dobj = -DBL_MAX;

  
  // arguments to compute gradience
  cs->d1 = cvCreateMat( N, 1, CV_64FC1 );
  cs->d2 = cvCreateMat( N, 1, CV_64FC1 );
  cs->q1 = cvCreateMat( N, 1, CV_64FC1 );
  cs->q2 = cvCreateMat( N, 1, CV_64FC1 );

  // arguments for PCG
  cs->gradPhi = cvCreateMat( 2*N, 1 , CV_64FC1);
  cs->prb = cvCreateMat( N, 1, CV_64FC1 );
  cs->prs = cvCreateMat( N, 1, CV_64FC1 );
  cs->dxu = cvCreateMat( 2*N, 1 , CV_64FC1);
  cs->diagxtx = cvCreateMat( 2*N, 1 , CV_64FC1);
  cs->P = cvCreateMat( 2*N, 2*N , CV_64FC1);
  cs->Pinv = cvCreateMat( 2*N, 2*N , CV_64FC1);
  cs->x1 = cvCreateMat( N, 1, CV_64FC1);
  cs->x2 = cvCreateMat( N, 1, CV_64FC1);
  cs->y1 = cvCreateMat( N, 1, CV_64FC1);
  cs->y2 = cvCreateMat( N, 1, CV_64FC1);


  // argumnets for newton step ( backtrack line search )
  cs->dx =  cvCreateMat( N, 1 , CV_64FC1);
  cs->du =  cvCreateMat( N, 1 , CV_64FC1);
  cs->newX = cvCreateMat( N, 1 , CV_64FC1);
  cs->newU = cvCreateMat( N, 1 , CV_64FC1);
  cs->newF = cvCreateMat( 2*N, 1 , CV_64FC1);
  cs->newZ = cvCreateMat( M, 1 , CV_64FC1);

  // parameters
  cs->mu = 2.0;
  cs->MAX_NT_ITER = 100;
  cs->alpha = 0.01;
  cs->beta = 0.5;
  cs->MAX_LS_ITER = 100;
  cs->retol = 0.02;
  cs->lambda = 1.0;

  
  // initial values
  cs->eta = 0.001;
  cs->t = min( max( 1, 1/cs->lambda), 2*N / pow(10, -3) );
  cs->s = DBL_MAX;
  cs->pobj = DBL_MAX;
  cs->dobj = -DBL_MAX;
  cvSetValue( cs->x, 0.0);
  cvSetValue( cs->u, 1.0);
  for(int i = 0; i < N; ++i){
    MAT( *cs->f, i, 0 ) = MAT( *cs->x, i, 0 ) - MAT( *cs->u, i, 0 );
    MAT( *cs->f, i+N, 0 ) = -MAT( *cs->x, i, 0 ) - MAT( *cs->u, i, 0 );
  }
  cvSetValue( cs->dxu, 0.0 );
  cvSetValue( cs->diagxtx, 2.0 );

  return cs;
}

void packImageToCSStruct( IplImage* input, IplImage* output, CvSize imgSize, CvSize psfSize, CSstruct *cs)
{
  
  int N = cs->x->rows;
  int M = cs->y->rows;
  
  int y_margin = input->height/2 - imgSize.height/2;
  int x_margin = input->width/2 - imgSize.width/2;
  
  for( int h = 0; h < imgSize.height; ++h){
    for( int w = 0 ; w < imgSize.width; ++w){
      int y, x;
      /*****************************************/
      /*             CAPTURED IMAGE            */
      /*****************************************/
      y = h + y_margin;
      x = w + x_margin;
      if(  y>= output->height || x>= output->width ) continue;
      int row = h * imgSize.width + w;
      MAT( *cs->y, row, 0 ) = (double)CV_IMAGE_ELEM( output, uchar, y, x) / 256.0;

      
      /*****************************************/
      /*            ORIGINAL IMAGE             */
      /*****************************************/
      for( y = 0; y < psfSize.height; ++y){
	for( x = 0; x < psfSize.width; ++x){
	  int col = y*psfSize.width+x;
	  int py = h + y -psfSize.height/2 + y_margin;
	  int px = w + x -psfSize.width/2 + x_margin;
	  if( py < 0 || py >= input->height || px < 0 || px >= input->width)
	    continue;
	  else
	    MAT( *cs->A, row, col) = (double)CV_IMAGE_ELEM( input, uchar, py, px ) / 256.0;
	}//x
      }//y
    }//w
  }//h

  printf("packing done\n");

}

void packImageToCSStructMat( CvMat* input, CvMat* output, CvSize imgSize, CvSize psfSize, CSstruct *cs)
{
  int N = cs->x->rows;
  int M = cs->y->rows;
  
  int y_margin = input->rows/2 - imgSize.height/2;
  int x_margin = input->cols/2 - imgSize.width/2;
  
  for( int h = 0; h < imgSize.height; ++h){
    for( int w = 0 ; w < imgSize.width; ++w){
      int y, x;
      /*****************************************/
      /*             CAPTURED IMAGE            */
      /*****************************************/
      y = h + y_margin;
      x = w + x_margin;
      if(  y>= output->rows || x>= output->cols ) continue;
      int row = h * imgSize.width + w;
      MAT( *cs->y, row, 0 ) = MAT( *output, y, x);

      
      /*****************************************/
      /*            ORIGINAL IMAGE             */
      /*****************************************/
      for( y = 0; y < psfSize.height; ++y){
	for( x = 0; x < psfSize.width; ++x){
	  int col = y*psfSize.width+x;
	  int py = h + y -psfSize.height/2 + y_margin;
	  int px = w + x -psfSize.width/2 + x_margin;
	  if( py < 0 || py >= input->rows || px < 0 || px >= input->cols)
	    continue;
	  else
	    MAT( *cs->A, row, col) = MAT( *input, py, px );
	}//x
      }//y
    }//w
  }//h

  printf("packing done\n");

}



void packImageToCSStructVarianceAligned( IplImage* input, IplImage* output, CvSize imgSize, CvSize psfSize, CSstruct *cs)
{
  CvMat *meanMat = cvCreateMat( imgSize.height, imgSize.width , CV_64FC1);
  CvMat *varMat  = cvCreateMat( imgSize.height, imgSize.width , CV_64FC1);
  CvMat *selectedRows = cvCreateMat( cs->A->rows, 2 , CV_16SC1);
  
  /*****************************************/
  /*              CALC MEAN                */
  /*****************************************/
  cvSetZero( meanMat );
  for( int h = 0; h < input->height; ++h){
    for( int w = 0; w < input->width; ++w){
      for( int y = 0; y < psfSize.height; ++y){
	for( int x = 0; x < psfSize.width; ++x){
	  int py = h+y-psfSize.height/2;
	  int px = w+x-psfSize.width/2;
	  if(px<0||px>=input->width||py<0||py>=input->height)
	    continue;
	  else
	    MAT(*meanMat, h, w) += (double)CV_IMAGE_ELEM( input, uchar, py, px) / 256.0;
	}// x
      }// y
      MAT( *meanMat, h, w) /= (double)(psfSize.height*psfSize.width);
    }// w
  }// h

  
  /*****************************************/
  /*            CALC VARIANCE              */
  /*****************************************/
  cvSetZero( varMat );
  for( int h = 0; h < input->height; ++h){
    for( int w = 0; w < input->width; ++w){
      for( int y = 0; y < psfSize.height; ++y){
	for( int x = 0; x < psfSize.width; ++x){
	  int py = h+y-psfSize.height/2;
	  int px = w+x-psfSize.width/2;
	  if(px<0||px>=input->width||py<0||py>=input->height)
	    continue;
	  else
	    MAT( *varMat, h, w) +=
	      SQUARE( (double)CV_IMAGE_ELEM(input, uchar, py, px)/256.0
		      - MAT(*meanMat, h, w) );
	}// x
      }// y
      MAT(*varMat, h, w) /= (double)(psfSize.height*psfSize.width);
    }// w
  }// h


  /*****************************************/
  /*              TOP ROWS                 */
  /*****************************************/
  for( int i = 0; i < selectedRows->rows; ++i){ 
    double max;
    CvPoint maxPoint;
    cvMinMaxLoc( varMat, NULL, &max, NULL, &maxPoint, NULL);
    CV_MAT_ELEM( *selectedRows, int, i, 0 ) = maxPoint.x;
    CV_MAT_ELEM( *selectedRows, int, i, 1 ) = maxPoint.y;
    MAT( *varMat, maxPoint.y, maxPoint.x ) = 0.0;
  }


  /*****************************************/
  /*              PACKING                  */
  /*****************************************/
  for( int i = 0; i < selectedRows->rows; ++i){ 
    int w = CV_MAT_ELEM( *selectedRows, int, i, 0);
    int h = CV_MAT_ELEM( *selectedRows, int, i, 1);

    // captured (output)
    MAT( *cs->y, i, 0) = (double)CV_IMAGE_ELEM( output, uchar, h, w) / 256.0;

    // original (input)
    for( int y = 0; y < psfSize.height; ++y){
      for( int x = 0; x < psfSize.width; ++x){
	int py = h + y -psfSize.height/2;
	int px = w + x -psfSize.width /2;
	int col = y*psfSize.width+x;
	if( py<0 || py>= input->height || px < 0 || px >=input->width){
	  MAT(*cs->A, i, col ) = 0.0;
	}else{
	  MAT(*cs->A, i, col ) = (double)CV_IMAGE_ELEM(input, uchar, py, px)/256.0;
	}// else
      }// x
    }// y
  }// i

  cvReleaseMat( &varMat);
  cvReleaseMat( &meanMat);
  cvReleaseMat( &selectedRows);
  return;
}

void unpackCSStrutct2Mat( CSstruct* cs, CvMat *mat )
{
  for( int r = 0; r < mat->rows; ++r){
    for( int c = 0; c < mat->cols; ++c){
      int idx = r*mat->cols+c;
      MAT( *mat, r, c) = MAT( *cs->x, idx, 0);
    }
  }

}




void solveCompressiveSensing( CSstruct *cs )
{
  int N = cs->xSize;
  int M = cs->ySize;


  // temporarly matrix for matrix calculation
  CvMat* tmpM = cvCreateMat( M, 1, CV_64FC1);
  CvMat* tmpN = cvCreateMat( N, 1, CV_64FC1);
  CvMat* tmpNN = cvCreateMat( N, N, CV_64FC1);

  /*************************************************************/
  /*                      MAIN LOOP                            */
  /*************************************************************/
  int count;
  for( count = 0; count < cs->MAX_NT_ITER; ++count){
    
    cvGEMM( cs->A, cs->x, 1.0, cs->y, -1.0, cs->z, 0); // z = Ax-y


    /*************************************************************/
    /*                 CALC DUALITY GAP                          */
    /*************************************************************/
    cvConvertScale( cs->z, cs->nu, 2.0, 0.0 ); // nu = 2z

    cvGEMM( cs->A, cs->nu, 1.0, NULL, 0.0, tmpN, CV_GEMM_A_T); // tmpN = A'nu
    double maxAnu = cvNorm( tmpN, NULL, CV_C, NULL); // maxAnu = |A'nu|inf
    if( maxAnu > cs->lambda )
      cvConvertScale( cs->nu, cs->nu, cs->lambda/maxAnu, 0.0 );

    // pobj = z'z + lambda * |x|1
    cs->pobj = cvDotProduct( cs->z, cs->z ) + cs->lambda * cvNorm( cs->x, NULL, CV_L1, NULL);
    // dobj = max( nu'nu / 4 - nu'y, dobj )
    cs->dobj = max( -0.25* cvDotProduct( cs->nu,cs->nu) - cvDotProduct( cs->nu, cs->y ), cs->dobj);
    cs->gap = cs->pobj - cs->dobj;

    
    /*************************************************************/
    /*                 STOPPING CRITERON                         */
    /*************************************************************/
    printf("pobj = %lf, dobj = %lf, gap = %lf\n", cs->pobj, cs->dobj, cs->gap);
    printf("gap / dobj = %lf, retol = %lf\n", cs->gap/cs->dobj, cs->retol);
    if( cs->gap / cs->dobj < cs->retol ) break;



    /*************************************************************/
    /*                     UPDATE t                              */
    /*************************************************************/
    if( cs->s >= 0.5 )
      cs->t = max(  min( cs->mu*2.0*(double)N/ cs->mu*cs->gap, cs->t ), cs->t );
    printf("t=%lf\n", cs->t);

    
    /*************************************************************/
    /*                     NEWTON STEP                           */
    /*************************************************************/
    for(int i = 0; i < N; ++i){
      MAT( *cs->q1, i, 0 ) = 1.0 / ( MAT( *cs->u, i, 0) + MAT( *cs->x, i, 0) ); // q1 = 1/(u+x)
      MAT( *cs->q2, i, 0 ) = 1.0 / ( MAT( *cs->u, i, 0) - MAT( *cs->x, i, 0) ); // q2 = 1/(u-x)
      MAT( *cs->d1, i, 0 ) = (SQUARE( MAT(*cs->q1, i, 0)) + SQUARE( MAT(*cs->q2, i, 0)) ) / cs->t;
      MAT( *cs->d2, i, 0 ) = (SQUARE( MAT(*cs->q1, i, 0)) - SQUARE( MAT(*cs->q2, i, 0)) ) / cs->t;
    }


    // calculate gradient
    cvGEMM( cs->A, cs->z, 2.0, NULL, 0.0, tmpN, CV_GEMM_A_T); // tmpN = 2*A'z
    for( int i = 0; i < N ; ++i){
      MAT( *cs->gradPhi, i, 0 )  // gradPhi(i) = 2A'z - (q1-q2)/t
	= MAT( *tmpN, i, 0 ) - ( MAT(*cs->q1, i, 0 ) - MAT(*cs->q2, i, 0 ) )/ cs->t;
      MAT( *cs->gradPhi, i+N, 0 )  // gradPhi(i+N) = lambda - (q1+q2) / t
	= cs->lambda - ( MAT( *cs->q1, i, 0 ) + MAT( *cs->q2, i, 0 ) ) / cs->t;
    }

    // vectors to be used in the preconditioner
    for( int i = 0; i < N ; ++i){
      MAT( *cs->prb, i, 0 ) = MAT( *cs->diagxtx, i, 0 ) + MAT( *cs->d1, i, 0);
      MAT( *cs->prs, i, 0 ) = MAT( *cs->prb, i, 0 ) * MAT( *cs->d1, i, 0) - SQUARE( MAT(*cs->d2, i, 0));
    }

    // set PCG relatave torerance
    double normg = cvNorm( cs->gradPhi, NULL, CV_L2, NULL );
    double pcgtol = min( 0.01, cs->eta* cs->gap / min( 1.0, normg ));

    printf("run PCG ... pcgtol = %lf\n", pcgtol);
    // run PCG
    cvConvertScale( cs->gradPhi, cs->gradPhi, -1.0, 0.0);
    PCG_MatMulOperator( mulA, cs->gradPhi, cs->dxu, mulPinv, pcgtol, cs );
    cvConvertScale( cs->gradPhi, cs->gradPhi, -1.0, 0.0);


    // convert results
    for(int i = 0; i < N; ++i){
      MAT( *cs->dx, i, 0 ) = MAT( *cs->dxu, i, 0);
      MAT( *cs->du, i, 0 ) = MAT( *cs->dxu, i+N, 0);
    }


    /*************************************************************/
    /*                 BACKTRACK LINE SEARCH                     */
    /*************************************************************/
    double phi = cvDotProduct( cs->z, cs->z ) + cs->lambda * cvSum( cs->u ).val[0];
    for(int i = 0; i < cs->f->rows; ++i)
      phi -= log( -MAT( *cs->f, i, 0 ) ) / cs->t;
    // phi = z'z + lambda*sum(u) - sum( log(-f) ) / t

    cs->s = 1.0;
    double gdx = cvDotProduct( cs->gradPhi, cs->dxu );
    int k;
    double newPhi ;
    for( k = 0; k < cs->MAX_LS_ITER; ++k){
      cvScaleAdd( cs->dx, cvScalarAll(cs->s), cs->x, cs->newX); // newX = x + s*dx
      cvScaleAdd( cs->du, cvScalarAll(cs->s), cs->u, cs->newU); // newU = u + s*du

      int NonNegativeFlag = 0;
      for(int i = 0; i < N; ++i){
	MAT( *cs->newF, i, 0 ) = MAT( *cs->newX, i, 0 ) - MAT( *cs->newU, i, 0 );
	MAT( *cs->newF, i+N, 0 ) = -MAT( *cs->newX, i, 0 ) - MAT( *cs->newU, i, 0 );

	if( MAT( *cs->newF, i, 0 ) >0 ||MAT( *cs->newF, i+N, 0 ) >0 )
	  NonNegativeFlag = 1;
      }

      if( NonNegativeFlag == 0 ){
	cvGEMM( cs->A, cs->newX, 1.0, cs->y, -1.0, cs->newZ, 0 );
	newPhi
	  = cvDotProduct( cs->newZ, cs->newZ ) + cs->lambda * cvSum( cs->newU ).val[0] ;
	for(int i = 0; i < N; ++i)
	  newPhi -= log( -MAT( *cs->newF, i, 0 ) ) / cs->t;

	if( newPhi - phi <= cs->alpha * cs->s * gdx )
	  break;
      }

      cs->s *= cs->beta;
    }//k 
    printf("k = %d newphi = %lf\n", k, newPhi);
    cvConvert( cs->newX, cs->x);
    cvConvert( cs->newU, cs->u);
    cvConvert( cs->newF, cs->f);

    if( k == cs->MAX_LS_ITER ) break;

  }// main loop

  cvReleaseMat( &tmpN );
  cvReleaseMat( &tmpM );
  cvReleaseMat( &tmpNN );
  
  return;

}



void mulA( CSstruct *cs, CvMat* x, CvMat* y){
  /*****************************************
   y = hes_phi * x
   where hes_phi 
   = [ A'*A * 2 + d1 , d2]
     [       d2      , d1]
   ****************************************/
  int N = x->rows/2;

  // x = [x1; x2]
  for( int i = 0; i < N; ++i){
    MAT( *cs->x1, i, 0 ) = MAT( *x, i, 0);
    MAT( *cs->x2, i, 0 ) = MAT( *x, i+N, 0);
  }
  
  // y = [y1; y2]
  cvSetZero(cs->y1); cvSetZero(cs->y2);
  
  CvMat* tmp = cvCreateMat( cs->A->rows, 1, CV_64FC1);

  // y1 = 2A' (A*x1) + d1.*x1 + d2.*x1
  // y2 = d2.*x1 + d1.*x2
  cvMatMul( cs->A, cs->x1, tmp ); // tmp = A*x1
  cvGEMM(cs->A, tmp, 2.0 , NULL, 0.0, cs->y1, CV_GEMM_A_T); // y1 = 2 * A'tmp = 2 * A'(A*x1)
  for( int i = 0; i < N; ++i){
    MAT( *cs->y1, i, 0) += MAT(*cs->d1, i, 0)*MAT(*cs->x1, i, 0) + MAT(*cs->d2, i, 0)*MAT(*cs->x2, i, 0);
    MAT( *cs->y2, i, 0) += MAT(*cs->d2, i, 0)*MAT(*cs->x1, i, 0) + MAT(*cs->d1, i, 0)*MAT(*cs->x2, i, 0);
  }

  for(int i = 0; i < N; ++i){
    MAT( *y, i, 0 ) = MAT( *cs->y1, i, 0 );
    MAT( *y, i+N, 0 ) = MAT( *cs->y2, i, 0 );
  }

  cvReleaseMat( &tmp );

  return;

}


void mulPinv( CSstruct *cs, CvMat* x, CvMat* y)
{
  /****************************************
   y = Pinv * x
   where Pinv = [ p1 , -p2]
                [-p2 ,  p3]
     where p1(i) = d1(i) / prs(i)
           p2(i) = d2(i) / prs(i)
           p3(i) = prb(i) / prs(i)
   ****************************************/
  int N = x->rows/2;
  for(int i = 0; i < N; ++i){
    double p1 = MAT( *cs->d1, i, 0 ) / MAT( *cs->prs, i, 0 );
    double p2 = MAT( *cs->d2, i, 0 ) / MAT( *cs->prs, i, 0 );
    double p3 = MAT( *cs->prb, i, 0 ) / MAT( *cs->prs, i, 0 );

    MAT( *y, i, 0 )   =  p1 * MAT( *x, i, 0 ) - p2 * MAT( *x, i+N, 0);
    MAT( *y, i+N, 0 ) = -p2 * MAT( *x, i, 0 ) + p3 * MAT( *x, i+N, 0);
  }
}
