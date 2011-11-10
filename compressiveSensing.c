#include "compressiveSensing.h"

double convexFunc( const CSstruct* cs , CvMat* x, CvMat* u);
void packForPCG( CSstruct* cs );
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


CSstruct* createCPStructure( IplImage* input, IplImage* output , int filterSize, CvSize imgSize)
{
  CSstruct* cs = (CSstruct*)malloc(sizeof(CSstruct));

  cs->xSize = filterSize;
  cs->ySize = imgSize.height * imgSize.width;
 
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
  cs->retol = pow( 10.0, -3 );
  cs->lambda = 0.01;

  
  // initial values
  cs->eta = pow( 10.0, -3 );
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
    // dobj = max( nu'nu / 4 - nu'y, pobj )
    cs->dobj = max( -0.25* cvDotProduct( cs->nu,cs->nu) - cvDotProduct( cs->nu, cs->y ), cs->dobj);
    cs->gap = cs->pobj - cs->dobj;

    
    /*************************************************************/
    /*                 STOPPING CRITERON                         */
    /*************************************************************/
    printf("gap / dobj = %lf, retol = %lf\n", cs->gap/cs->dobj, cs->retol);
    if( cs->gap / cs->dobj < cs->retol ) break;



    /*************************************************************/
    /*                     UPDATE t                              */
    /*************************************************************/
    if( cs->s >= 0.5 )
      cs->t = max( min( 2*N*cs->mu/ cs->gap, cs->mu*cs->t ), cs->t );
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
    double pcgtol = min( 0.1, cs->eta* cs->gap / min( 1.0, normg ));

    // calculate P
    // p = [ A'A*2 + d1 , d2 ]
    //     [     d2     , d1 ]
    cvSetZero( cs->P );



    //cvMulTransposed( cs->A, tmpNN, 0, NULL, 1.0 );
    for( int r = 0; r < N; ++r){
      for( int c = 0; c < N; ++c){
	MAT( *cs->P, r, c) = 0.0;
	for(int i=0; i < M; ++i)
	  MAT( *cs->P, r, c) += 2.0*MAT(*cs->A, i, r ) * MAT( *cs->A, i, c );
      }
      printf("r=%d done\n", r);
    }


    for( int i = 0; i < N ; ++i){
      MAT( *cs->P, i  , i   ) += MAT( *cs->d1, i, 0);
      MAT( *cs->P, i  , i+N ) = MAT( *cs->d2, i, 0);
      MAT( *cs->P, i+N, i   ) = MAT( *cs->d2, i, 0);
      MAT( *cs->P, i+N, i+N ) = MAT( *cs->d1, i, 0);
    }

    // calculate Preconditoner ( Pinv )
    // Pinv = [ P1, -P2]
    //        [-P2,  P3]
    // P1 = d1 / prs
    // P2 = d2 / prs
    // P3 = prb/ prs
    cvSetZero( cs->Pinv );
    for( int i = 0; i < N; ++i){
      MAT( *cs->Pinv, i  , i   ) =  MAT( *cs->d1, i, 0 ) / MAT( *cs->prs, i, 0 );
      MAT( *cs->Pinv, i  , i+N ) = -MAT( *cs->d2, i, 0 ) / MAT( *cs->prs, i, 0 );
      MAT( *cs->Pinv, i+N, i   ) = -MAT( *cs->d1, i, 0 ) / MAT( *cs->prs, i, 0 );
      MAT( *cs->Pinv, i+N, i+N ) =  MAT( *cs->prb, i, 0 ) / MAT( *cs->prs, i, 0 );
    }
    // run PCG
    cvConvertScale( cs->gradPhi, cs->gradPhi, -1.0, 0.0);
    PCG( cs->P, cs->gradPhi, cs->dxu, cs->Pinv, pcgtol );
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

    for( int k = 0; k < cs->MAX_LS_ITER; ++k){
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
	double newPhi 
	  = cvDotProduct( cs->newZ, cs->newZ ) + cs->lambda * cvSum( cs->newU ).val[0] ;
	for(int i = 0; i < N; ++i)
	  newPhi -= log( -MAT( *cs->newF, i, 0 ) ) / cs->t;

	if( newPhi - phi <= cs->alpha * cs->s * gdx )
	  break;
      }

      cs->s *= cs->beta;
    }//k 

    cvConvert( cs->newX, cs->x);
    cvConvert( cs->newU, cs->u);

  }// main loop

  cvReleaseMat( &tmpN );
  cvReleaseMat( &tmpM );
  cvReleaseMat( &tmpNN );
  
  return;

}

double convexFunc( const CSstruct* cs , CvMat* x, CvMat* u)
{

}

void packForPCG( CSstruct* cs )
{

}
