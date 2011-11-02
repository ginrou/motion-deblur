#include "compressiveSensing.h"

double convexFunc( const CSstruct* cs , CvMat* x, CvMat* u);
void packForPCG( CSstruct* cs );
void saveMat( CvMat *mat, char filename[] );

IplImage *solveCPImage( IplImage* src, IplImage *filter)
{
  return NULL;
}


CSstruct* createCPStructure( IplImage* input, IplImage* output , int filterSize, CvSize imgSize)
{
  CSstruct* cs = (CSstruct*)malloc(sizeof(CSstruct));
  cs->xSize = filterSize; 
  cs->ySize = imgSize.width * imgSize.height;
  
  // input and output data
  cs->A = cvCreateMat( cs->ySize, cs->xSize, CV_64FC1 );
  cs->x = cvCreateMat( cs->xSize, 1, CV_64FC1 );
  cs->y = cvCreateMat( cs->ySize,  1, CV_64FC1 );

  // computing data
  cs->u = cvCreateMat( cs->xSize, 1, CV_64FC1);
  cs->dx = cvCreateMat( cs->xSize, 1, CV_64FC1);
  cs->du = cvCreateMat( cs->xSize, 1, CV_64FC1);
  cs->nu = cvCreateMat( cs->ySize, 1, CV_64FC1);

  // PCG variables
  for(int i = 0; i< 4; ++i) 
    cs->subMat[i] = cvCreateMat( cs->xSize, cs->xSize , CV_64FC1);
  cs->P = cvCreateMat( cs->xSize * 2, cs->xSize * 2 , CV_64FC1);
  cs->b = cvCreateMat( cs->xSize * 2, 1, CV_64FC1);
  cs->var = cvCreateMat( cs->xSize * 2, 1, CV_64FC1);
  cs->Pinv = cvCreateMat( cs->xSize * 2, cs->xSize * 2 , CV_64FC1);

  // buffer fo convex function
  cs->buf = cvCreateMat( cs->ySize, 1, CV_64FC1 );
  cs->bufX = cvCreateMat( cs->xSize, 1, CV_64FC1 );
  cs->tmpX = cvCreateMat( cs->xSize, 1, CV_64FC1 );
  cs->tmpU = cvCreateMat( cs->xSize, 1, CV_64FC1 );

  //parameters settings
  cs->lambda = 0.01;
  cs->epRel = 0.05;
  cs->alpha = 0.01;
  cs->beta = 0.5; 
  cs->mu = 2.0;
  cs->sMin = 0.5;
  cs->nabla = 0.01;

  return cs;
}



void solveCompressiveSensing( CSstruct *cs )
{
  int N = cs->xSize;
  int M = cs->ySize;

  // initialize
  cs->t = 1.0 / cs->lambda;
  cvSetZero( cs->x );
  cvSet( cs->u, cvScalarAll(1.0), NULL );
  int count = 0;

  // iteration
  while(1){

    // compute search direction
    packForPCG( cs ); // packing
    PCG( cs->P, cs->b, cs->var, cs->Pinv, 0.1 ); // computing
    for(int i = 0; i < N; ++i){ // moving to dx, du
      MAT( *cs->dx, i, 0 ) = MAT( *cs->var, i, 0);
      MAT( *cs->du, i, 0 ) = MAT( *cs->var, i+N, 0);
    }

    // back tracking line search
    // search minimum integer K
    int K = 0;
    double fxu = convexFunc( cs, cs->x, cs->u );
    double dxu = -cvDotProduct( cs->b, cs->var );
    while(1){
      double bk = pow( cs->beta, (double)K );
      cvScaleAdd( cs->dx, cvScalarAll(bk), cs->x, cs->tmpX );
      cvScaleAdd( cs->du, cvScalarAll(bk), cs->u, cs->tmpU );
      double newFxu = convexFunc( cs, cs->tmpX, cs->tmpU);
      double FxuLine = fxu + cs->alpha * bk * dxu;
      if( newFxu <= FxuLine ) break;
      K++;
    }

    // update x and u :  x <- x + B^k dx
    cvScaleAdd( cs->dx, cvScalarAll( pow( cs->beta, K )), cs->x, cs->x );
    cvScaleAdd( cs->du, cvScalarAll( pow( cs->beta, K )), cs->u, cs->u );

    // dual feasible point
    double s = 1.0;
    if( count != 0  ){ // scaling parameter
      cvGEMM( cs->A, cs->nu, 1.0, NULL, 0.0, cs->bufX, CV_GEMM_A_T ); // buf = A'v
      double max = cvNorm( cs->bufX, NULL, CV_C, NULL);
      if( max < 1.0 ) s = max;
    }

    cvMatMul( cs->A, cs->x, cs->buf); // buf = Ax
    cvSub( cs->buf, cs->y, cs->buf, NULL);// buf = Ax - y 
    cvConvertScale( cs->buf, cs->nu, 2.0*s, 0.0); // nu = 2s * buf 

    // dualitry gap
    double g = - cvDotProduct( cs->nu, cs->nu) / 4.0 - cvDotProduct( cs->nu, cs->y );
    cs->eta = cvDotProduct( cs->buf, cs->buf ) - g ;
    for(int i = 0; i < N; ++i)
      cs->eta += cs->lambda * fabs( MAT( *cs->x, i, 0 ));

    // quit
    printf("cs->eta / g = %lf / %lf = %lf\n", cs->eta, g, cs->eta/g);
    if(  cs->eta/g  <= cs->epRel ) break;

    // update t
    if( pow( cs->beta, K ) >= cs->sMin ){
      double nextT = 2.0 * (double)N / cs->eta;
      if( cs->t < nextT ) nextT = cs->t;
      cs->t *= cs->mu;
      if( cs->t < nextT) cs->t = nextT;
    }
    printf("step %d, K = %d, s = %lf, t = %lf\n", count, K, s, cs->t);    
    count++;
  }


}

double convexFunc( const CSstruct* cs , CvMat* x, CvMat* u)
{
  cvMatMul( cs->A, x, cs->buf );
  cvSub( cs->buf, cs->y, cs->buf, NULL);
  double ret = cs->t * cvDotProduct( cs->buf, cs->buf );
  for(int i = 0; i < cs->xSize; ++i){
    ret += cs->t * cs->lambda * MAT( *u, i, 0 );
    ret -= log( MAT( *u, i, 0) + MAT( *x, i, 0));
    ret -= log( MAT( *u, i, 0) - MAT( *x, i, 0));
  }
  return ret;
}

void packForPCG( CSstruct* cs )
{
  int N = cs->xSize;
  int M = cs->ySize;

  // b 
  for(int j = 0; j < N; ++j){

    // dx
    double dxj = 0.0;
    for( int k = 0; k < N; ++k){
      double sum = 0.0;
      for(int i = 0; i < N; ++i)
	sum += MAT( *cs->A, i, j ) * MAT( *cs->A, i, k );
      dxj += MAT( *cs->x, k, 0) * sum - MAT( *cs->A, k, j ) * MAT( *cs->y, k, 0 );
    }
    dxj *= 2.0*cs->t;
    dxj -= 1.0 / ( MAT( *cs->u, j, 0 ) + MAT( *cs->x, j, 0));
    dxj += 1.0 / ( MAT( *cs->u, j, 0 ) - MAT( *cs->x, j, 0));
    MAT( *cs->b, j, 0 ) = -dxj;

    // du
    double duj = 0.0;
    duj = cs->t * cs->lambda * MAT( *cs->u, j, 0 );
    duj -= 1.0 / ( MAT( *cs->u, j, 0 ) + MAT( *cs->x, j, 0));
    duj -= 1.0 / ( MAT( *cs->u, j, 0 ) - MAT( *cs->x, j, 0));
    MAT( *cs->b, j + N, 0 ) = -duj;

  }    

  // sub mat[0] for dxdx
  for( int j = 0; j < N; ++j){
    for( int l = 0; l < N; ++l){
      double dxjdxl = 0.0;
      for(int i = 0; i < N; ++i)
	dxjdxl += MAT( *cs->A, i, j ) * MAT( *cs->A, i, l );

      if( j == l ){
	dxjdxl += 1.0/ SQUARE( MAT( *cs->u, j, 0) + MAT( *cs->x, j, 0 ) );
	dxjdxl += 1.0/ SQUARE( MAT( *cs->u, j, 0) - MAT( *cs->x, j, 0 ) );
      }
      MAT( *cs->subMat[0] , j, l ) = dxjdxl;
    }
  }

  // sub mat[1] for dxdu
  cvSetZero( cs->subMat[1] );
  for( int i = 0; i < N; ++i){
    double tmp = 0.0;
    tmp += 1.0/ SQUARE( MAT( *cs->u, i, 0) + MAT( *cs->x, i, 0 ) );
    tmp -= 1.0/ SQUARE( MAT( *cs->u, i, 0) - MAT( *cs->x, i, 0 ) );
    MAT( *cs->subMat[1], i, i ) = tmp ;
  }

  // sub mat[2] for dudx
  cvSetZero( cs->subMat[2] );
  for( int i = 0; i < N; ++i){
    double tmp = 0.0;
    tmp += 1.0/ SQUARE( MAT( *cs->u, i, 0) + MAT( *cs->x, i, 0 ) );
    tmp -= 1.0/ SQUARE( MAT( *cs->u, i, 0) - MAT( *cs->x, i, 0 ) );
    MAT( *cs->subMat[2], i, i ) = tmp ;
  }

  // sub mat[3] for dudu
  cvSetZero( cs->subMat[3] );
  for( int i = 0; i < N; ++i){
    double tmp = cs->t * cs->lambda;
    tmp += 1.0/ SQUARE( MAT( *cs->u, i, 0) + MAT( *cs->x, i, 0 ) );
    tmp += 1.0/ SQUARE( MAT( *cs->u, i, 0) - MAT( *cs->x, i, 0 ) );
    MAT( *cs->subMat[3], i, i ) = tmp ;
  }
  
  // P
  cvSetZero( cs->P );
  for(int r = 0; r < N; ++r){ // sub mat[0]
    for(int c = 0; c < N; ++c){
      MAT( *cs->P, r, c) = MAT( *cs->subMat[0], r, c);
    }}
  for( int i = 0; i < N; ++i) // sub mat[1]
    MAT( *cs->P, i, i+N ) = MAT( *cs->subMat[1], i, i);
  for( int i = 0; i < N; ++i) // sub mat[2]
    MAT( *cs->P, i+N, i) = MAT( *cs->subMat[2], i, i);
  for( int i = 0; i < N; ++i) // sub mat[3]
    MAT( *cs->P, i+N, i+N ) = MAT( *cs->subMat[3], i, i);

  // Pinv
  cvSetZero( cs->Pinv );
  for(int i = 0; i < N ; ++i){
    double det = MAT(*cs->subMat[0],i,i) * MAT(*cs->subMat[3],i,i) - MAT(*cs->subMat[1],i,i) * MAT(*cs->subMat[2],i,i);
    MAT( *cs->Pinv, i  , i   ) =  MAT( *cs->subMat[4], i, i ) / det ;
    MAT( *cs->Pinv, i  , i+N ) = -MAT( *cs->subMat[1], i, i ) / det ;
    MAT( *cs->Pinv, i+N, i   ) = -MAT( *cs->subMat[2], i, i ) / det ;
    MAT( *cs->Pinv, i+N, i+N ) =  MAT( *cs->subMat[0], i, i ) / det ;
  }
  
}


void saveMat( CvMat *mat, char filename[] )
{
  FILE *fp = fopen( filename, "w" );
  for( int r = 0; r < mat->rows; ++r){
    for( int c = 0; c < mat->cols; ++c){
      fprintf( fp, "%3.7f,  ", MAT( *mat, r, c));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}
