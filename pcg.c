#include "pcg.h"

#define PCG_MAX 100

double dotProduct( CvMat* src1, CvMat* src2 );
void matMul( CvMat* src1, CvMat* src2, CvMat* dst ); // src1 * src2 = dst


void PCG( CvMat* A, CvMat* b, CvMat* x, CvMat* Pinv, double th)
{
  // size check

  if(A == NULL || b == NULL || x == NULL || Pinv == NULL || 
     A->cols != x->rows || A->rows != b->rows || b->cols != 1 || 
     x->cols != 1 || A->cols != Pinv->cols || A->rows != Pinv->rows ) {
    printf("size error in ");
    return ;
  }

  int dim = x->rows;

  // initalize
  int k = 0, i;
  cvSetZero(x);
  CvMat* r = cvCloneMat(b);
  CvMat* p = cvCreateMat( dim, 1, CV_64FC1 );
  cvMatMul( Pinv, b, p );
  CvMat* y = cvCreateMat( dim, 1, CV_64FC1 );
  cvMatMul( Pinv, r, y );
  CvMat* z = cvCreateMat( dim, 1, CV_64FC1 );

  // iteration
  while(1){
    k++;
    cvMatMul( A, p, z );
    double prevDot = cvDotProduct( y, r);
    double v = prevDot / cvDotProduct( p, z);
    
    for(i = 0; i < dim; ++i){
      CV_MAT_ELEM( *x, double, i, 0 ) += v * CV_MAT_ELEM( *p, double, i, 0 );
      CV_MAT_ELEM( *r, double, i, 0 ) -= v * CV_MAT_ELEM( *z, double, i, 0 );
    }
    
    cvMatMul( Pinv, r, y );
    double mu = cvDotProduct(y, r) / prevDot;

    if( cvNorm( r, NULL, CV_L2, NULL ) < th ) break;
    if( k > 100 ) break;
    else{
      
      for(i = 0; i < dim; ++i){
	CV_MAT_ELEM( *p, double, i, 0 ) 
	  = CV_MAT_ELEM( *y, double, i, 0) + mu * CV_MAT_ELEM( *p, double, i, 0);
      }
    }
  }

  // clean up
  cvReleaseMat( &r );
  cvReleaseMat( &p );
  cvReleaseMat( &y );
  cvReleaseMat( &z );

  return;

}

void PCG_MatMulOperator( void(*mulA)( CSstruct* cs, CvMat* u, CvMat* y), 
			 CvMat* b, CvMat* x, 
			 void(*mulPinv)( CSstruct* cs, CvMat* u, CvMat* y), 
			 double th, CSstruct* cs)
{
  int dim = x->rows;

  // initlaize
  int k = 0;
  int i;
  //cvSetZero(x);
  CvMat* r = cvCloneMat(b);
  CvMat* p = cvCreateMat( dim, 1, CV_64FC1);
  mulPinv( cs, b, p); // <-- changed Pinv
  CvMat* y = cvCreateMat( dim, 1, CV_64FC1);
  mulPinv( cs, r, y); // <-- changed Pinv
  CvMat* z = cvCreateMat( dim, 1, CV_64FC1);

  //iteration
  while(1){
    k++;
    mulA( cs, p, z ); // <-- changed mulA
    double prevDot = cvDotProduct( y, r);
    double v = prevDot / cvDotProduct( p, z);
    
    for( i = 0; i < dim; ++i){
      MAT( *x, i, 0 ) += v * MAT( *p, i, 0);
      MAT( *r, i, 0 ) -= v * MAT( *z, i, 0);
    }
    
    mulPinv( cs, r, y);  // <-- changed mulPinv
    double mu = cvDotProduct( y, r) / prevDot;
    double norm = cvNorm( r, NULL, CV_L2, NULL );
    if( norm < th ) break;
    if( k > dim ) break;
    else{
      for(i=0;i<dim;++i)
	MAT( *p, i, 0) = MAT( *y, i, 0 ) + mu * MAT( *p, i, 0 );
    }//else

    printf("iteration %d\t", k);
    printf("err = %lf\n", norm);


  }// while

  cvReleaseMat( &r );
  cvReleaseMat( &p );
  cvReleaseMat( &y );
  cvReleaseMat( &z );

  return;
}
