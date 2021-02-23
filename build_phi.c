/*==========================================================
 *
 * construct Phi and predPhi^T
 *
 * Compile as below
 *
 * UNIX
 gcc -Wall -shared -O3 -fopenmp -o build_phi.so -fPIC build_phi.c 
 *
 *========================================================*/

#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "omp.h"

/* The computational routine */
void build_phi(int n_obs, int n_pred, int n_m, double *basnrs,
        double *obs, double *X, double *Y, double Lx, double Ly, double A, double B,
        int *nrSegments, int *addPrevSegments, double *Phi, double *predPhi_T)
{
    
    int ss, tt, zz, qq, outerInd;
    
    if (n_obs>n_pred) {
        outerInd=n_obs;
    }else{
        outerInd=n_pred;
    }
    
    omp_set_num_threads(omp_get_max_threads()); /* automatically use the maximum number of available threads */
    #pragma omp parallel for private(ss,tt,qq,zz)
    for (ss=0; ss<outerInd; ss++) {
        if (ss<n_obs) {
            int compSegs= addPrevSegments[ss];
            double L1 = sqrt(  (obs[1+4*(ss+compSegs)]-obs[4*(ss+compSegs)])*(obs[1+4*(ss+compSegs)]-obs[4*(ss+compSegs)]) +
                    (obs[3+4*(ss+compSegs)]-obs[2+4*(ss+compSegs)])*(obs[3+4*(ss+compSegs)]-obs[2+4*(ss+compSegs)]) );
            double x01 = obs[4*(ss+compSegs)], y01=obs[2+4*(ss+compSegs)],
                    nx=(obs[1+4*(ss+compSegs)]-obs[4*(ss+compSegs)])/L1,   /* (x1-x0)/L */
                    ny=(obs[3+4*(ss+compSegs)]-obs[2+4*(ss+compSegs)])/L1; /* (y1-y0)/L */
            double Ltot=0;
            
            for (zz=0; zz<nrSegments[ss]; zz++) {
                /* calculate L=sqrt( (x1-x0)^2 + (y1-y0)^2 ); */
                double L = sqrt(  (obs[1+4*(ss+compSegs+zz)]-obs[4*(ss+compSegs+zz)])*(obs[1+4*(ss+compSegs+zz)]-obs[4*(ss+compSegs+zz)]) +
                        (obs[3+4*(ss+compSegs+zz)]-obs[2+4*(ss+compSegs+zz)])*(obs[3+4*(ss+compSegs+zz)]-obs[2+4*(ss+compSegs+zz)]) );
                double x0 = obs[4*(ss+compSegs+zz)], y0=obs[2+4*(ss+compSegs+zz)];
                double Lstart=sqrt( (x0-x01)*(x0-x01) + (y0-y01)*(y0-y01) ), Lend=Lstart+L;
                Ltot = Ltot + Lend-Lstart;
                
                for (tt=0; tt<n_m; tt++) {
                    if (zz==0)
                        Phi[ss+n_obs*tt]=0;
                    double lambdaXin=0.5*M_PI*basnrs[tt]/Lx, lambdaYin=0.5*M_PI*basnrs[tt+n_m]/Ly;
                    double lambdaX=nx*lambdaXin, lambdaY=ny*lambdaYin,
                            BX=(x01+Lx)*lambdaXin, BY=(y01+Ly)*lambdaYin;
                    double lambda_min=lambdaX-lambdaY, B_min=BX-BY,
                            lambda_plus=lambdaX+lambdaY, B_plus=BX+BY;
                    double superConst=nx*nx*(lambdaXin*lambdaXin-A*lambdaYin*lambdaYin)+
                            ny*ny*(lambdaYin*lambdaYin-A*lambdaXin*lambdaXin),
                            otherConst=2*B*nx*ny*lambdaXin*lambdaYin;
                    double int1=0, int2=0;
                    if (abs(lambda_min)>1e-10)
                        int1 = ( superConst + otherConst ) * ( sin(lambda_min*Lend+B_min) - sin(lambda_min*Lstart+B_min) ) / lambda_min;
                    else
                        int1 = ( superConst + otherConst ) * (Lend-Lstart) * cos(B_min);
                    if (abs(lambda_plus)>1e-10)
                        int2 = ( superConst - otherConst ) * ( sin(lambda_plus*Lstart+B_plus) -sin(lambda_plus*Lend+B_plus) ) / lambda_plus;
                    else
                        int2 = ( superConst - otherConst ) * (Lend-Lstart) * cos(B_plus);    
                    double theInt = int1 + int2;
                    Phi[ss+n_obs*tt]+=theInt;
                    if (zz==nrSegments[ss]-1)
                        Phi[ss+n_obs*tt]/=2*Ltot*(sqrt(Ly*Lx));
                }
            }
        }
        
        if (ss<n_pred) {
            for (qq=0; qq<n_m; qq++) {
                double lambdaXin=0.5*M_PI*basnrs[qq]/Lx, lambdaYin=0.5*M_PI*basnrs[qq+n_m]/Ly;
                double dx2=-lambdaXin*lambdaXin*(1/sqrt(Ly*Lx))*sin(lambdaXin*(X[ss]+Lx))*
                        sin(lambdaYin*(Y[ss]+Ly));
                double dy2=-lambdaYin*lambdaYin*(1/sqrt(Ly*Lx))*sin(lambdaXin*(X[ss]+Lx))*
                        sin(lambdaYin*(Y[ss]+Ly));
                double dxdy=lambdaYin*lambdaXin*(1/sqrt(Ly*Lx))*cos(lambdaXin*(X[ss]+Lx))*
                        cos(lambdaYin*(Y[ss]+Ly));
                predPhi_T[3*ss+3*n_pred*qq] = A*dy2-dx2;
                predPhi_T[3*ss+1+3*n_pred*qq] = B*dxdy;
                predPhi_T[3*ss+2+3*n_pred*qq] = A*dx2-dy2;
            }
        }
        
    }
    
}
