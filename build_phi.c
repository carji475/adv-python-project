/*==========================================================
 *
 * build Phi and predPhi^T
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
void build_phi(int n_obs, int n_pred, int n_m, int *basnrs,
        double *obs, double *pred, double Lx, double Ly, double A, double B,
        int *nrSegments, int *addPrevSegments, double *Phi, double *predPhi_T)
{

    int ss, tt, zz, qq, outerInd, nobscols=addPrevSegments[n_obs-1]+n_obs;

    if (n_obs>n_pred) {
        outerInd=n_obs;
    }else{
        outerInd=n_pred;
    }

//    omp_set_num_threads(omp_get_max_threads()); /* automatically use the maximum number of available threads */
//    #pragma omp parallel for private(ss,tt,qq,zz)
    for (ss=0; ss<outerInd; ss++) {
        if (ss<n_obs) {
            int compSegs= addPrevSegments[ss];
            double L1 = sqrt(  (obs[ss+compSegs+nobscols]-obs[ss+compSegs])*(obs[ss+compSegs+nobscols]-obs[ss+compSegs]) +
                    (obs[ss+compSegs+3*nobscols]-obs[ss+compSegs+2*nobscols])*(obs[ss+compSegs+3*nobscols]-obs[ss+compSegs+2*nobscols]) );
            double x01 = obs[ss+compSegs], y01=obs[ss+compSegs+2*nobscols],
                    nx=(obs[ss+compSegs+nobscols]-obs[ss+compSegs])/L1,   /* (x1-x0)/L */
                    ny=(obs[ss+compSegs+3*nobscols]-obs[ss+compSegs+2*nobscols])/L1; /* (y1-y0)/L */
            double Ltot=0;

            for (zz=0; zz<nrSegments[ss]; zz++) {
                /* calculate L=sqrt( (x1-x0)^2 + (y1-y0)^2 ); */
                double L = sqrt(  (obs[ss+compSegs+nobscols+zz]-obs[ss+compSegs+zz])*(obs[ss+compSegs+nobscols+zz]-obs[ss+compSegs+zz]) +
                        (obs[ss+compSegs+3*nobscols+zz]-obs[ss+compSegs+2*nobscols+zz])*(obs[ss+compSegs+3*nobscols+zz]-obs[ss+compSegs+2*nobscols+zz]) );
                double x0 = obs[ss+compSegs+zz], y0=obs[ss+compSegs+2*nobscols+zz];
                double Lstart=sqrt( (x0-x01)*(x0-x01) + (y0-y01)*(y0-y01) ), Lend=Lstart+L;
                Ltot = Ltot + Lend-Lstart;

                for (tt=0; tt<n_m; tt++) {
                    if (zz==0)
                        Phi[n_m*ss+tt]=0;
                    double lambdaXin=0.5*M_PI*basnrs[2*tt]/Lx, lambdaYin=0.5*M_PI*basnrs[2*tt+1]/Ly;
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
                    Phi[n_m*ss+tt]+=theInt;
                    if (zz==nrSegments[ss]-1)
                        Phi[n_m*ss+tt]/=2*Ltot*(sqrt(Ly*Lx));
                }
            }
        }

        if (ss<n_pred) {
            for (qq=0; qq<n_m; qq++) {
                double lambdaXin=0.5*M_PI*basnrs[2*qq]/Lx, lambdaYin=0.5*M_PI*basnrs[2*qq+1]/Ly;
                double dx2 = -lambdaXin*lambdaXin*(1/sqrt(Ly*Lx))*sin(lambdaXin*(pred[2*ss]+Lx))*sin(lambdaYin*(pred[2*ss+1]+Ly));
                double dy2 = -lambdaYin*lambdaYin*(1/sqrt(Ly*Lx))*sin(lambdaXin*(pred[2*ss]+Lx))*sin(lambdaYin*(pred[2*ss+1]+Ly));
                double dxdy = lambdaYin*lambdaXin*(1/sqrt(Ly*Lx))*cos(lambdaXin*(pred[2*ss]+Lx))*cos(lambdaYin*(pred[2*ss+1]+Ly));
                predPhi_T[3*n_m*ss+qq] = A*dy2-dx2;
                predPhi_T[3*n_m*ss+n_m+qq] = B*dxdy;
                predPhi_T[3*n_m*ss+2*n_m+qq] = A*dx2-dy2;
            }
        }

    }
    
}
