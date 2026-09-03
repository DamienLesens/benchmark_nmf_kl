/* Scalar Newton method for solving KL NMF problem 
 * Modified by Damien Lesens in 2026 for python integration: deleted the matlab related code
 * written by LTK Hien, July 2020
 * Run mex SN.cpp in Matlab
 * Usage: [Wout, Hout, e ,t]=SN(X,r,max_iter,max_time,W',H,inneriter,delta,obj_compute);
 * (W,H) is initial point
 * inneriter: number of iterations of inner loop to update a column of H or a row of W
 * if obj_compute = 1 (default value) then the output e would be the sequence of the objective values
 * otherwise, the output e would be 0.
 * the output t is the sequence of taken time.
*/
#include "math.h"
// #include "mex.h" 
#include <time.h>
#include <algorithm> 
#include <iostream> 



double objective(int m, int n, double *X, double *WH)
// this function is to calculate the KL objective function, size(X)=[m,n] 
{  
	double obj = 0, temp;
	for ( int i=0 ; i<m*n ; i++ )
    {   temp=(X[i]+2e-16)/(WH[i]+2e-16);
        if (temp<2e-16)  //numerical error
            temp=2e-16;
         obj = obj + X[i]*log(temp)-X[i]+WH[i];
    }
        
	return (obj);
}

void update_colH(int m, int r, double sc_coeff, double *Xcol, double *Hcol, double *WHcol, double *Wt, int inneriter, double delta)
{   
    double grad, hessian, temp, vold, vnew, diff,Wttemp, d, s;
    double lambda, eps0, eps1, WHcol_temp, vdiff;
    int id_i, im, idH;
      	char matlab_output[1024];

    for (int cnt=0; cnt<inneriter;cnt++)
    {   diff=0;
        for ( int i=0 ; i<r ; i++ )
            {  // find gradient and hessian at H_{ij}, fix column j
                 grad=0;
                 hessian=0;
                 im=i*m;
                 for (int idx=0 ; idx<m ; idx++ )
                     { id_i = i+idx*r; //W(idx,i)
                      // id_j	= jm+idx;
                       Wttemp=Wt[id_i]; // just to save time
                       WHcol_temp=WHcol[idx];
                       temp = Xcol[idx]/(WHcol_temp+1e-16);
                       grad = grad + Wttemp*(1-temp); 
                       hessian = hessian + Wttemp*Wttemp*temp/(WHcol_temp+1e-16);   
                     }
                 // find Newton direction
                
                vold = Hcol[i]; //H(i,j)
                s=vold-grad/(hessian+1e-10);
                if (s<1e-16)
                    s=1e-16;
                
                if (grad>0) 
                 { d=s-vold;
                   lambda=sc_coeff*sqrt(hessian)*fabs(d);
                   if (lambda > 0.683802)
                    {vnew=vold+1/(1+lambda)*d; // damp Newton step
                     if (vnew<1e-16)
                        vnew=1e-16;
                    }
                   else
                   vnew=s;
                 }
                else
                { vnew=s;
                }
                // update the j column of matrix WH
                vdiff=vnew-vold; 
                diff += vdiff*vdiff;
                for (int idx=0 ; idx<m ; idx++ )
                    {  
                       //id_j	= jm+idx;
                       WHcol[idx] += Wt[i+idx*r]*vdiff;
                       if ( WHcol[idx]< 1e-16)
                            WHcol[idx]=1e-16;
                    }
                // update H(i,j)
                Hcol[i]=vnew;
            }
    if (cnt==0) 
    {
        eps0=sqrt(diff);
        eps1=eps0;
    }
    else eps1=sqrt(diff);
    if (eps1<delta*eps0)
        break;
    }
    
    
}



int mainupdate(int m, int n, int r, int maxiter, double maxtime, double *X, double *Wt, double *H, double *obj, double *time, int inneriter, double delta, int obj_compute)
{
    double total = 0, timestart, temp, totaltime, checkout=0;
  	char matlab_output[1024];
    double *WH = (double *)malloc(sizeof(double)*m*n);
    double *col=(double *)malloc(sizeof(double)*n);
    double *row=(double *)malloc(sizeof(double)*m);
    
    double *Xt=(double *)malloc(sizeof(double)*m*n);
    double *WH_row=(double *)malloc(sizeof(double)*n);
    int reallength=0;
    
    int idx=0, id, jr, ir,jm;
    
    
    timestart = clock();
    // find W*H from the input Wt and H
    for (int j=0 ; j<n ; j++ )
    {    
        jr=j*r;
       for ( int i=0 ; i<m ; i++ )
		{   // find WH(i,j)
			temp = 0;
            ir=i*r;
            for (int k=0 ; k<r ; k++ )
            {
                temp += Wt[k+ir]*H[k+jr];
            }
            WH[idx]=temp;
            idx +=  1; 
		}
    }
    // find col(k):  xk=X(:,k);    col(k)=sqrt(min(xk(xk>0)));
    for (int j=0; j<n; j++)
        {  jm=j*m;
            col[j]=sqrt(X[jm]);
         for (int i=1; i<m; i++)    
             {   id=jm+i;
                 if (X[id]>0 && col[j]>0)
                 {
                     col[j]=std::min(col[j],X[id]);
                 }

                 else if (X[id]>0 && col[j]==0)
                 {
                     col[j]=sqrt(X[id]);   
                 }
             }
         col[j]=1/col[j];  
         
        } 
    
    // find row(k): min by row 
    for (int i=0; i<m; i++)
        {   //id=i*n;
            row[i]=sqrt(X[i]);
            for (int j=1; j<n; j++)    
             {   idx=j*m+i;
                 if (X[idx]>0 && row[i]>0)
                 {
                     row[i]=std::min(row[i],X[idx]);
                 }

                 else if (X[idx]>0 && row[i]==0)
                 { row[i]=sqrt(X[idx]);
                 }

             }
           row[i]=1/row[i];   
        }
    // Xt=X';
     for (int idW=0; idW<n; idW++)
         { id=idW*m;
          for (int cnt=0; cnt<m; cnt++)
                Xt[cnt*n+idW]=X[id+cnt];
         }
    totaltime=(clock()-timestart)/CLOCKS_PER_SEC;
   	time[0] = totaltime;
    if (obj_compute==1)
     {obj[0]=objective(m, n, X, WH);
     }
    else 
        obj[0]=0;
    for ( int iter=1 ; iter<maxiter ; iter++)
    {     timestart = clock();
        // update H
           reallength +=1;
           for (int idH=0; idH<n; idH++)
           { int sc_coeff=col[idH];
             double *X_col= &(X[idH*m]);
             double *H_col= &(H[idH*r]);
             double *WH_col= &(WH[idH*m]);
             update_colH(m, r,sc_coeff, X_col,  H_col,  WH_col, Wt,inneriter,delta);
           }
       
        // update W
        
            for (int idW=0; idW<m; idW++)
            {   int sc_coeff=row[idW];
                double *Wt_col=&(Wt[idW*r]);
                double *X_row =&(Xt[idW*n]);
                for (int cnt=0; cnt<n; cnt++)
                {// int i=idW+cnt*m;
                 // X_row[cnt]=X[i];
                  WH_row[cnt]=WH[idW+cnt*m];
                }
                update_colH(n, r,sc_coeff, X_row,  Wt_col,  WH_row, H,inneriter, delta);
                
                for (int cnt=0; cnt<n; cnt++)
                { int i = idW + cnt*m;
                  WH[i]=WH_row[cnt];
                }
            }
         
         totaltime += (clock()-timestart)/CLOCKS_PER_SEC;
         time[iter ] =totaltime;
         if (obj_compute==1)
         {obj[iter]=objective(m, n, X, WH);
                
        //  sprintf(matlab_output, "display('SN: Iteration %d Objective: %lf Time taken: %lf');", iter, obj[iter], time[iter]);
		// 	mexEvalString(matlab_output);
         }
         else 
             obj[iter]=0;
         if (totaltime>maxtime)
         {break;
         }
    }
    
  
	free(WH);
    free(col);
    free(row);
    free(WH_row);
    free(Xt);
    return (reallength+1);
}


