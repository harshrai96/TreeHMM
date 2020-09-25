library(reticulate)
library(RGF)
library("Matrix")
library(nmslibR)

    
if (reticulate::py_available() && reticulate::py_module_available("scipy")) {
  
    library(RGF)
    #library(nmslibR)
    
    print(fadjm)
    #res = TO_scipy_sparse(fadjm)
    #res = dgCMatrix_2scipy_sparse(fadjm)
    #print("SDF")
    #print(res$shape)
    
    
    
  }


