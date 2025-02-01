module {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1000xf64>, %arg7: memref<1000x900xf64>, %arg8: memref<900x1100xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<1200x1100xf64>, %arg11: memref<800x1100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg12 = 0 to 800 {
      affine.for %arg13 = 0 to 900 {
        affine.store %cst, %arg5[%arg12, %arg13] : memref<800x900xf64>
        affine.for %arg14 = 0 to 1000 {
          %0 = affine.load %arg6[%arg12, %arg14] : memref<800x1000xf64>
          %1 = affine.load %arg7[%arg14, %arg13] : memref<1000x900xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg5[%arg12, %arg13] : memref<800x900xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg5[%arg12, %arg13] : memref<800x900xf64>
        }
      }
    }
    affine.for %arg12 = 0 to 900 {
      affine.for %arg13 = 0 to 1100 {
        affine.store %cst, %arg8[%arg12, %arg13] : memref<900x1100xf64>
        affine.for %arg14 = 0 to 1200 {
          %0 = affine.load %arg9[%arg12, %arg14] : memref<900x1200xf64>
          %1 = affine.load %arg10[%arg14, %arg13] : memref<1200x1100xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg8[%arg12, %arg13] : memref<900x1100xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg8[%arg12, %arg13] : memref<900x1100xf64>
        }
      }
    }
    affine.for %arg12 = 0 to 800 {
      affine.for %arg13 = 0 to 1100 {
        affine.store %cst, %arg11[%arg12, %arg13] : memref<800x1100xf64>
        affine.for %arg14 = 0 to 900 {
          %0 = affine.load %arg5[%arg12, %arg14] : memref<800x900xf64>
          %1 = affine.load %arg8[%arg14, %arg13] : memref<900x1100xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg11[%arg12, %arg13] : memref<800x1100xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg11[%arg12, %arg13] : memref<800x1100xf64>
        }
      }
    }
    return
  }
}

