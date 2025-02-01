module {
  func.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<2100x1900xf64>, %arg3: memref<1900xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>, %arg6: memref<2100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg7 = 0 to 1900 {
      affine.store %cst, %arg3[%arg7] : memref<1900xf64>
    }
    affine.for %arg7 = 0 to 2100 {
      affine.store %cst, %arg4[%arg7] : memref<2100xf64>
      affine.for %arg8 = 0 to 1900 {
        %0 = affine.load %arg3[%arg8] : memref<1900xf64>
        %1 = affine.load %arg6[%arg7] : memref<2100xf64>
        %2 = affine.load %arg2[%arg7, %arg8] : memref<2100x1900xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg3[%arg8] : memref<1900xf64>
        %5 = affine.load %arg4[%arg7] : memref<2100xf64>
        %6 = affine.load %arg2[%arg7, %arg8] : memref<2100x1900xf64>
        %7 = affine.load %arg5[%arg8] : memref<1900xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %5, %8 : f64
        affine.store %9, %arg4[%arg7] : memref<2100xf64>
      }
    }
    return
  }
}

