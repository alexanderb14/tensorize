module {
  func.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 2100 {
      affine.store %cst, %arg4[%arg6] : memref<2100xf64>
    }
    affine.for %arg6 = 0 to 1900 {
      affine.store %cst, %arg5[%arg6] : memref<1900xf64>
      affine.for %arg7 = 0 to 2100 {
        %0 = affine.load %arg5[%arg6] : memref<1900xf64>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
        %2 = affine.load %arg3[%arg7] : memref<2100xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg5[%arg6] : memref<1900xf64>
      }
    }
    affine.for %arg6 = 0 to 1900 {
      affine.for %arg7 = 0 to 2100 {
        %0 = affine.load %arg4[%arg7] : memref<2100xf64>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
        %2 = affine.load %arg5[%arg6] : memref<1900xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg4[%arg7] : memref<2100xf64>
      }
    }
    return
  }
}

