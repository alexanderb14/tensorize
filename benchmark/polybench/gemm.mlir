module {
  func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1000x1100xf64>, %arg6: memref<1000x1200xf64>, %arg7: memref<1200x1100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg8 = 0 to 1000 {
      affine.for %arg9 = 0 to 1100 {
        %0 = affine.load %arg5[%arg8, %arg9] : memref<1000x1100xf64>
        %1 = arith.mulf %0, %arg4 : f64
        affine.store %1, %arg5[%arg8, %arg9] : memref<1000x1100xf64>
      }
      affine.for %arg9 = 0 to 1200 {
        affine.for %arg10 = 0 to 1100 {
          %0 = affine.load %arg6[%arg8, %arg9] : memref<1000x1200xf64>
          %1 = arith.mulf %arg3, %0 : f64
          %2 = affine.load %arg7[%arg9, %arg10] : memref<1200x1100xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg5[%arg8, %arg10] : memref<1000x1100xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg5[%arg8, %arg10] : memref<1000x1100xf64>
        }
      }
    }
    return
  }
}

