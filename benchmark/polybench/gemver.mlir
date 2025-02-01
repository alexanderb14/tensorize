module {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg12 = 0 to 2000 {
      affine.for %arg13 = 0 to 2000 {
        %0 = affine.load %arg9[%arg12] : memref<2000xf64>
        %1 = affine.load %arg3[%arg13, %arg12] : memref<2000x2000xf64>
        %2 = arith.mulf %arg2, %1 : f64
        %3 = affine.load %arg10[%arg13] : memref<2000xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg9[%arg12] : memref<2000xf64>
      }
    }
    affine.for %arg12 = 0 to 2000 {
      %0 = affine.load %arg9[%arg12] : memref<2000xf64>
      %1 = affine.load %arg11[%arg12] : memref<2000xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg9[%arg12] : memref<2000xf64>
    }
    affine.for %arg12 = 0 to 2000 {
      affine.for %arg13 = 0 to 2000 {
        %0 = affine.load %arg8[%arg12] : memref<2000xf64>
        %1 = affine.load %arg3[%arg12, %arg13] : memref<2000x2000xf64>
        %2 = arith.mulf %arg1, %1 : f64
        %3 = affine.load %arg9[%arg13] : memref<2000xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg8[%arg12] : memref<2000xf64>
      }
    }
    return
  }
}

