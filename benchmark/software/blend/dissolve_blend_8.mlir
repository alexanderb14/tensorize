module {
  func.func @fn(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>, %arg2: f64, %arg3: f64) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+02 : f64
    affine.for %arg4 = 0 to 1000 {
      affine.for %arg5 = 0 to 1000 {
        %0 = affine.load %arg0[%arg4, %arg5] : memref<1000x1000xf64>
        %1 = affine.load %arg1[%arg4, %arg5] : memref<1000x1000xf64>
        %2 = arith.remf %arg3, %cst_1 : f64
        %3 = arith.addf %2, %cst_0 : f64
        %4 = arith.divf %3, %cst_1 : f64
        %5 = arith.subf %arg2, %4 : f64
        affine.store %5, %arg0[%arg4, %arg5] : memref<1000x1000xf64>
      }
    }
    return
  }
}

