module {
  func.func @fn(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>) {
    %cst = arith.constant 3.200000e+01 : f64
    affine.for %arg2 = 0 to 1000 {
      affine.for %arg3 = 0 to 1000 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<1000x1000xf64>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<1000x1000xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = arith.divf %2, %cst : f64
        %4 = arith.addf %0, %1 : f64
        %5 = arith.subf %4, %3 : f64
        affine.store %5, %arg0[%arg2, %arg3] : memref<1000x1000xf64>
      }
    }
    return
  }
}

