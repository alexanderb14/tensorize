module {
  func.func @lerp(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: memref<1000xf64>, %arg3: f64) {
    %cst = arith.constant 1.000000e+00 : f64
    affine.for %arg4 = 0 to 1000 {
      %0 = affine.load %arg1[%arg4] : memref<1000xf64>
      %1 = affine.load %arg2[%arg4] : memref<1000xf64>
      %2 = arith.mulf %arg3, %0 : f64
      %3 = arith.subf %cst, %arg3 : f64
      %4 = arith.mulf %3, %1 : f64
      %5 = arith.addf %2, %4 : f64
      affine.store %5, %arg0[%arg4] : memref<1000xf64>
    }
    return
  }
}

