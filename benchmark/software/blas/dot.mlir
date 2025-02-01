module {
  func.func @dot(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: memref<f64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.store %cst, %arg2[] : memref<f64>
    %c1000 = arith.constant 1000 : index
    affine.for %arg3 = 0 to 1000 {
      %0 = affine.load %arg0[%arg3] : memref<1000xf64>
      %1 = affine.load %arg1[%arg3] : memref<1000xf64>
      %2 = arith.mulf %0, %1 : f64
      %3 = affine.load %arg2[] : memref<f64>
      %4 = arith.addf %3, %2 : f64
      affine.store %4, %arg2[] : memref<f64>
    }
    return
  }
}

