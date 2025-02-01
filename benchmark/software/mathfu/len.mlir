module {
  func.func @len(%arg0: memref<1000xf64>, %arg1: f64) -> memref<f64> {
    %alloca = memref.alloca() : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    affine.store %cst, %alloca[] : memref<f64>
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg0[%arg2] : memref<1000xf64>
      %1 = affine.load %arg0[%arg2] : memref<1000xf64>
      %2 = arith.mulf %0, %1 : f64
      %3 = affine.load %alloca[] : memref<f64>
      %4 = arith.addf %3, %2 : f64
      affine.store %4, %alloca[] : memref<f64>
    }
    return %alloca : memref<f64>
  }
}

