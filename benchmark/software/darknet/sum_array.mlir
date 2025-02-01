module {
  func.func @sum_array(%arg0: memref<1000xf64>) -> memref<f64> {
    %alloca = memref.alloca() : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    affine.store %cst, %alloca[] : memref<f64>
    affine.for %arg1 = 0 to 1000 {
      %0 = affine.load %alloca[] : memref<f64>
      %1 = affine.load %arg0[%arg1] : memref<1000xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %alloca[] : memref<f64>
    }
    return %alloca : memref<f64>
  }
}

