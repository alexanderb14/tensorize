module {
  func.func @rmsnorm_part2(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: memref<f64>) {
    affine.for %arg3 = 0 to 1 {
      %0 = affine.load %arg2[] : memref<f64>
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 3.000000e+00 : f64
      %1 = arith.divf %0, %cst_0 : f64
      %2 = arith.addf %1, %cst : f64
      %3 = math.sqrt %2 : f64
      %4 = arith.divf %cst, %3 : f64
      affine.store %4, %arg2[] : memref<f64>
    }
    affine.for %arg3 = 0 to 1000 {
      %0 = affine.load %arg2[] : memref<f64>
      %1 = affine.load %arg0[%arg3] : memref<1000xf64>
      %2 = affine.load %arg1[%arg3] : memref<1000xf64>
      %3 = arith.mulf %1, %0 : f64
      %4 = arith.mulf %3, %2 : f64
      affine.store %4, %arg0[%arg3] : memref<1000xf64>
    }
    return
  }
}

