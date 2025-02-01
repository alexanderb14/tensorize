module {
  func.func @fn(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>) {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.600000e+01 : f64
    %cst_1 = arith.constant 3.200000e+01 : f64
    affine.for %arg2 = 0 to 1000 {
      affine.for %arg3 = 0 to 1000 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<1000x1000xf64>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<1000x1000xf64>
        %2 = arith.cmpf oge, %0, %cst_0 : f64
        %3 = arith.mulf %cst, %0 : f64
        %4 = arith.mulf %0, %0 : f64
        %5 = arith.mulf %cst, %4 : f64
        %6 = arith.divf %5, %cst_1 : f64
        %7 = arith.addf %3, %0 : f64
        %8 = arith.subf %7, %6 : f64
        %9 = arith.subf %8, %cst_1 : f64
        %10 = arith.mulf %0, %0 : f64
        %11 = arith.mulf %cst, %10 : f64
        %12 = arith.divf %11, %cst_1 : f64
        %13 = arith.select %2, %9, %12 : f64
        affine.store %13, %arg0[%arg2, %arg3] : memref<1000x1000xf64>
      }
    }
    return
  }
}

