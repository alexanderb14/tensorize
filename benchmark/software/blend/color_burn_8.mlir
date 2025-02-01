module {
  func.func @fn(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 3.200000e+01 : f64
    affine.for %arg2 = 0 to 1000 {
      affine.for %arg3 = 0 to 1000 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<1000x1000xf64>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<1000x1000xf64>
        %2 = arith.cmpf oeq, %cst, %1 : f64
        %3 = arith.subf %cst_0, %0 : f64
        %4 = arith.subf %cst_0, %3 : f64
        %5 = arith.divf %4, %1 : f64
        %6 = arith.subf %cst_0, %5 : f64
        %7 = arith.select %2, %cst_0, %6 : f64
        affine.store %7, %arg0[%arg2, %arg3] : memref<1000x1000xf64>
      }
    }
    return
  }
}

