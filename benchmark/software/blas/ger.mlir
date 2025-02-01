module {
  func.func @ger(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: memref<1000x1000xf64>) {
    affine.for %arg3 = 0 to 1000 {
      affine.for %arg4 = 0 to 1000 {
        %0 = affine.load %arg0[%arg3] : memref<1000xf64>
        %1 = affine.load %arg1[%arg4] : memref<1000xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %arg2[%arg3, %arg4] : memref<1000x1000xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg2[%arg3, %arg4] : memref<1000x1000xf64>
      }
    }
    return
  }
}

