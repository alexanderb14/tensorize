module {
  func.func @matsub(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>) {
    affine.for %arg2 = 0 to 1000 {
      affine.for %arg3 = 0 to 1000 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<1000x1000xf64>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<1000x1000xf64>
        %2 = arith.subf %0, %1 : f64
        affine.store %2, %arg0[%arg2, %arg3] : memref<1000x1000xf64>
      }
    }
    return
  }
}

