module {
  func.func @softmax_part1(%arg0: memref<1000xf64>) -> memref<f64> {
    %alloca = memref.alloca() : memref<f64>
    affine.for %arg1 = 0 to 1 {
      %0 = affine.load %arg0[0] : memref<1000xf64>
      affine.store %0, %alloca[] : memref<f64>
      affine.for %arg2 = 1 to 1000 {
        %1 = affine.load %alloca[] : memref<f64>
        %2 = affine.load %arg0[%arg2] : memref<1000xf64>
        %3 = arith.maxf %1, %2 : f64
        affine.store %3, %alloca[] : memref<f64>
      }
    }
    return %alloca : memref<f64>
  }
}

