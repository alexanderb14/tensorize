module {
  func.func @softmax_part2(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: f64) {
    affine.for %arg3 = 0 to 1000 {
      %0 = affine.load %arg0[%arg3] : memref<1000xf64>
      %1 = arith.subf %0, %arg2 : f64
      %2 = math.exp %1 : f64
      affine.store %2, %arg1[%arg3] : memref<1000xf64>
    }
    return
  }
}

