#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1000x1000xf64> {irsynth.lower_triangular}, %arg4: memref<1000x1200xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg5 = 0 to 1000 {
      affine.for %arg6 = 0 to 1200 {
        affine.for %arg7 = #map(%arg5) to 1000 {
          %2 = affine.load %arg3[%arg7, %arg5] : memref<1000x1000xf64>
          %3 = affine.load %arg4[%arg7, %arg6] : memref<1000x1200xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %arg4[%arg5, %arg6] : memref<1000x1200xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %arg4[%arg5, %arg6] : memref<1000x1200xf64>
        }
        %0 = affine.load %arg4[%arg5, %arg6] : memref<1000x1200xf64>
        %1 = arith.mulf %arg2, %0 : f64
        affine.store %1, %arg4[%arg5, %arg6] : memref<1000x1200xf64>
      }
    }
    return
  }
}

