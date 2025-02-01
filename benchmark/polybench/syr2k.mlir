#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>, %arg6: memref<1200x1000xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg7 = 0 to 1200 {
      affine.for %arg8 = 0 to #map(%arg7) {
        %0 = affine.load %arg4[%arg7, %arg8] : memref<1200x1200xf64>
        %1 = arith.mulf %0, %arg3 : f64
        affine.store %1, %arg4[%arg7, %arg8] : memref<1200x1200xf64>
      }
      affine.for %arg8 = 0 to 1000 {
        affine.for %arg9 = 0 to #map(%arg7) {
          %0 = affine.load %arg5[%arg9, %arg8] : memref<1200x1000xf64>
          %1 = arith.mulf %0, %arg2 : f64
          %2 = affine.load %arg6[%arg7, %arg8] : memref<1200x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg6[%arg9, %arg8] : memref<1200x1000xf64>
          %5 = arith.mulf %4, %arg2 : f64
          %6 = affine.load %arg5[%arg7, %arg8] : memref<1200x1000xf64>
          %7 = arith.mulf %5, %6 : f64
          %8 = arith.addf %3, %7 : f64
          %9 = affine.load %arg4[%arg7, %arg9] : memref<1200x1200xf64>
          %10 = arith.addf %9, %8 : f64
          affine.store %10, %arg4[%arg7, %arg9] : memref<1200x1200xf64>
        }
      }
    }
    return
  }
}

