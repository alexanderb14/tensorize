#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg6 = 0 to 1200 {
      affine.for %arg7 = 0 to #map(%arg6) {
        %0 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
        %1 = arith.mulf %0, %arg3 : f64
        affine.store %1, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
      }
      affine.for %arg7 = 0 to 1000 {
        affine.for %arg8 = 0 to #map(%arg6) {
          %0 = affine.load %arg5[%arg6, %arg7] : memref<1200x1000xf64>
          %1 = arith.mulf %arg2, %0 : f64
          %2 = affine.load %arg5[%arg8, %arg7] : memref<1200x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg6, %arg8] : memref<1200x1200xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg6, %arg8] : memref<1200x1200xf64>
        }
      }
    }
    return
  }
}

