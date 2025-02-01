#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>, %arg6: memref<1200xf64>) attributes {change_sizes.size_mode = "Uniform", llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-01 : f64
    affine.for %arg7 = 0 to 1200 {
      affine.store %cst_0, %arg5[%arg7] : memref<1200xf64>
      affine.for %arg8 = 0 to 1400 {
        %2 = affine.load %arg3[%arg8, %arg7] : memref<1400x1200xf64>
        %3 = affine.load %arg5[%arg7] : memref<1200xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg5[%arg7] : memref<1200xf64>
      }
      %0 = affine.load %arg5[%arg7] : memref<1200xf64>
      %1 = arith.divf %0, %arg2 : f64
      affine.store %1, %arg5[%arg7] : memref<1200xf64>
    }
    affine.for %arg7 = 0 to 1200 {
      affine.store %cst_0, %arg6[%arg7] : memref<1200xf64>
      affine.for %arg8 = 0 to 1400 {
        %0 = affine.load %arg3[%arg8, %arg7] : memref<1400x1200xf64>
        %1 = affine.load %arg5[%arg7] : memref<1200xf64>
        %2 = arith.subf %0, %1 : f64
        %3 = arith.mulf %2, %2 : f64
        %4 = affine.load %arg6[%arg7] : memref<1200xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg6[%arg7] : memref<1200xf64>
      }
    }
    affine.for %arg7 = 0 to 1200 {
      %0 = affine.load %arg6[%arg7] : memref<1200xf64>
      %1 = arith.divf %0, %arg2 : f64
      %2 = math.sqrt %1 : f64
      %3 = arith.cmpf ole, %2, %cst_1 : f64
      %4 = arith.select %3, %cst, %2 : f64
      affine.store %4, %arg6[%arg7] : memref<1200xf64>
    }
    affine.for %arg7 = 0 to 1400 {
      affine.for %arg8 = 0 to 1200 {
        %0 = affine.load %arg5[%arg8] : memref<1200xf64>
        %1 = affine.load %arg3[%arg7, %arg8] : memref<1400x1200xf64>
        %2 = arith.subf %1, %0 : f64
        affine.store %2, %arg3[%arg7, %arg8] : memref<1400x1200xf64>
        %3 = affine.load %arg6[%arg8] : memref<1200xf64>
        %4 = math.sqrt %arg2 : f64
        %5 = arith.mulf %4, %3 : f64
        %6 = arith.divf %2, %5 : f64
        affine.store %6, %arg3[%arg7, %arg8] : memref<1400x1200xf64>
      }
    }
    affine.for %arg7 = 0 to 1200 {
      affine.store %cst, %arg4[%arg7, %arg7] : memref<1200x1200xf64>
      affine.for %arg8 = #map(%arg7) to 1200 {
        affine.store %cst_0, %arg4[%arg7, %arg8] : memref<1200x1200xf64>
        affine.for %arg9 = 0 to 1400 {
          %1 = affine.load %arg3[%arg9, %arg7] : memref<1400x1200xf64>
          %2 = affine.load %arg3[%arg9, %arg8] : memref<1400x1200xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg7, %arg8] : memref<1200x1200xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg7, %arg8] : memref<1200x1200xf64>
        }
        %0 = affine.load %arg4[%arg7, %arg8] : memref<1200x1200xf64>
        affine.store %0, %arg4[%arg8, %arg7] : memref<1200x1200xf64>
      }
    }
    affine.store %cst, %arg4[1199, 1199] : memref<1200x1200xf64>
    return
  }
}

