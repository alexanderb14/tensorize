#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64> {irsynth.symmetric}, %arg6: memref<1000x1200xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<f64>
    affine.for %arg7 = 0 to 1000 {
      affine.for %arg8 = 0 to 1200 {
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg9 = 0 to #map(%arg7) {
          %10 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
          %11 = arith.mulf %arg2, %10 : f64
          %12 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg4[%arg9, %arg8] : memref<1000x1200xf64>
          %15 = arith.addf %14, %13 : f64
          affine.store %15, %arg4[%arg9, %arg8] : memref<1000x1200xf64>
          %16 = affine.load %arg6[%arg9, %arg8] : memref<1000x1200xf64>
          %17 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
          %18 = arith.mulf %16, %17 : f64
          %19 = affine.load %alloca[] : memref<f64>
          %20 = arith.addf %19, %18 : f64
          affine.store %20, %alloca[] : memref<f64>
        }
        %0 = affine.load %arg4[%arg7, %arg8] : memref<1000x1200xf64>
        %1 = arith.mulf %arg3, %0 : f64
        %2 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
        %3 = arith.mulf %arg2, %2 : f64
        %4 = affine.load %arg5[%arg7, %arg7] : memref<1000x1000xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = arith.addf %1, %5 : f64
        %7 = affine.load %alloca[] : memref<f64>
        %8 = arith.mulf %arg2, %7 : f64
        %9 = arith.addf %6, %8 : f64
        affine.store %9, %arg4[%arg7, %arg8] : memref<1000x1200xf64>
      }
    }
    return
  }
}

