module {
  func.func @kernel_mvt(%arg0: i32, %arg1: memref<2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000x2000xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg6 = 0 to 2000 {
      affine.for %arg7 = 0 to 2000 {
        %0 = affine.load %arg1[%arg6] : memref<2000xf64>
        %1 = affine.load %arg5[%arg6, %arg7] : memref<2000x2000xf64>
        %2 = affine.load %arg3[%arg7] : memref<2000xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg1[%arg6] : memref<2000xf64>
      }
    }
    affine.for %arg6 = 0 to 2000 {
      affine.for %arg7 = 0 to 2000 {
        %0 = affine.load %arg2[%arg6] : memref<2000xf64>
        %1 = affine.load %arg5[%arg7, %arg6] : memref<2000x2000xf64>
        %2 = affine.load %arg4[%arg7] : memref<2000xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg2[%arg6] : memref<2000xf64>
      }
    }
    return
  }
}

