module {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<150x140x160xf64>, %arg4: memref<160x160xf64>, %arg5: memref<160xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 150 {
      affine.for %arg7 = 0 to 140 {
        affine.for %arg8 = 0 to 160 {
          affine.store %cst, %arg5[%arg8] : memref<160xf64>
          affine.for %arg9 = 0 to 160 {
            %0 = affine.load %arg3[%arg6, %arg7, %arg9] : memref<150x140x160xf64>
            %1 = affine.load %arg4[%arg9, %arg8] : memref<160x160xf64>
            %2 = arith.mulf %0, %1 : f64
            %3 = affine.load %arg5[%arg8] : memref<160xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %arg5[%arg8] : memref<160xf64>
          }
        }
        affine.for %arg8 = 0 to 160 {
          %0 = affine.load %arg5[%arg8] : memref<160xf64>
          affine.store %0, %arg3[%arg6, %arg7, %arg8] : memref<150x140x160xf64>
        }
      }
    }
    return
  }
}

