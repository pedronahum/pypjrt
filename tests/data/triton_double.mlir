module {
  tt.func public @double_kernel(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %off = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %pin = tt.splat %in : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %pi = tt.addptr %pin, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %pi : tensor<64x!tt.ptr<f32>>
    %two = arith.constant dense<2.000000e+00> : tensor<64xf32>
    %r = arith.mulf %v, %two : tensor<64xf32>
    %pout = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %po = tt.addptr %pout, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %po, %r : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
