// Single Zig compilation root: exports both the numerical kernels and R .C adapter.
const core = @import("semantic_core.zig");
const r_abi = @import("r_abi.zig");
comptime {
    _ = core.sl_softmax_f64;
    _ = core.sl_entropy_f64;
    _ = r_abi.sl_softmax_r;
    _ = r_abi.sl_entropy_r;
}
