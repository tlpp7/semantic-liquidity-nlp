// Single compilation unit: explicitly import modules that own exported C ABI symbols.
comptime {
    _ = @import("semantic_core.zig");
    _ = @import("r_abi.zig");
}
