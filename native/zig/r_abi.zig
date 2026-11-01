/// R .C ABI adapter: all arguments are pointers, including status and dimensions.
extern fn sl_softmax_f64(input: ?[*]const f64, output: ?[*]f64, rows: usize, cols: usize) c_int;
extern fn sl_entropy_f64(input: ?[*]const f64, output: ?[*]f64, rows: usize, cols: usize) c_int;
export fn sl_softmax_r(input: [*]const f64, output: [*]f64, rows: *const i32, cols: *const i32, status: *i32) void {
    if (rows.* <= 0 or cols.* < 2) { status.* = 1; return; }
    status.* = sl_softmax_f64(input, output, @intCast(rows.*), @intCast(cols.*));
}
export fn sl_entropy_r(input: [*]const f64, output: [*]f64, rows: *const i32, cols: *const i32, status: *i32) void {
    if (rows.* <= 0 or cols.* < 2) { status.* = 1; return; }
    status.* = sl_entropy_f64(input, output, @intCast(rows.*), @intCast(cols.*));
}
