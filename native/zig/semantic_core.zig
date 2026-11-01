const std = @import("std");
const math = std.math;

/// C ABI v1: 0 success, 1 invalid pointer/dimensions, 2 non-finite input.
/// Input and output are contiguous row-major f64 matrices (rows x cols).
export fn sl_softmax_f64(input: ?[*]const f64, output: ?[*]f64, rows: usize, cols: usize) c_int {
    const src = input orelse return 1;
    const dst = output orelse return 1;
    if (rows == 0 or cols == 0 or rows > 1000000 or cols > 1000000 or rows > math.maxInt(usize) / cols) return 1;
    for (0..rows) |r| {
        const offset = r * cols;
        var max_value: f64 = -math.inf(f64);
        for (0..cols) |c| {
            const v = src[offset + c];
            if (!math.isFinite(v)) return 2;
            max_value = @max(max_value, v);
        }
        var total: f64 = 0;
        for (0..cols) |c| {
            const value = @exp(src[offset + c] - max_value);
            dst[offset + c] = value;
            total += value;
        }
        for (0..cols) |c| dst[offset + c] /= total;
    }
    return 0;
}

/// Normalized entropy for a probability matrix; cols >= 2.
export fn sl_entropy_f64(input: ?[*]const f64, output: ?[*]f64, rows: usize, cols: usize) c_int {
    const src = input orelse return 1;
    const dst = output orelse return 1;
    if (rows == 0 or cols < 2 or rows > 1000000 or cols > 1000000 or rows > math.maxInt(usize) / cols) return 1;
    for (0..rows) |r| {
        var total: f64 = 0;
        var entropy: f64 = 0;
        for (0..cols) |c| {
            const p = src[r * cols + c];
            if (!math.isFinite(p) or p < 0 or p > 1) return 2;
            total += p;
            if (p > 0) entropy -= p * @log(p);
        }
        if (@abs(total - 1) > 1e-8) return 2;
        dst[r] = entropy / @log(@as(f64, @floatFromInt(cols)));
    }
    return 0;
}
