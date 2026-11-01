/// SLP v1: fixed 32-byte little-endian header + row-major f64 payload.
/// Zig implementation intentionally handles only framing; no network I/O.
const std = @import("std");
pub const header_size: usize = 32;
pub const magic = "SLP1";
pub const Op = enum(u8) { softmax = 1, entropy = 2 };

fn put32(dst: []u8, value: u32) void {
    for (0..4) |i| dst[i] = @intCast((value >> @as(u5, @intCast(i * 8))) & 0xff);
}
fn put64(dst: []u8, value: u64) void {
    for (0..8) |i| dst[i] = @intCast((value >> @as(u6, @intCast(i * 8))) & 0xff);
}
fn get32(src: []const u8) u32 {
    var v: u32 = 0;
    for (0..4) |i| v |= @as(u32, src[i]) << @as(u5, @intCast(i * 8));
    return v;
}
fn get64(src: []const u8) u64 {
    var v: u64 = 0;
    for (0..8) |i| v |= @as(u64, src[i]) << @as(u6, @intCast(i * 8));
    return v;
}
pub const Header = struct {
    op: Op,
    rows: u32,
    cols: u32,
    payload_bytes: u64,
    request_id: u64,
};
pub fn encode(header: Header) [header_size]u8 {
    var out: [header_size]u8 = [_]u8{0} ** header_size;
    @memcpy(out[0..4], magic);
    out[4] = 1; // version
    out[5] = @intFromEnum(header.op);
    out[6] = 1; // f64
    out[7] = 0; // flags
    put32(out[8..12], header.rows);
    put32(out[12..16], header.cols);
    put64(out[16..24], header.payload_bytes);
    put64(out[24..32], header.request_id);
    return out;
}
pub fn decode(src: []const u8) !Header {
    if (src.len < header_size or !std.mem.eql(u8, src[0..4], magic) or src[4] != 1 or src[6] != 1 or src[7] != 0) return error.InvalidHeader;
    const op = std.meta.intToEnum(Op, src[5]) catch return error.InvalidOperation;
    const rows = get32(src[8..12]);
    const cols = get32(src[12..16]);
    const bytes = get64(src[16..24]);
    if (rows == 0 or cols < 2 or rows > 1000000 or cols > 1000000) return error.InvalidShape;
    const expected = @as(u64, rows) * @as(u64, cols) * 8;
    if (bytes != expected or bytes > 128 * 1024 * 1024) return error.InvalidPayloadSize;
    return .{ .op = op, .rows = rows, .cols = cols, .payload_bytes = bytes, .request_id = get64(src[24..32]) };
}
test "SLP v1 header roundtrip" {
    const input = Header{ .op = .softmax, .rows = 2, .cols = 3, .payload_bytes = 48, .request_id = 99 };
    const decoded = try decode(&encode(input));
    try std.testing.expectEqual(input.request_id, decoded.request_id);
    try std.testing.expectEqual(input.payload_bytes, decoded.payload_bytes);
}
