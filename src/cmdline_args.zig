
const std = @import("std");

const Self = @This();

const default_input = "input.gltf";
const default_output = "output.png";

in: [:0]const u8 = default_input,
out: [:0]const u8 = default_output,
width: ?u16 = null,
height: ?u16 = null,

pub fn init(allocator: std.mem.Allocator) !Self {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    var result: Self = .{};
    while (i < args.len) {
        const arg = args[i];
        i += 1;
        if (std.mem.eql(u8, arg, "--in")) {
            result.in = try allocator.dupeZ(u8, args[i]);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--out")) {
            result.out = try allocator.dupeZ(u8, args[i]);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--width")) {
            result.width = try std.fmt.parseInt(u16, args[i], 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--height")) {
            result.height = try std.fmt.parseInt(u16, args[i], 10);
            i += 1;
        } else {
            return error.UnsupportedCmdlineArgument;
        }

    }
    return result;
}

pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
    if (self.in.ptr != default_input.ptr) {
        allocator.free(self.in);
    }
    if (self.out.ptr != default_output.ptr) {
        allocator.free(self.out);
    }
}

pub fn print(self: Self) void {
    std.log.debug("--in {s} --out {s} --width {?} --height {?}", .{
        self.in,
        self.out,
        self.width,
        self.height,
    });
}
