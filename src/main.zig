const std = @import("std");

const c = @cImport({
    @cInclude("stb/stb_image_write.h");
    @cInclude("cgltf/cgltf.h");
});

const default_input = "input.gltf";
const default_output = "output.png";

const CmdlineArgs = struct {
    in: [:0]const u8 = default_input,
    out: [:0]const u8 = default_output,
    width: u16 = 1280,
    height: u16 = 720,

    fn deinit(self: CmdlineArgs, allocator: std.mem.Allocator) void {
        if (self.in.ptr != default_input.ptr) {
            allocator.free(self.in);
        }
        if (self.out.ptr != default_output.ptr) {
            allocator.free(self.out);
        }
    }

    fn print(self: CmdlineArgs) void {
        std.log.debug("--in {s} --out {s} --width {} --height {}", .{
            self.in,
            self.out,
            self.width,
            self.height,
        });
    }
};

fn parseCmdline(allocator: std.mem.Allocator) !CmdlineArgs {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    var result: CmdlineArgs = .{};
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

const RGB = struct {
    r: u8,
    g: u8,
    b: u8,
};

test "RGB size should be 3" {
    try std.testing.expect(@sizeOf(RGB) == 3);
}

fn writeImg(args: CmdlineArgs, allocator: std.mem.Allocator) !void {
    const w: u32 = args.width;
    const h: u32 = args.height;
    var img = try allocator.alloc(RGB, w * h);
    defer allocator.free(img);

    for (0..h) |y| {
        for (0..w) |x| {
            img[y*w+x] = .{
                .r = @intFromFloat(@round(@as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(w-1)) * 255)),
                .g = @intFromFloat(@round(@as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(h-1)) * 255)),
                .b = 0,
            };
        }
    }
    const res = c.stbi_write_png(args.out.ptr,
        @intCast(w),
        @intCast(h),
        @intCast(@sizeOf(RGB)),
        img.ptr,
        @intCast(@sizeOf(RGB) * w));

    if (res != 1) {
        return error.WritePngFail;
    }
}

pub fn main() !void {
    const start_time = std.time.nanoTimestamp();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const args = try parseCmdline(allocator);
    defer args.deinit(allocator);
    args.print();

    try writeImg(args, allocator);

    const end_time = std.time.nanoTimestamp();
    const time_ns: u64 = @intCast(end_time - start_time);
    std.log.info("Done in {}", .{std.fmt.fmtDuration(time_ns)});
}
