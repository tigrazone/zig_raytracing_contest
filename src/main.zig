const std = @import("std");

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
        std.debug.print("--in {s} --out {s} --width {} --height {}\n", .{
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);

    const allocator = gpa.allocator();

    const cmdline_args = try parseCmdline(allocator);
    defer cmdline_args.deinit(allocator);

    cmdline_args.print();
}
