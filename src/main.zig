const std = @import("std");

const c = @cImport({
    @cInclude("stb/stb_image_write.h");
    @cInclude("cgltf/cgltf.h");
});

fn CGLTF_CHECK(result: c.cgltf_result) !void {
    return if (result == c.cgltf_result_success) {} else error.GltfError;
}

const default_input = "input.gltf";
const default_output = "output.png";

const CmdlineArgs = struct {
    in: [:0]const u8 = default_input,
    out: [:0]const u8 = default_output,
    width: ?u16 = null,
    height: ?u16 = null,

    fn deinit(self: CmdlineArgs, allocator: std.mem.Allocator) void {
        if (self.in.ptr != default_input.ptr) {
            allocator.free(self.in);
        }
        if (self.out.ptr != default_output.ptr) {
            allocator.free(self.out);
        }
    }

    fn print(self: CmdlineArgs) void {
        std.log.debug("--in {s} --out {s} --width {?} --height {?}", .{
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

const Vec3 = struct {
    data: @Vector(3, f32),

    fn x(self: Vec3) f32 {
        return self.data[0];
    }
    fn y(self: Vec3) f32 {
        return self.data[1];
    }
    fn z(self: Vec3) f32 {
        return self.data[2];
    }

    fn zeroes() Vec3 {
        return .{ .data = .{0.0, 0.0, 0.0}};
    }

    fn ones() Vec3 {
        return .{ .data = .{1.0, 1.0, 1.0}};
    }

    fn init(_x: f32, _y: f32, _z: f32) Vec3 {
        return .{ .data = .{_x, _y, _z}};
    }

    fn sqrt(self: Vec3) Vec3 {
        return .{ .data = @sqrt(self.data) };
    }

    fn clamp(self: Vec3, min: f32, max: f32) Vec3 {
        return .{ .data = @min(@max(self.data, @splat(3, min)), @splat(3, max)) };
    }

    fn scale(self: Vec3, s: f32) Vec3 {
        return .{ .data = self.data * @splat(3, s) };
    }

    fn toRGB(self: Vec3) RGB {
        const rgb = self.clamp(0.0, 0.999999).scale(256);
        return .{
            .r = @intFromFloat(rgb.data[0]),
            .g = @intFromFloat(rgb.data[1]),
            .b = @intFromFloat(rgb.data[2]),
        };
    }

    fn length(self: Vec3) f32 {
        return @sqrt(@reduce(.Add, self.data * self.data));
    }

    fn normalize(self: Vec3) Vec3 {
        return self.scale(1.0 / self.length());
    }

    fn add(self: Vec3, b: Vec3) Vec3 {
        return .{.data = self.data + b.data};
    }

    fn subtract(self: Vec3, b: Vec3) Vec3 {
        return .{.data = self.data - b.data};
    }
};

fn add(a: Vec3, b: Vec3) Vec3 {
    return .{.data = a.data + b.data};
}

fn subtract(a: Vec3, b: Vec3) Vec3 {
    return .{.data = a.data - b.data};
}

fn dot(a: Vec3, b: Vec3) f32 {
    return @reduce(.Add, a.data * b.data);
}

fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{ .data = .{
        a.y()*b.z() - a.z()*b.y(),
        a.z()*b.x() - a.x()*b.z(),
        a.x()*b.y() - a.y()*b.x(),
    }};
}

fn vec3(x: f32, y: f32, z: f32) Vec3 {
    return .{ .data = .{x, y, z}};
}

test "cross product" {
    const a = vec3(1,-8,12);
    const b = vec3(4,6,3);
    const result = vec3(-96,45,38);
    try std.testing.expectEqual(cross(a,b), result);
}

test "vector length" {
    const v = vec3(1.5, 100.0, -21.1);
    try std.testing.expectApproxEqAbs(v.length(), 102.21281720019266, 0.0001);
}

const Camera = struct {
    w: u32,
    h: u32,
    origin: Vec3,
    lower_left_corner: Vec3,
    right: Vec3,
    up: Vec3,

    fn findCameraNode(gltf_data: *c.cgltf_data) !*c.cgltf_node {
        for (0..gltf_data.nodes_count) |node_idx| {
            const node: *c.cgltf_node = gltf_data.nodes + node_idx;
            if (node.camera != null) {
                return node;
            }
        }
        return error.CameraNodeNotFound; // TODO: implement recursive search / deal with multiple instances of the same camera
    }

    fn init(gltf_data: *c.cgltf_data, width: ?u16, height: ?u16) !Camera {

        const camera_node = try findCameraNode(gltf_data);

        const camera: *c.cgltf_camera = camera_node.camera;
        if (camera.type != c.cgltf_camera_type_perspective) {
            return error.OnlyPerspectiveCamerasSupported;
        }

        var w: u32 = 0;
        var h: u32 = 0;

        if (width == null and height == null)
        {
            return error.OutputImgSizeIsNotSpecified;
        }
        else if (width != null and height != null)
        {
            if (camera.data.perspective.has_aspect_ratio != 0) {
                return error.CameraHasAspectRatio;
            }
            w = width.?;
            h = height.?;
        }
        else
        {
            if (camera.data.perspective.has_aspect_ratio == 0) {
                return error.CameraHasntAspectRatio;
            }
            const aspect_ratio = camera.data.perspective.aspect_ratio;
            w = width orelse @intFromFloat(@as(f32, @floatFromInt(height.?)) * aspect_ratio);
            h = height orelse @intFromFloat(@as(f32, @floatFromInt(width.?)) / aspect_ratio);
        }

        const f_w: f32 = @floatFromInt(w);
        const f_h: f32 = @floatFromInt(h);

        const origin = vec3(0,0,-1000);
        const lookat = vec3(0,0,0);
        const world_up = vec3(0,1,0);

        const fwd = subtract(lookat, origin).normalize();
        const right = cross(world_up, fwd).normalize();
        const up = cross(fwd, right);

        const focal_length = (f_h / 2) / @tan(camera.data.perspective.yfov / 2);

        const lower_left_corner = fwd.scale(focal_length)
            .subtract(right.scale(f_w / 2))
            .subtract(up.scale(f_h / 2));

        return .{
            .w = w,
            .h = h,
            .origin = origin,
            .lower_left_corner = lower_left_corner,
            .right = right,
            .up = up
        };
    }

    fn getRandomRay(self: Camera, x: u16, y: u16) Ray {
        const f_x: f32 = @floatFromInt(x); // TODO: add rand
        const f_y: f32 = @floatFromInt(y); // TODO: add rand
        return .{
            .orig = self.origin,
            .dir = self.lower_left_corner
                .add(self.right.scale(f_x))
                .add(self.up.scale(f_y))
                .normalize()
        };
    }
};

const Ray = struct {
    orig: Vec3,
    dir: Vec3,

    fn at(self: Ray, t: f32) Vec3 {
        return add(self.orig, self.dir.scale(t));
    }
};

fn getEnvColor(ray: Ray) Vec3 {
    const t = 0.5*(ray.dir.y()+1.0);
    return add(
        Vec3.ones().scale(1.0-t),
        Vec3.init(0.5, 0.7, 1.0).scale(t)
    );
}

fn rayTriangleIntersection(ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3) bool
{
    const v0v1 = subtract(v1, v0);
    const v0v2 = subtract(v2, v0);
    const N = cross(v0v1, v0v2); // no need to normalize

    const epsilon = 0.00000001; // TODO

    // Step 1: finding P

    const NdotRayDirection = dot(N, ray.dir);
    if (@fabs(NdotRayDirection) < epsilon) {
        return false; // they are parallel, so they don't intersect!
    }

    const d = -dot(N, v0);
    const t = -(dot(N, ray.orig) + d) / NdotRayDirection;

    if (t < 0) return false; // the triangle is behind

    const P = ray.at(t);

    // Step 2: inside-outside test

    const edge0 = subtract(v1, v0);
    const edge1 = subtract(v2, v1);
    const edge2 = subtract(v0, v2);

    const vp0 = subtract(P, v0);
    const vp1 = subtract(P, v1);
    const vp2 = subtract(P, v2);

    if (dot(N, cross(edge0, vp0)) < 0) return false;
    if (dot(N, cross(edge1, vp1)) < 0) return false;
    if (dot(N, cross(edge2, vp2)) < 0) return false;

    return true;
}

fn traceRay(ray: Ray, gltf_data: *c.cgltf_data) bool {
    std.debug.assert(gltf_data.meshes_count == 1);
    const mesh = gltf_data.meshes[0];
    std.debug.assert(mesh.primitives_count == 1);
    const primitive = mesh.primitives[0];
    std.debug.assert(primitive.type == c.cgltf_primitive_type_triangles);

    const positions: *c.cgltf_accessor = blk: {
        for (0..primitive.attributes_count) |i| {
            if (primitive.attributes[i].type == c.cgltf_attribute_type_position) {
                break :blk primitive.attributes[i].data;
            }
        }
        @panic("imlement me");
    };
    const indices: *c.cgltf_accessor = primitive.indices;

    const vertex_address =
        @as([*]u8, @ptrCast(positions.buffer_view.*.buffer.*.data)) +
        positions.buffer_view.*.offset +
        positions.offset;

    const index_address =
        @as([*]u8, @ptrCast(indices.buffer_view.*.buffer.*.data)) +
        indices.buffer_view.*.offset +
        indices.offset;

    std.debug.assert(positions.type == c.cgltf_type_vec3);
    std.debug.assert(indices.type == c.cgltf_type_scalar);
    std.debug.assert(positions.component_type == c.cgltf_component_type_r_32f);
    std.debug.assert(indices.component_type == c.cgltf_component_type_r_16u);

    var i: usize = 0;
    while (i < indices.count) : (i += 3)
    {
        const index: [*]u16 = @ptrCast(@alignCast(index_address + i*indices.stride));

        const pos0: [*]f32 = @ptrCast(@alignCast(vertex_address + index[0]*positions.stride));
        const pos1: [*]f32 = @ptrCast(@alignCast(vertex_address + index[1]*positions.stride));
        const pos2: [*]f32 = @ptrCast(@alignCast(vertex_address + index[2]*positions.stride));

        const v0 = vec3(pos0[0], pos0[1], pos0[2]);
        const v1 = vec3(pos1[0], pos1[1], pos1[2]);
        const v2 = vec3(pos2[0], pos2[1], pos2[2]);

        if (rayTriangleIntersection(ray, v0, v1, v2)) {
            return true;
        }
    }
    return false;
}

fn getRayColor(ray: Ray, gltf_data: *c.cgltf_data) Vec3 {
    const hit = traceRay(ray, gltf_data);

    if (hit == false) {
        return getEnvColor(ray);
    }

    return vec3(0, 0, 0);
}

fn getPixelColor(x: u16, y: u16, camera: Camera, gltf_data: *c.cgltf_data) RGB
{
    const ray = camera.getRandomRay(x, y);
    // std.debug.print("{}\n", .{ray});
    const color = getRayColor(ray, gltf_data);
    return color.sqrt().toRGB();
}

pub const std_options = struct {
    pub const log_level = .info;
};

pub fn main() !void {
    const start_time = std.time.nanoTimestamp();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const args = try parseCmdline(allocator);
    defer args.deinit(allocator);
    args.print();

    const options = std.mem.zeroes(c.cgltf_options);
    var gltf_data: ?*c.cgltf_data = null;
    try CGLTF_CHECK(c.cgltf_parse_file(&options, args.in.ptr, &gltf_data));
    defer c.cgltf_free(gltf_data);

    try CGLTF_CHECK(c.cgltf_load_buffers(&options, gltf_data, std.fs.path.dirname(args.in).?.ptr));

    const camera = try Camera.init(gltf_data.?, args.width, args.height);

    var img = try allocator.alloc(RGB, camera.w * camera.h);
    defer allocator.free(img);

    for (0..camera.h) |y| {
        for (0..camera.w) |x| {
            const row = camera.h - 1 - y;
            img[row*camera.w+x] = getPixelColor(@intCast(x), @intCast(y), camera, gltf_data.?);
        }
    }

    const res = c.stbi_write_png(args.out.ptr,
        @intCast(camera.w),
        @intCast(camera.h),
        @intCast(@sizeOf(RGB)),
        img.ptr,
        @intCast(@sizeOf(RGB) * camera.w));

    if (res != 1) {
        return error.WritePngFail;
    }

    const end_time = std.time.nanoTimestamp();
    const time_ns: u64 = @intCast(end_time - start_time);
    std.log.info("Done in {}", .{std.fmt.fmtDuration(time_ns)});
}
