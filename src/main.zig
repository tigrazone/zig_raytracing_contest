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

const Matrix4 = struct {
    data: [16]f32,

    fn get(self: Matrix4, row: usize, column: usize) f32 {
        return self.data[column*4+row];
    }

    fn set(self: *Matrix4, row: usize, column: usize, val: f32) void {
        self.data[column*4+row] = val;
    }

    fn identity() Matrix4 {
        var result: Matrix4 = .{ .data = [_]f32{0} ** 16 };
        for (0..4) |i| {
            result.data[i*4+i] = 1;
        }
        return result;
    }

    fn translation(t: [3]f32) Matrix4 {
        return .{ .data = .{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            t[0], t[1], t[2], 1,
        }};
    }

    fn rotation(q: [4]f32) Matrix4 {
        const x = q[0];
        const y = q[1];
        const z = q[2];
        const w = q[3];

        const x2 = x * x;
        const y2 = y * y;
        const z2 = z * z;
        const xy = x * y;
        const xz = x * z;
        const yz = y * z;
        const wx = w * x;
        const wy = w * y;
        const wz = w * z;

        return .{ .data = .{
            1.0-2*(y2+z2), 2*(xy+wz),     2*(xz-wy), 0,
            2*(xy-wz),     1.0-2*(x2+z2), 2*(yz+wx), 0,
            2*(xz+wy),     2*(yz-wx),     1.0-2*(x2+y2), 0,
            0, 0, 0, 1,
        }};
    }

    fn scale(s: [3]f32) Matrix4 {
        return .{ .data = .{
            s[0], 0, 0, 0,
            0, s[1], 0, 0,
            0, 0, s[2], 0,
            0, 0, 0, 1,
        }};
    }

    fn col3(self: Matrix4, column: usize) Vec3 {
        return vec3(
            self.get(0, column),
            self.get(1, column),
            self.get(2, column),
        );
    }

    fn transformPosition(self: Matrix4, v: Vec3) Vec3 {
        return .{ .data =
            self.col3(0).scale(v.x()).data +
            self.col3(1).scale(v.y()).data +
            self.col3(2).scale(v.z()).data +
            self.col3(3).data
        };
    }
};

fn mul(a: Matrix4, b: Matrix4) Matrix4 {
    var result: Matrix4 = undefined;
    for (0..4) |column| {
        for (0..4) |row| {
            var acc: f32 = 0;
            for (0..4) |i| {
                acc += a.get(row, i) * b.get(i, column);
            }
            result.set(row, column, acc);
        }
    }
    return result;
}

fn getNodeTransform(node: *c.cgltf_node) !Matrix4 {
    var matrix: Matrix4 = undefined;
    if (node.has_matrix != 0) {
        matrix = .{ .data = node.matrix };
    } else if (node.has_translation != 0 or node.has_rotation != 0 or node.has_scale != 0) {
        matrix = Matrix4.identity();
        if (node.has_translation != 0) {
            matrix = mul(matrix, Matrix4.translation(node.translation));
        }
        if (node.has_rotation != 0) {
            matrix = mul(matrix, Matrix4.rotation(node.rotation));
        }
        if (node.has_scale != 0) {
            matrix = mul(matrix, Matrix4.scale(node.scale));
        }
    } else {
        matrix = Matrix4.identity();
    }
    if (node.parent == null) {
        return matrix;
    } else {
        return mul(try getNodeTransform(node.parent), matrix);
    }
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

        const matrix = try getNodeTransform(camera_node);
        const origin = vec3(matrix.get(0,3), matrix.get(1,3), matrix.get(2,3));
        const fwd = vec3(matrix.get(0,2), matrix.get(1,2), matrix.get(2,2)).scale(-1).normalize();
        const world_up = vec3(0,1,0);

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

fn rayTriangleIntersection(ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3, t_min: f32, t_max: f32) ?Hit
{
    const v0v1 = subtract(v1, v0);
    const v0v2 = subtract(v2, v0);
    const N = cross(v0v1, v0v2);

    const epsilon = 0.00000001; // TODO

    // Step 1: finding P

    const NdotRayDir = dot(N, ray.dir);
    if (@fabs(NdotRayDir) < epsilon) {
        return null; // they are parallel, so they don't intersect!
    }

    const d = -dot(N, v0);
    const t = -(dot(N, ray.orig) + d) / NdotRayDir;

    if (t < t_min or t > t_max) return null;

    const P = ray.at(t);

    // Step 2: inside-outside test

    const edge0 = subtract(v1, v0);
    const edge1 = subtract(v2, v1);
    const edge2 = subtract(v0, v2);

    const vp0 = subtract(P, v0);
    const vp1 = subtract(P, v1);
    const vp2 = subtract(P, v2);

    if (dot(N, cross(edge0, vp0)) < 0) return null;
    if (dot(N, cross(edge1, vp1)) < 0) return null;
    if (dot(N, cross(edge2, vp2)) < 0) return null;

    return .{
        .p = P,
        .normal = N.normalize(),
        .t = t,
    };
}

const Triangle = struct {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
};

const Hit = struct {
    p: Vec3,
    normal: Vec3,
    t: f32,
};

const Scene = struct {
    camera: Camera,
    triangles: []Triangle,

    fn findMeshNode(gltf_data: *c.cgltf_data) !*c.cgltf_node {
        for (0..gltf_data.nodes_count) |node_idx| {
            const node: *c.cgltf_node = gltf_data.nodes + node_idx;
            if (node.mesh != null) {
                return node;
            }
        }
        return error.NoMeshFound; // TODO: implement recursive search / deal with multiple meshes
    }

    fn load(args: CmdlineArgs, allocator: std.mem.Allocator) !Scene {
        const options = std.mem.zeroes(c.cgltf_options);
        var gltf_data: ?*c.cgltf_data = null;
        try CGLTF_CHECK(c.cgltf_parse_file(&options, args.in.ptr, &gltf_data));
        defer c.cgltf_free(gltf_data);

        try CGLTF_CHECK(c.cgltf_load_buffers(&options, gltf_data, std.fs.path.dirname(args.in).?.ptr));

        const mesh_node = try findMeshNode(gltf_data.?);

        const mesh = mesh_node.mesh;
        std.debug.assert(mesh.*.primitives_count == 1);
        const primitive = mesh.*.primitives[0];
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

        const triangles_count = indices.count / 3;
        const triangles = try allocator.alloc(Triangle, triangles_count);
        errdefer allocator.free(triangles);

        const matrix = try getNodeTransform(mesh_node);

        for (0..triangles_count) |triangle_idx| {
            const index: [*]u16 = @ptrCast(@alignCast(index_address + triangle_idx*3*indices.stride));

            const pos0: [*]f32 = @ptrCast(@alignCast(vertex_address + index[0]*positions.stride));
            const pos1: [*]f32 = @ptrCast(@alignCast(vertex_address + index[1]*positions.stride));
            const pos2: [*]f32 = @ptrCast(@alignCast(vertex_address + index[2]*positions.stride));

            triangles[triangle_idx] = .{
                .v0 = matrix.transformPosition(vec3(pos0[0], pos0[1], pos0[2])),
                .v1 = matrix.transformPosition(vec3(pos1[0], pos1[1], pos1[2])),
                .v2 = matrix.transformPosition(vec3(pos2[0], pos2[1], pos2[2])),
            };
        }

        return .{
            .camera = try Camera.init(gltf_data.?, args.width, args.height),
            .triangles = triangles,
        };
    }

    fn deinit(self: Scene, allocator: std.mem.Allocator) void {
        allocator.free(self.triangles);
    }

    fn traceRay(scene: Scene, ray: Ray) ?Hit
    {
        var t = std.math.inf(f32);
        var result: ?Hit = null;
        for (scene.triangles) |triangle| {
            if (rayTriangleIntersection(ray, triangle.v0, triangle.v1, triangle.v2, 0, t)) |hit| {
                if (hit.t < t) {
                    result = hit;
                    t = hit.t;
                }
            }
        }
        return result;
    }

    fn getEnvColor(ray: Ray) Vec3 {
        const t = 0.5*(ray.dir.y()+1.0);
        return add(
            Vec3.ones().scale(1.0-t),
            Vec3.init(0.5, 0.7, 1.0).scale(t)
        );
    }

    fn getSampleColor(scene: Scene, ray: Ray) Vec3 {
        if (scene.traceRay(ray)) |hit| {
            return hit.normal;
        }

        return getEnvColor(ray);
    }

    fn getPixelColor(scene: Scene, x: u16, y: u16) RGB
    {
        const ray = scene.camera.getRandomRay(x, y);
        const color = scene.getSampleColor(ray);
        return color.sqrt().toRGB();
    }
};

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

    const scene = try Scene.load(args, allocator);
    defer scene.deinit(allocator);

    const w = scene.camera.w;
    const h = scene.camera.h;

    var img = try allocator.alloc(RGB, w * h);
    defer allocator.free(img);

    for (0..h) |y| {
        for (0..w) |x| {
            const row = h - 1 - y;
            const column = w - 1 - x;
            img[row*w+column] = scene.getPixelColor(@intCast(x), @intCast(y));
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

    const end_time = std.time.nanoTimestamp();
    const time_ns: u64 = @intCast(end_time - start_time);
    std.log.info("Done in {}", .{std.fmt.fmtDuration(time_ns)});
}
