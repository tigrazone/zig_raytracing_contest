const std = @import("std");
const math = @import("math.zig");

const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const RGB = math.RGB;

const vec3 = Vec3.init;
const dot = Vec3.dot;
const cross = Vec3.cross;
const subtract = Vec3.subtract;
const add = Vec3.add;

const c = @cImport({
    @cInclude("stb/stb_image_write.h");
    @cInclude("cgltf/cgltf.h");
});

// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================











fn CGLTF_CHECK(result: c.cgltf_result) !void {
    if (result != c.cgltf_result_success) {
        std.log.err("GLTF error: {}", .{result});
        return error.GltfError;
    }
}

fn getNodeTransform(node: *c.cgltf_node) !Mat4 {
    var matrix: Mat4 = undefined;
    if (node.has_matrix != 0) {
        matrix = .{ .data = node.matrix };
    } else if (node.has_translation != 0 or node.has_rotation != 0 or node.has_scale != 0) {
        matrix = Mat4.identity();
        if (node.has_translation != 0) {
            matrix = Mat4.mul(matrix, Mat4.translation(node.translation));
        }
        if (node.has_rotation != 0) {
            matrix = Mat4.mul(matrix, Mat4.rotation(node.rotation));
        }
        if (node.has_scale != 0) {
            matrix = Mat4.mul(matrix, Mat4.scale(node.scale));
        }
    } else {
        matrix = Mat4.identity();
    }
    if (node.parent == null) {
        return matrix;
    } else {
        return Mat4.mul(try getNodeTransform(node.parent), matrix);
    }
}

fn findCameraNode(gltf_data: *c.cgltf_data) !*c.cgltf_node {
    for (0..gltf_data.nodes_count) |node_idx| {
        const node: *c.cgltf_node = gltf_data.nodes + node_idx;
        if (node.camera != null) {
            return node;
        }
    }
    return error.CameraNodeNotFound; // TODO: implement recursive search / deal with multiple instances of the same camera
}

fn findMeshNode(gltf_data: *c.cgltf_data) !*c.cgltf_node {
    for (0..gltf_data.nodes_count) |node_idx| {
        const node: *c.cgltf_node = gltf_data.nodes + node_idx;
        if (node.mesh != null) {
            return node;
        }
    }
    return error.NoMeshFound; // TODO: implement recursive search / deal with multiple meshes
}

fn findPrimitiveAttribute(primitive: c.cgltf_primitive, comptime attr_type: c.cgltf_attribute_type) !*c.cgltf_accessor {
    for (0..primitive.attributes_count) |i| {
        if (primitive.attributes[i].type == attr_type) {
            return primitive.attributes[i].data;
        }
    }
    return error.AttributeNotFound;
}

fn Accessor(comptime T: type) type {
    return struct {
        const Self = @This();

        accessor: *c.cgltf_accessor,
        base_address: [*]u8,

        fn init(accessor: *c.cgltf_accessor) Self
        {
            switch (T) {
                Vec3 => {
                    std.debug.assert(accessor.type == c.cgltf_type_vec3);
                    std.debug.assert(accessor.component_type == c.cgltf_component_type_r_32f);
                },
                [2]f32 => {
                    std.debug.assert(accessor.type == c.cgltf_type_vec2);
                    std.debug.assert(accessor.component_type == c.cgltf_component_type_r_32f);
                },
                u16 => {
                    std.debug.assert(accessor.type == c.cgltf_type_scalar);
                    std.debug.assert(accessor.component_type == c.cgltf_component_type_r_16u);
                },
                else => {
                    @compileError("Implement me");
                }
            }
            return .{
                .accessor = accessor,
                .base_address =
                    @as([*]u8, @ptrCast(accessor.buffer_view.*.buffer.*.data)) +
                    accessor.buffer_view.*.offset +
                    accessor.offset
            };
        }

        fn num(self: Self) usize { return self.accessor.count; }
        fn at(self: Self, idx: usize) T {
            switch (T) {
                Vec3 => {
                    const ptr: [*]f32 = @ptrCast(@alignCast(self.base_address + idx*self.accessor.stride));
                    return vec3(ptr[0],ptr[1],ptr[2]);
                },
                [2]f32 => {
                    const ptr: [*]f32 = @ptrCast(@alignCast(self.base_address + idx*self.accessor.stride));
                    return ptr[0..2].*;
                },
                u16 => {
                    const ptr: [*]u16 = @ptrCast(@alignCast(self.base_address + idx*self.accessor.stride));
                    return ptr[0];
                },
                else => {
                    @compileError("Implement me");
                }
            }
        }
    };
}









// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

const Camera = struct {
    w: u32,
    h: u32,
    origin: Vec3,
    lower_left_corner: Vec3,
    right: Vec3,
    up: Vec3,

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

        std.log.info("Pixels count: {}", .{w*h});

        const f_w: f32 = @floatFromInt(w);
        const f_h: f32 = @floatFromInt(h);

        const matrix = try getNodeTransform(camera_node);
        const origin = matrix.col3(3);
        const fwd = matrix.col3(2).scale(-1).normalize();
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
        return self.orig.add(self.dir.scale(t));
    }
};

const Triangle = struct {
    v: [3]Vec3,
};

const Hit = struct {
    t: f32,
    triangle_idx: usize,
};

const Vertex = struct {
    normal: Vec3,
    texcoord: [2]f32,
};

const TriangleData = struct {
    v: [3]Vertex,
};

const World = struct {
    triangles: []Triangle,
    triangles_data: []TriangleData,

    fn init(gltf_data: *c.cgltf_data, allocator: std.mem.Allocator) !World {
        const mesh_node = try findMeshNode(gltf_data);

        const mesh = mesh_node.mesh;
        std.debug.assert(mesh.*.primitives_count == 1);
        const primitive = mesh.*.primitives[0];
        std.debug.assert(primitive.type == c.cgltf_primitive_type_triangles);

        const positions = Accessor(Vec3).init(try findPrimitiveAttribute(primitive, c.cgltf_attribute_type_position));
        const normals = Accessor(Vec3).init(try findPrimitiveAttribute(primitive, c.cgltf_attribute_type_normal));
        const texcoords = Accessor([2]f32).init(try findPrimitiveAttribute(primitive, c.cgltf_attribute_type_texcoord));
        const indices = Accessor(u16).init(primitive.indices);

        const triangles_count = indices.num() / 3;

        const triangles = try allocator.alloc(Triangle, triangles_count);
        errdefer allocator.free(triangles);
        const triangles_data = try allocator.alloc(TriangleData, triangles_count);
        errdefer allocator.free(triangles_data);

        std.log.info("Triangle count: {}", .{triangles_count});

        const matrix = try getNodeTransform(mesh_node);

        for (0..triangles_count) |triangle_idx| {
            for (0..3) |i| {
                const index_idx = triangle_idx*3+i;
                const vertex_idx = indices.at(index_idx);
                triangles[triangle_idx].v[i] = matrix.transformPosition(positions.at(vertex_idx));
                triangles_data[triangle_idx].v[i] = .{
                    .normal = matrix.transformDirection(normals.at(vertex_idx)).normalize(), // TODO: use adjusent matrix
                    .texcoord = texcoords.at(vertex_idx),
                };
            }
        }

        return .{
            .triangles = triangles,
            .triangles_data = triangles_data,
        };
    }

    fn deinit(self: World, allocator: std.mem.Allocator) void {
        allocator.free(self.triangles);
        allocator.free(self.triangles_data);
    }

    fn rayTriangleIntersection(ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3, t_min: f32, t_max: f32) ?f32
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

        return t;
    }

    fn traceRay(world: World, ray: Ray) ?Hit
    {
        var nearest_t = std.math.inf(f32);
        var nearest_triangle_idx: usize = undefined;
        var found = false;
        for (world.triangles, 0..) |triangle, triangle_idx| {
            if (rayTriangleIntersection(ray, triangle.v[0], triangle.v[1], triangle.v[2], 0, nearest_t)) |t| {
                if (nearest_t > t) {
                    nearest_t = t;
                    nearest_triangle_idx = triangle_idx;
                    found = true;
                }
            }
        }
        if (found) {
            return .{
                .t = nearest_t,
                .triangle_idx = nearest_triangle_idx,
            };
        }
        return null;
    }

    fn getEnvColor(ray: Ray) Vec3 {
        const t = 0.5*(ray.dir.y()+1.0);
        return add(
            Vec3.ones().scale(1.0-t),
            Vec3.init(0.5, 0.7, 1.0).scale(t)
        );
    }

    fn getSampleColor(world: World, ray: Ray) Vec3 {
        if (world.traceRay(ray)) |hit| {
            // const uv: [2]f32 = world.triangles_data[hit.triangle_idx].getUV(hit.barycentric);
            // return vec3(uv[0], uv[1], 0);
            return world.triangles_data[hit.triangle_idx].v[0].normal;
        }

        return getEnvColor(ray);
    }
};

const Scene = struct {
    camera: Camera,
    world: World,

    fn load(args: CmdlineArgs, allocator: std.mem.Allocator) !Scene {
        const options = std.mem.zeroes(c.cgltf_options);
        var gltf_data: ?*c.cgltf_data = null;
        try CGLTF_CHECK(c.cgltf_parse_file(&options, args.in.ptr, &gltf_data));
        defer c.cgltf_free(gltf_data);

        try CGLTF_CHECK(c.cgltf_load_buffers(&options, gltf_data, std.fs.path.dirname(args.in).?.ptr));

        return .{
            .camera = try Camera.init(gltf_data.?, args.width, args.height),
            .world = try World.init(gltf_data.?, allocator),
        };
    }

    fn deinit(self: Scene, allocator: std.mem.Allocator) void {
        self.world.deinit(allocator);
    }

    fn getPixelColor(self: Scene, x: u16, y: u16) RGB
    {
        const ray = self.camera.getRandomRay(x, y);
        const color = self.world.getSampleColor(ray);
        return color.sqrt().toRGB();
    }
};









// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

pub const std_options = struct {
    pub const log_level = .info;
};

const default_input = "input.gltf";
const default_output = "output.png";

const CmdlineArgs = struct {
    in: [:0]const u8 = default_input,
    out: [:0]const u8 = default_output,
    width: ?u16 = null,
    height: ?u16 = null,

    fn init(allocator: std.mem.Allocator) !CmdlineArgs {
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

pub fn main() !void {
    const start_time = std.time.nanoTimestamp();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const args = try CmdlineArgs.init(allocator);
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
