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
    @cInclude("stb/stb_image.h");
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

fn getNodeTransform(node: *const c.cgltf_node) !Mat4 {
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
    w: usize,
    h: usize,
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
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,

    fn init(v0: Vec3, v1: Vec3, v2: Vec3) Triangle {
        return .{
            .v0 = v0,
            .e1 = subtract(v1, v0),
            .e2 = subtract(v2, v0),
        };
    }
};

const Hit = struct {
    t: f32,
    u: f32,
    v: f32,
    triangle_idx: usize,
};

const Vertex = struct {
    normal: Vec3,
    texcoord: [2]f32,
};

const TriangleData = struct {
    v: [3]Vertex,

    fn interpolate(self: TriangleData, comptime field_name: []const u8, u: f32, v: f32) @TypeOf(@field(self.v[0], field_name)) {
        const v0 = @field(self.v[0], field_name);
        const v1 = @field(self.v[1], field_name);
        const v2 = @field(self.v[2], field_name);
        switch (@TypeOf(@field(self.v[0], field_name))) {
            Vec3 => {
                return v0.scale(1-u-v).add(v1.scale(u)).add(v2.scale(v));
            },
            [2]f32 => {
                return .{
                    v0[0]*(1-u-v) + v1[0]*u + v2[0]*v,
                    v0[1]*(1-u-v) + v1[1]*u + v2[1]*v,
                };
            },
            else => {
                @compileError("Implement me");
            }
        }
    }
};

const Image = struct {
    data: [*]f32,
    w: f32,
    h: f32,
    pitch: usize,

    fn init(image: c.cgltf_image) Image {
        const buffer: [*]u8 = @as([*]u8, @ptrCast(image.buffer_view.*.buffer.*.data)) + image.buffer_view.*.offset;
        var w: c_int = 0;
        var h: c_int = 0;
        var channels: c_int = 0;
        const data = c.stbi_loadf_from_memory(buffer, @intCast(image.buffer_view.*.size), &w, &h, &channels, 4);
        return .{
            .data = data,
            .w = @floatFromInt(w),
            .h = @floatFromInt(h),
            .pitch = @intCast(w),
        };
    }

    fn deinit(self: Image) void {
        c.stbi_image_free(self.data);
    }

    fn sample(self: Image, u: f32, v: f32) Vec3 {
        // TODO: move me into sampler and implement filtering
        const x: usize = @intFromFloat(self.w * u);
        const y: usize = @intFromFloat(self.h * v);
        const pos = (y * self.pitch + x) * 4;
        const color = vec3(self.data[pos], self.data[pos+1], self.data[pos+2]);
        return color;
    }
};

const World = struct {
    triangles: []Triangle,
    triangles_data: []TriangleData,
    images: []Image,

    fn calcTriangles(gltf_data: *c.cgltf_data) usize {
        var result: usize = 0;
        for (gltf_data.nodes[0..gltf_data.nodes_count]) |node| {
            if (node.mesh != null) {
                for (node.mesh.*.primitives[0..node.mesh.*.primitives_count]) |primitive| {
                    result += primitive.indices.*.count / 3;
                }
            }
        }
        return result;
    }

    fn init(gltf_data: *c.cgltf_data, allocator: std.mem.Allocator) !World {
        const total_triangles_count = calcTriangles(gltf_data);
        std.log.info("Triangle count: {}", .{total_triangles_count});

        const triangles = try allocator.alloc(Triangle, total_triangles_count);
        errdefer allocator.free(triangles);
        const triangles_data = try allocator.alloc(TriangleData, total_triangles_count);
        errdefer allocator.free(triangles_data);

        var global_triangle_counter: usize = 0;

        for (gltf_data.nodes[0..gltf_data.nodes_count]) |node| {
            if (node.mesh != null) {
                for (node.mesh.*.primitives[0..node.mesh.*.primitives_count]) |primitive|
                {
                    std.debug.assert(primitive.type == c.cgltf_primitive_type_triangles);

                    const positions = Accessor(Vec3).init(try findPrimitiveAttribute(primitive, c.cgltf_attribute_type_position));
                    const normals = Accessor(Vec3).init(try findPrimitiveAttribute(primitive, c.cgltf_attribute_type_normal));
                    const texcoords = Accessor([2]f32).init(try findPrimitiveAttribute(primitive, c.cgltf_attribute_type_texcoord));
                    const indices = Accessor(u16).init(primitive.indices);

                    const triangles_count = indices.num() / 3;

                    const matrix = try getNodeTransform(&node);

                    for (0..triangles_count) |triangle_idx| {
                        var pos: [3]Vec3 = undefined;
                        for (0..3) |i| {
                            const index_idx = triangle_idx*3+i;
                            const vertex_idx = indices.at(index_idx);
                            pos[i] = matrix.transformPosition(positions.at(vertex_idx));
                            triangles_data[global_triangle_counter].v[i] = .{
                                .normal = matrix.transformDirection(normals.at(vertex_idx)).normalize(), // TODO: use adjusent matrix
                                .texcoord = texcoords.at(vertex_idx),
                            };
                        }
                        triangles[global_triangle_counter] = Triangle.init(pos[0], pos[1], pos[2]);
                        global_triangle_counter += 1;
                    }
                }
            }
        }

        const images = try allocator.alloc(Image, gltf_data.images_count);
        errdefer allocator.free(images);

        for (0..gltf_data.images_count) |i| {
            images[i] = Image.init(gltf_data.images[i]);
        }

        return .{
            .triangles = triangles,
            .triangles_data = triangles_data,
            .images = images,
        };
    }

    fn deinit(self: World, allocator: std.mem.Allocator) void {
        allocator.free(self.triangles);
        allocator.free(self.triangles_data);
        for (self.images) |*image| {
            image.deinit();
        }
        allocator.free(self.images);
    }

    fn rayTriangleIntersection(ray: Ray, v0: Vec3, e1: Vec3, e2: Vec3, triangle_idx: usize) ?Hit
    {
        const pvec = cross(ray.dir, e2);
        const det = dot(e1, pvec);

        const epsilon = 0.00000001; // TODO

        // // if the determinant is negative, the triangle is 'back facing'
        // // if the determinant is close to 0, the ray misses the triangle
        if (det < epsilon) return null;

        const inv_det = 1.0 / det;

        const tvec = subtract(ray.orig, v0);
        const u = dot(tvec, pvec) * inv_det;
        if (u < 0 or u > 1) return null;

        const qvec = cross(tvec, e1);
        const v = dot(ray.dir, qvec) * inv_det;
        if (v < 0 or u+v > 1) return null;

        const t = dot(e2, qvec) * inv_det;

        return .{
            .t = t,
            .u = u,
            .v = v,
            .triangle_idx = triangle_idx,
        };
    }

    fn getEnvColor(ray: Ray) Vec3 {
        const t = 0.5*(ray.dir.y()+1.0);
        return add(
            Vec3.ones().scale(1.0-t),
            Vec3.init(0.5, 0.7, 1.0).scale(t)
        );
    }

    fn render(world: World, batch: []Sample) void {
        inline for (0..BATCH_SIZE) |i| {
            batch[i].hit.t = std.math.inf(f32);
        }
        for (world.triangles, 0..) |triangle, triangle_idx|
        {
            inline for (0..BATCH_SIZE) |i| {
                const sample = &batch[i];
                const result = rayTriangleIntersection(sample.ray,
                    triangle.v0, triangle.e1, triangle.e2,
                    triangle_idx
                );
                if (result) |hit| {
                    if (sample.hit.t > hit.t and hit.t > 0) {
                        sample.hit = hit;
                    }
                }
            }
        }
        inline for (0..BATCH_SIZE) |i| {
            const sample = &batch[i];
            if (sample.hit.t < std.math.inf(f32)) {
                const triangle = world.triangles_data[sample.hit.triangle_idx];
                const texcoord = triangle.interpolate("texcoord", sample.hit.u, sample.hit.v);
                sample.color = world.images[0].sample(texcoord[0], texcoord[1]);
            }
            else {
                sample.color = getEnvColor(sample.ray);
            }
        }
    }
};

const BATCH_SIZE = 64;

const Sample = struct {
    ray: Ray,
    hit: Hit,
    color: Vec3,
};

const Scene = struct {
    camera: Camera,
    world: World,
    samples: []Sample,

    fn load(args: CmdlineArgs, allocator: std.mem.Allocator) !Scene {
        const options = std.mem.zeroes(c.cgltf_options);
        var gltf_data: ?*c.cgltf_data = null;
        try CGLTF_CHECK(c.cgltf_parse_file(&options, args.in.ptr, &gltf_data));
        defer c.cgltf_free(gltf_data);

        try CGLTF_CHECK(c.cgltf_load_buffers(&options, gltf_data, std.fs.path.dirname(args.in).?.ptr));

        const camera = try Camera.init(gltf_data.?, args.width, args.height);
        const world = try World.init(gltf_data.?, allocator);

        var samples = try allocator.alloc(Sample, std.mem.alignForward(usize, camera.w * camera.h, BATCH_SIZE));
        errdefer allocator.free(samples);

        for (0..camera.h) |y| {
            for (0..camera.w) |x| {
                samples[y*camera.w+x] = .{
                    .ray = camera.getRandomRay(@intCast(x), @intCast(y)),
                    .color = undefined,
                    .hit = undefined,
                };
            }
        }

        return .{
            .camera = camera,
            .world = world,
            .samples = samples,
        };
    }

    fn deinit(self: Scene, allocator: std.mem.Allocator) void {
        self.world.deinit(allocator);
        allocator.free(self.samples);
    }

    fn render(self: Scene) void {
        var begin: usize = 0;
        while (begin != self.samples.len) {
            const end = begin + BATCH_SIZE;
            const batch = self.samples[begin..end];
            self.world.render(batch);
            begin = end;
        }
    }

    fn getPixelColor(self: Scene, x: u16, y: u16) RGB
    {
        return self.samples[y*self.camera.w+x].color.sqrt().toRGB();
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

    c.stbi_set_flip_vertically_on_load(0);
    c.stbi_set_flip_vertically_on_load_thread(0);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const args = try CmdlineArgs.init(allocator);
    defer args.deinit(allocator);
    args.print();

    const scene = try Scene.load(args, allocator);
    defer scene.deinit(allocator);

    scene.render();

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
