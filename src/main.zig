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
});

const Gltf = @import("zgltf");

// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================


fn findCameraNode(gltf: Gltf) !Gltf.Node {
    for (gltf.data.nodes.items) |node| {
        if (node.camera != null) {
            return node;
        }
    }
    return error.CameraNodeNotFound; // TODO: implement recursive search / deal with multiple instances of the same camera
}

fn findPrimitiveAttribute(primitive: Gltf.Primitive, comptime tag: std.meta.Tag(Gltf.Attribute)) !?Gltf.Index {
    for (primitive.attributes.items) |attribute| {
        if (attribute == tag) {
            return @field(attribute, @tagName(tag));
        }
    }
    return error.AttributeNotFound;
}

fn Accessor(comptime T: type) type {
    return struct {
        const Self = @This();

        base_address: [*]const u8,
        num: usize,
        stride: usize,

        fn init(gltf: Gltf, accessor_idx: ?Gltf.Index) Self
        {
            const accessor = gltf.data.accessors.items[accessor_idx.?];
            switch (T) {
                Vec3 => {
                    std.debug.assert(accessor.type == .vec3);
                    std.debug.assert(accessor.component_type == .float);
                },
                [2]f32 => {
                    std.debug.assert(accessor.type == .vec2);
                    std.debug.assert(accessor.component_type == .float);
                },
                u16 => {
                    std.debug.assert(accessor.type == .scalar);
                    std.debug.assert(accessor.component_type == .unsigned_short);
                },
                else => {
                    @compileError("Implement me");
                }
            }

            const buffer_view = gltf.data.buffer_views.items[accessor.buffer_view.?];
            const buffer = gltf.data.buffers.items[buffer_view.buffer];

            return .{
                .base_address =
                    buffer.data.?.ptr +
                    buffer_view.byte_offset +
                    accessor.byte_offset,
                .num = accessor.count,
                .stride = accessor.stride,
            };
        }

        fn at(self: Self, idx: usize) T {
            switch (T) {
                Vec3 => {
                    const ptr: [*]const f32 = @ptrCast(@alignCast(self.base_address + idx*self.stride));
                    return vec3(ptr[0],ptr[1],ptr[2]);
                },
                [2]f32 => {
                    const ptr: [*]const f32 = @ptrCast(@alignCast(self.base_address + idx*self.stride));
                    return ptr[0..2].*;
                },
                u16 => {
                    const ptr: [*]const u16 = @ptrCast(@alignCast(self.base_address + idx*self.stride));
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

    fn init(gltf: Gltf, width: ?u16, height: ?u16) !Camera {

        const camera_node = try findCameraNode(gltf);

        const camera = gltf.data.cameras.items[camera_node.camera.?];
        if (camera.type != .perspective) {
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
            if (camera.type.perspective.aspect_ratio != null) {
                return error.CameraHasAspectRatio;
            }
            w = width.?;
            h = height.?;
        }
        else
        {
            if (camera.type.perspective.aspect_ratio == null) {
                return error.CameraHasntAspectRatio;
            }
            const aspect_ratio = camera.type.perspective.aspect_ratio.?;
            w = width orelse @intFromFloat(@as(f32, @floatFromInt(height.?)) * aspect_ratio);
            h = height orelse @intFromFloat(@as(f32, @floatFromInt(width.?)) / aspect_ratio);
        }

        std.log.info("Pixels count: {}", .{w*h});

        const f_w: f32 = @floatFromInt(w);
        const f_h: f32 = @floatFromInt(h);

        const matrix = Mat4{.data = gltf.getGlobalTransform(camera_node)};
        const origin = matrix.col3(3);
        const fwd = matrix.col3(2).scale(-1).normalize();
        const world_up = vec3(0,1,0);

        const right = cross(world_up, fwd).normalize();
        const up = cross(fwd, right);

        const focal_length = (f_h / 2) / @tan(camera.type.perspective.yfov / 2);

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

    fn init(gltf: Gltf, image: Gltf.Image) Image {
        const buffer_view = gltf.data.buffer_views.items[image.buffer_view.?];
        const buffer = gltf.data.buffers.items[buffer_view.buffer];
        var w: c_int = 0;
        var h: c_int = 0;
        var channels: c_int = 0;
        const data = c.stbi_loadf_from_memory(
            buffer.data.?.ptr + buffer_view.byte_offset,
            @intCast(buffer_view.byte_length),
            &w, &h, &channels, 4);
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

    fn calcTriangles(gltf: Gltf) usize {
        var result: usize = 0;
        for (gltf.data.nodes.items) |node| {
            if (node.mesh != null) {
                const mesh = gltf.data.meshes.items[node.mesh.?];
                for (mesh.primitives.items) |primitive| {
                    const accessor = gltf.data.accessors.items[primitive.indices.?];
                    result += accessor.count / 3;
                }
            }
        }
        return result;
    }

    fn init(gltf: Gltf, allocator: std.mem.Allocator) !World {
        const total_triangles_count = calcTriangles(gltf);
        std.log.info("Triangle count: {}", .{total_triangles_count});

        const triangles = try allocator.alloc(Triangle, total_triangles_count);
        errdefer allocator.free(triangles);
        const triangles_data = try allocator.alloc(TriangleData, total_triangles_count);
        errdefer allocator.free(triangles_data);

        var global_triangle_counter: usize = 0;

        for (gltf.data.nodes.items) |node| {
            if (node.mesh != null) {
                const mesh = gltf.data.meshes.items[node.mesh.?];
                for (mesh.primitives.items) |primitive|
                {
                    std.debug.assert(primitive.mode == .triangles);

                    const positions = Accessor(Vec3).init(gltf, try findPrimitiveAttribute(primitive, .position));
                    const normals = Accessor(Vec3).init(gltf, try findPrimitiveAttribute(primitive, .normal));
                    const texcoords = Accessor([2]f32).init(gltf, try findPrimitiveAttribute(primitive, .texcoord));
                    const indices = Accessor(u16).init(gltf, primitive.indices);

                    const triangles_count = indices.num / 3;

                    const matrix = Mat4{.data = gltf.getGlobalTransform(node)};

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

        const images = try allocator.alloc(Image, gltf.data.images.items.len);
        errdefer allocator.free(images);

        for (gltf.data.images.items, 0..) |image, i| {
            images[i] = Image.init(gltf, image);
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

fn loadFile(path: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = try file.getEndPos();

    const buf = try allocator.alloc(u8, file_size);
    errdefer allocator.free(buf);

    const bytes_read = try file.readAll(buf);
    std.debug.assert(bytes_read == file_size);

    return buf;
}

const Scene = struct {
    camera: Camera,
    world: World,
    samples: []Sample,

    fn load(args: CmdlineArgs, allocator: std.mem.Allocator) !Scene {
        var gltf = Gltf.init(allocator);
        defer gltf.deinit();

        const buf = try loadFile(args.in, allocator);
        defer allocator.free(buf);

        try gltf.parse(buf);

        for (gltf.data.buffers.items, 0..) |*buffer, i| {
            if (i == 0 and buffer.uri == null) {
                buffer.data = gltf.glb_binary;
            }
        }

        const camera = try Camera.init(gltf, args.width, args.height);
        const world = try World.init(gltf, allocator);

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
