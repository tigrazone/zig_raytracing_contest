const std = @import("std");
const builtin = @import("builtin");

const linalg = @import("linalg.zig");

const Mat4 = linalg.Mat4;
const Vec3 = linalg.Vec3;
const Vec3u = linalg.Vec3u;
const Ray = linalg.Ray;
const Bbox = linalg.Bbox;
const Grid = linalg.Grid;

const vec3 = Vec3.init;
const vec3u = Vec3u.init;
const dot = Vec3.dot;
const cross = Vec3.cross;
const subtract = Vec3.subtract;
const add = Vec3.add;

const Gltf = @import("zgltf");
const zigimg = @import("zigimg");
const zigargs = @import("zigargs");

test {
    _ = @import("linalg.zig");
}





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

    fn getRay(self: Camera, x: f32, y: f32) Ray {
        return .{
            .orig = self.origin,
            .dir = self.lower_left_corner
                .add(self.right.scale(x))
                .add(self.up.scale(y))
                .normalize()
        };
    }
};

const Triangle = struct {
    pos: Pos,
    data: Data,

    const Pos = linalg.Triangle;

    const Data = struct {
        v: [3]Vertex,
        material_idx: usize,

        const Vertex = struct {
            normal: Vec3,
            texcoord: [2]f32,
        };

        fn interpolate(self: Data, comptime field_name: []const u8, u: f32, v: f32) @TypeOf(@field(self.v[0], field_name)) {
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
};

const Hit = struct {
    t: f32,
    u: f32,
    v: f32,
    triangle_idx: usize,
};

const Texture = struct {
    data: []Vec3,
    w: f32,
    h: f32,
    w_int: usize,

    fn initColor(allocator: std.mem.Allocator, gltf: Gltf, texture_info: ?Gltf.TextureInfo, _factor: []const f32) !Texture {
        const factor = vec3(_factor[0], _factor[1], _factor[2]);
        if (texture_info) |info| {
            const texture = gltf.data.textures.items[info.index];
            const image = gltf.data.images.items[texture.source.?];
            const im: *const zigimg.Image = @alignCast(@ptrCast(image.data.?));
            const data = try allocator.alloc(Vec3, im.width*im.height);
            var iter = im.iterator();
            while (iter.next()) |pixel| {
                data[iter.current_index-1] = vec3(pixel.r, pixel.g, pixel.b).mul(factor);
            }
            return .{
                .data = data,
                .w = @floatFromInt(im.width),
                .h = @floatFromInt(im.height),
                .w_int = im.width,
            };
        } else {
            const data = try allocator.alloc(Vec3, 1);
            data[0] = factor;
            return .{
                .data = data,
                .w = 1,
                .h = 1,
                .w_int = 1,
            };
        }
    }

    fn frac(v: f32) f32 {
        return @fabs(v - @trunc(v));
    }

    fn sample(self: Texture, u: f32, v: f32) Vec3 {
        // TODO: move me into sampler and implement filtering
        const x: usize = @intFromFloat(self.w * frac(u));
        const y: usize = @intFromFloat(self.h * frac(v));
        const pos = y * self.w_int + x;
        return self.data[pos];
    }
};

const Material = struct {
    base_color: Texture,
    emissive: Texture,

    fn init(allocator: std.mem.Allocator, gltf: Gltf, material_idx: Gltf.Index) !Material {
        const material = gltf.data.materials.items[material_idx];
        return .{
            .base_color = try Texture.initColor(allocator, gltf,
                material.metallic_roughness.base_color_texture,
                &material.metallic_roughness.base_color_factor),
            .emissive = try Texture.initColor(allocator, gltf,
                material.emissive_texture,
                &material.emissive_factor), // TODO: material.emissive_strength
        };
    }
};

const World = struct {
    const Cell = struct {
        first_triangle: u32,
        num_triangles: u32,
    };

    grid: Grid,
    cells: []Cell,
    triangles_pos: []Triangle.Pos,
    triangles_data: []Triangle.Data,
    materials: []Material,

    fn initGrid(gltf: Gltf, grid: *Grid) !usize {
        var unique_triangles: usize = 0;
        var mean_triangle_size = Vec3.zeroes();
        var bbox: Bbox = .{};

        for (gltf.data.nodes.items) |node| {
            if (node.mesh != null) {
                const mesh = gltf.data.meshes.items[node.mesh.?];
                for (mesh.primitives.items) |primitive| {
                    const positions = Accessor(Vec3).init(gltf, try findPrimitiveAttribute(primitive, .position));
                    const indices = Accessor(u16).init(gltf, primitive.indices);

                    const triangles_count = indices.num / 3;

                    const matrix = Mat4{.data = gltf.getGlobalTransform(node)};

                    for (0..triangles_count) |triangle_idx| {
                        var triangle_bbox: Bbox = .{};
                        for (0..3) |i| {
                            const index_idx = triangle_idx*3+i;
                            const vertex_idx = indices.at(index_idx);
                            const pos = matrix.transformPosition(positions.at(vertex_idx));
                            triangle_bbox.extendBy(pos);
                        }
                        mean_triangle_size = Vec3.add(mean_triangle_size, triangle_bbox.size());
                        bbox.unionWith(triangle_bbox);
                    }
                    unique_triangles += triangles_count;
                }
            }
        }

        mean_triangle_size = mean_triangle_size.div(Vec3.fromScalar(@floatFromInt(unique_triangles)));
        const mean_triangle_count = bbox.size().div(mean_triangle_size);

        std.log.info("Mean triangle count: {d:.1}", .{mean_triangle_count.data});
        const resolution = config.grid_resolution orelse mean_triangle_count.div(Vec3.fromScalar(4)).ceil().toInt(u32);
        std.log.info("Grid resolution: {}", .{resolution.data});

        grid.* = Grid.init(bbox, resolution);

        return unique_triangles;
    }

    fn initCells(gltf: Gltf, cells: []Cell, grid: Grid) !usize {
        @memset(cells, .{.first_triangle = 0, .num_triangles = 0});
        for (gltf.data.nodes.items) |node| {
            if (node.mesh != null) {
                const mesh = gltf.data.meshes.items[node.mesh.?];
                for (mesh.primitives.items) |primitive| {
                    const positions = Accessor(Vec3).init(gltf, try findPrimitiveAttribute(primitive, .position));
                    const indices = Accessor(u16).init(gltf, primitive.indices);

                    const triangles_count = indices.num / 3;

                    const matrix = Mat4{.data = gltf.getGlobalTransform(node)};

                    for (0..triangles_count) |triangle_idx| {
                        var pos: [3]Vec3 = undefined;
                        for (0..3) |i| {
                            const index_idx = triangle_idx*3+i;
                            const vertex_idx = indices.at(index_idx);
                            pos[i] = matrix.transformPosition(positions.at(vertex_idx));
                        }
                        const min = grid.getCellPos(Vec3.min(pos[0], Vec3.min(pos[1], pos[2])));
                        const max = grid.getCellPos(Vec3.max(pos[0], Vec3.max(pos[1], pos[2])));

                        for (min.z()..max.z()+1) |z| {
                            for (min.y()..max.y()+1) |y| {
                                for (min.x()..max.x()+1) |x| {
                                    const index = grid.getCellIdx(x,y,z);
                                    cells[index].num_triangles += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        var min_triangles: u32 = std.math.maxInt(u32);
        var max_triangles: u32 = 0;
        var empty_cells: u32 = 0;
        var total_triangles_count: u32 = 0;
        for (cells) |*cell| {
            if (cell.num_triangles != 0) {
                min_triangles = @min(min_triangles, cell.num_triangles);
                max_triangles = @max(max_triangles, cell.num_triangles);
            } else {
                empty_cells += 1;
            }
            cell.first_triangle = total_triangles_count;
            total_triangles_count += cell.num_triangles;
            cell.num_triangles = 0;
        }
        const mean_triangles = total_triangles_count / (grid.numCells() - empty_cells);
        std.log.info("Empty cells: {}/{} ({d:.2}%) min triangles: {} max triangles: {} mean_triangles: {}",
            .{empty_cells, grid.numCells(),
                @as(f32, @floatFromInt(empty_cells)) / @as(f32, @floatFromInt(grid.numCells())) * 100,
                min_triangles, max_triangles, mean_triangles});
        return total_triangles_count;
    }

    fn initTriangles(gltf: Gltf, triangles: *std.MultiArrayList(Triangle), cells: []Cell, grid: Grid) !void {
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
                        var data = Triangle.Data {
                            .v = undefined,
                            .material_idx = primitive.material.?,
                        };
                        for (0..3) |i| {
                            const index_idx = triangle_idx*3+i;
                            const vertex_idx = indices.at(index_idx);
                            pos[i] = matrix.transformPosition(positions.at(vertex_idx));
                            data.v[i] = .{
                                .normal = matrix.transformDirection(normals.at(vertex_idx)).normalize(), // TODO: use adjusent matrix
                                .texcoord = texcoords.at(vertex_idx),
                            };
                        }
                        const triangle = Triangle{
                            .pos = Triangle.Pos.init(pos[0], pos[1], pos[2]),
                            .data = data,
                        };
                        const min = grid.getCellPos(Vec3.min(pos[0], Vec3.min(pos[1], pos[2])));
                        const max = grid.getCellPos(Vec3.max(pos[0], Vec3.max(pos[1], pos[2])));

                        for (min.z()..max.z()+1) |z| {
                            for (min.y()..max.y()+1) |y| {
                                for (min.x()..max.x()+1) |x| {
                                    const index = grid.getCellIdx(x,y,z);
                                    var cell = &cells[index];
                                    triangles.set(cell.first_triangle + cell.num_triangles, triangle);
                                    cell.num_triangles += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn init(gltf: Gltf, arena_allocator: std.mem.Allocator) !World {
        var grid: Grid = undefined;
        const unique_triangles = try initGrid(gltf, &grid);

        const cells = try arena_allocator.alloc(Cell, grid.resolution.reduceMul());
        const total_triangles_count = try initCells(gltf, cells, grid);

        var triangles = std.MultiArrayList(Triangle){};
        try triangles.resize(arena_allocator, total_triangles_count);
        try initTriangles(gltf, &triangles, cells, grid);

        const materials = try arena_allocator.alloc(Material, gltf.data.materials.items.len);
        for (materials, 0..) |*material, i| {
            material.* = try Material.init(arena_allocator, gltf, i);
        }

        std.log.info("Unique triangle count: {}/{} ({d:.2}%)",
            .{unique_triangles, total_triangles_count,
                @as(f32, @floatFromInt(unique_triangles)) / @as(f32, @floatFromInt(total_triangles_count)) * 100});
        std.log.info("Materials count: {}", .{materials.len});

        return .{
            .grid = grid,
            .cells = cells,
            .triangles_pos = triangles.items(.pos),
            .triangles_data = triangles.items(.data),
            .materials = materials,
        };
    }

    fn getEnvColor(ray: Ray) Vec3 {
        const t = 0.5*(ray.dir.y()+1.0);
        return add(
            Vec3.ones().scale(1.0-t),
            Vec3.init(0.5, 0.7, 1.0).scale(t)
        );
    }

    fn traceRay(world: World, ray: Ray, ignore_triangle: usize) Hit {
        var nearest_hit = Hit{
            .t = std.math.inf(f32),
            .u = undefined,
            .v = undefined,
            .triangle_idx = undefined,
        };
        var tmp = world.grid.traceRay(ray);
        if (tmp) |*grid_it| {
            while (true) {
                const cell_idx = world.grid.getCellIdx(grid_it.cell[0], grid_it.cell[1], grid_it.cell[2]);
                const cell = world.cells[cell_idx];
                const begin = cell.first_triangle;
                const end = begin + cell.num_triangles;
                for (begin..end) |triangle_idx|
                {
                    const triangle = world.triangles_pos[triangle_idx];
                    var hit = Hit{
                        .t = undefined,
                        .u = undefined,
                        .v = undefined,
                        .triangle_idx = triangle_idx
                    };
                    if (triangle.rayIntersection(ray, &hit.t, &hit.u, &hit.v)) {
                        if (nearest_hit.t > hit.t and hit.t > 0 and triangle_idx != ignore_triangle) {
                            nearest_hit = hit;
                        }
                    }
                }
                const t_next_crossing = grid_it.next();
                if (nearest_hit.t <= t_next_crossing) {
                    break;
                }
            }
        }
        return nearest_hit;
    }

    fn traceRayRecursive(world: World, ray: Ray, depth: u16, ignore_triangle: usize) Vec3 {
        if (depth == 0) {
            return Vec3.zeroes();
        }

        const hit = world.traceRay(ray, ignore_triangle);

        if (hit.t == std.math.inf(f32)) {
            return getEnvColor(ray);
        }

        const triangle = world.triangles_data[hit.triangle_idx];
        const material = world.materials[triangle.material_idx];

        const texcoord = triangle.interpolate("texcoord", hit.u, hit.v); // TODO: get texcoord channel from Texture
        const albedo = material.base_color.sample(texcoord[0], texcoord[1]);
        const emissive = material.emissive.sample(texcoord[0], texcoord[1]);
        const normal = triangle.interpolate("normal", hit.u, hit.v);
        const scattered_dir = normal.add(Vec3.randomUnitVector()).normalize();
        const new_ray = .{
            .orig = ray.at(hit.t),
            .dir = scattered_dir,
        };
        return emissive.add(albedo.mul(world.traceRayRecursive(new_ray, depth-1, hit.triangle_idx)));
    }
};

const Scene = struct {
    camera: Camera,
    world: World,
    pixels: []Vec3,

    fn load(gltf: Gltf, args: CmdlineArgs, arena_allocator: std.mem.Allocator) !Scene {
        const camera = try Camera.init(gltf, args.width, args.height);
        const world = try World.init(gltf, arena_allocator);

        var pixels = try arena_allocator.alloc(Vec3, camera.w * camera.h);
        errdefer arena_allocator.free(pixels);

        return .{
            .camera = camera,
            .world = world,
            .pixels = pixels,
        };
    }

    fn renderWorker(self: Scene, thread_idx: usize, thread_num: usize) void {
        const inv_num_samples = Vec3.ones().div(Vec3.fromScalar(@floatFromInt(config.num_samples)));

        var prng = std.rand.DefaultPrng.init(thread_idx);
        const random = prng.random();

        const pixels_per_thread = (self.pixels.len + thread_num - 1) / thread_num;
        var i = pixels_per_thread * thread_idx;
        for (0..pixels_per_thread) |_| {
            if (i >= self.pixels.len) {
                break;
            }
            const x: f32 = @floatFromInt(@mod(i, self.camera.w));
            const y: f32 = @floatFromInt(i / self.camera.w);
            var pixel = Vec3.zeroes();
            for (0..config.num_samples) |_| {
                const ray = self.camera.getRay(x + random.float(f32), y + random.float(f32));
                const ray_color = self.world.traceRayRecursive(ray, config.max_bounce, std.math.maxInt(usize));
                pixel = pixel.add(ray_color);
            }
            self.pixels[i] = pixel.mul(inv_num_samples);
            i += 1;
        }
    }

    fn render(self: Scene, threads: []std.Thread) !void {
        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, renderWorker, .{
                self, i, threads.len
            });
        }
        for (threads) |*thread| {
            thread.join();
        }
    }

    fn getPixelColor(self: Scene, x: u16, y: u16) zigimg.color.Rgb24
    {
        return self.pixels[y*self.camera.w+x].toRGB();
    }
};









// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

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

fn loadGltfImages(tmp_allocator: std.mem.Allocator, arena_allocator: std.mem.Allocator,
    gltf_dir: []const u8, gltf: *Gltf,
    thread_idx: usize, thread_num: usize) !void
{
    var image_idx = thread_idx;
    while (image_idx < gltf.data.images.items.len) : (image_idx += thread_num) {
        const image = &gltf.data.images.items[image_idx];
        const im = blk: {
            if (image.buffer_view != null) {
                const buffer_view = gltf.data.buffer_views.items[image.buffer_view.?];
                const buffer = gltf.data.buffers.items[buffer_view.buffer];
                const begin = buffer_view.byte_offset;
                const end = begin + buffer_view.byte_length;
                break :blk try zigimg.Image.fromMemory(arena_allocator, buffer.data.?[begin..end]);
            } else {
                var tmp = [_]u8{undefined} ** 256;
                const img_path = try std.fmt.bufPrintZ(&tmp, "{s}/{s}", .{gltf_dir, image.uri.?});
                std.log.debug("Loading image {s}", .{img_path});
                const data = try loadFile(img_path, tmp_allocator);
                defer tmp_allocator.free(data);
                break :blk try zigimg.Image.fromMemory(arena_allocator, data);
            }
        };
        const ptr = try arena_allocator.create(zigimg.Image);
        ptr.* = im;
        image.data = @ptrCast(ptr);
    }
}

fn loadGltf(_allocator: std.mem.Allocator, path: []const u8, threads: []std.Thread) !Gltf {
    const gltf_dir = std.fs.path.dirname(path).?;

    var gltf = Gltf.init(_allocator);
    errdefer gltf.deinit();

    std.log.debug("Loading scene {s}", .{path});
    const arena_allocator = gltf.arena.allocator();
    const buf = try loadFile(path, arena_allocator);
    try gltf.parse(buf);

    for (gltf.data.buffers.items, 0..) |*buffer, i| {
        if (i == 0 and buffer.uri == null) {
            buffer.data = gltf.glb_binary;
            continue;
        }
        var tmp = [_]u8{undefined} ** 256;
        const buf_path = try std.fmt.bufPrint(&tmp, "{s}/{s}", .{gltf_dir, buffer.uri.?});
        std.log.debug("Loading buffer {s}", .{buf_path});
        buffer.data = try loadFile(buf_path, arena_allocator);
    }

    var safe_tmp_allocator = std.heap.ThreadSafeAllocator{.child_allocator = _allocator};
    var safe_arena_allocator = std.heap.ThreadSafeAllocator{.child_allocator = arena_allocator};
    for (threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, loadGltfImages, .{
            safe_tmp_allocator.allocator(), safe_arena_allocator.allocator(),
            gltf_dir, &gltf,
            i, threads.len
        });
    }
    for (threads) |*thread| {
        thread.join();
    }

    return gltf;
}







// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

fn getDuration(start_time: i128) @TypeOf(std.fmt.fmtDuration(1)) {
    const end_time = std.time.nanoTimestamp();
    return std.fmt.fmtDuration(@intCast(end_time - start_time));
}

pub const std_options = struct {
    pub const log_level = if (builtin.mode == .Debug) .debug else .info;
};

const CmdlineArgs = struct {
    in: []const u8 = "input.gltf",
    out: []const u8 = "output.png",
    width: ?u16 = null,
    height: ?u16 = null,
};

const Config = struct {
    grid_resolution: ?Vec3u,
    num_threads: ?u8,
    num_samples: u16,
    max_bounce: u16,

    fn load(path: []const u8, allocator: std.mem.Allocator) !Config {
        const buf = try loadFile(path, allocator);
        defer allocator.free(buf);
        var parsed_str = try std.json.parseFromSlice(Config, allocator, buf, .{});
        defer parsed_str.deinit();
        return parsed_str.value;
    }
};

var config: Config = undefined;

pub fn main() !void {
    std.log.info("{}", .{std_options.log_level});

    const start_time = std.time.nanoTimestamp();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();

    const args = try zigargs.parseForCurrentProcess(CmdlineArgs, allocator, .print);
    defer args.deinit();

    config = try Config.load("config.json", allocator);
    std.log.info("Num samples: {}, max bounce {}", .{config.num_samples, config.max_bounce});

    const num_threads = config.num_threads orelse try std.Thread.getCpuCount();
    const threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);
    std.log.info("Num threads: {}", .{num_threads});

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const scene = blk: {
        const loading_time = std.time.nanoTimestamp();
        var gltf = try loadGltf(allocator, args.options.in, threads);
        defer gltf.deinit();
        std.log.info("Loaded in {}", .{getDuration(loading_time)});

        const preprocessing_time = std.time.nanoTimestamp();
        const scene = try Scene.load(gltf, args.options, arena.allocator());
        std.log.info("Preprocessed in {}", .{getDuration(preprocessing_time)});

        break :blk scene;
    };

    const render_time = std.time.nanoTimestamp();
    try scene.render(threads);
    std.log.info("Rendered in {}", .{getDuration(render_time)});

    const w = scene.camera.w;
    const h = scene.camera.h;

    var img = try zigimg.Image.create(allocator, w, h, .rgb24);
    defer img.deinit();

    const resolve_time = std.time.nanoTimestamp();
    for (0..h) |y| {
        for (0..w) |x| {
            const row = h - 1 - y;
            const column = w - 1 - x;
            img.pixels.rgb24[row*w+column] = scene.getPixelColor(@intCast(x), @intCast(y));
        }
    }
    std.log.info("Resolved in {}", .{getDuration(resolve_time)});

    const save_time = std.time.nanoTimestamp();
    try img.writeToFilePath(args.options.out, .{.png = .{}});
    std.log.info("Saved in {}", .{getDuration(save_time)});

    std.log.info("Done in {}", .{getDuration(start_time)});
}
