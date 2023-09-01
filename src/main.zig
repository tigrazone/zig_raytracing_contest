const std = @import("std");
const builtin = @import("builtin");

const linalg = @import("linalg.zig");
const stage1 = @import("stage1.zig");

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

pub const Camera = struct {
    w: usize,
    h: usize,
    origin: Vec3,
    lower_left_corner: Vec3,
    right: Vec3,
    up: Vec3,

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

pub const Triangle = struct {
    pos: Pos,
    data: Data,

    pub const Pos = linalg.Triangle;

    pub const Data = struct {
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

pub const Texture = struct {
    data: []Vec3,
    w: f32,
    h: f32,
    w_int: usize,

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

pub const Material = struct {
    base_color: Texture,
    emissive: Texture,
};

pub const World = struct {
    pub const Cell = struct {
        first_triangle: u32,
        num_triangles: u32,
    };

    grid: Grid,
    cells: []Cell,
    triangles_pos: []Triangle.Pos,
    triangles_data: []Triangle.Data,
    materials: []Material,

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

pub const Scene = struct {
    camera: Camera,
    world: World,
    pixels: []Vec3,

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

fn getDuration(start_time: i128) @TypeOf(std.fmt.fmtDuration(1)) {
    const end_time = std.time.nanoTimestamp();
    return std.fmt.fmtDuration(@intCast(end_time - start_time));
}

pub const std_options = struct {
    pub const log_level = if (builtin.mode == .Debug) .debug else .info;
};

pub const CmdlineArgs = struct {
    in: []const u8 = "input.gltf",
    out: []const u8 = "output.png",
    width: ?u16 = null,
    height: ?u16 = null,
};

pub fn loadFile(path: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = try file.getEndPos();

    const buf = try allocator.alloc(u8, file_size);
    errdefer allocator.free(buf);

    const bytes_read = try file.readAll(buf);
    std.debug.assert(bytes_read == file_size);

    return buf;
}

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

pub var config: Config = undefined;

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
        var gltf = try stage1.loadGltfFile(allocator, args.options.in, threads);
        defer gltf.deinit();
        std.log.info("Loaded in {}", .{getDuration(loading_time)});

        const preprocessing_time = std.time.nanoTimestamp();
        const scene = try stage1.loadScene(gltf, args.options, arena.allocator());
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
