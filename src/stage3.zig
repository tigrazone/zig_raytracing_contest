
const std = @import("std");

const main = @import("main.zig");
const linalg = @import("linalg.zig");

const Vec3 = linalg.Vec3;
const Ray = linalg.Ray;
const Grid = linalg.Grid;
const RGB = linalg.RGB;

const add = Vec3.add;

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

pub fn Texture(comptime T: type) type {
    return struct {
        data: []const T,
        w: f32,
        h: f32,
        w_int: usize,
        h_int: usize,
        u_min: usize,
        u_max: usize,
        v_min: usize,
        v_max: usize,

        fn frac(v: f32) f32 {
            return @fabs(v - @trunc(v));
        }

        fn getPixel(self: @This(), x: usize, y: usize) T {
            const pos = y * self.w_int + x;
            return self.data[pos];
        }

        fn lerp(a: T, b: T, x: f32) T {
            if (T == Vec3) {
                return Vec3.lerp(a, b, Vec3.splat(x));
            } else if (T == f32) {
                return std.math.lerp(a, b, x);
            }
        }

        fn sample(self: @This(), u: f32, v: f32) T {
            const ui: usize = @intFromFloat(@floor(self.w * u));
            const vi: usize = @intFromFloat(@floor(self.h * v));
            const x1: usize = std.math.clamp(ui, self.u_min, self.u_max) % self.w_int;
            const y1: usize = std.math.clamp(vi, self.v_min, self.v_max) % self.h_int;
            const x2: usize = std.math.clamp(ui + 1, self.u_min, self.u_max) % self.w_int;
            const y2: usize = std.math.clamp(vi + 1, self.v_min, self.v_max) % self.h_int;
            const r1 = lerp(self.getPixel(x1, y1), self.getPixel(x2, y1), frac(u));
            const r2 = lerp(self.getPixel(x1, y2), self.getPixel(x2, y2), frac(u));
            return lerp(r1, r2, frac(v));
        }
    };
}

pub const Material = struct {
    base_color: Texture(Vec3),
    emissive: Texture(Vec3),
    transparency: Texture(f32),
};

pub const Cell = struct {
    begin: u32,
    end: u32,
};

pub const Scene = struct {
    grid: Grid,
    cells: []Cell,
    triangles_pos: []Triangle.Pos,
    triangles_data: []Triangle.Data,
    materials: []Material,
    arena: std.heap.ArenaAllocator,

    fn getEnvColor(ray: Ray) Vec3 {
        const t = 0.5*(ray.dir.y()+1.0);
        return add(
            Vec3.ones().scale(1.0-t),
            Vec3.init(0.5, 0.7, 1.0).scale(t)
        );
    }

    fn traceRay(scene: Scene, ray: Ray, ignore_triangle: usize) Hit {
        var nearest_hit = Hit{
            .t = std.math.inf(f32),
            .u = undefined,
            .v = undefined,
            .triangle_idx = undefined,
        };
        var tmp = scene.grid.traceRay(ray);
        if (tmp) |*grid_it| {
            while (true) {
                const cell_idx = scene.grid.linearlizeCellIdx(grid_it.cell[0], grid_it.cell[1], grid_it.cell[2]);
                const cell = scene.cells[cell_idx];
                for (cell.begin..cell.end) |triangle_idx|
                {
                    const triangle = scene.triangles_pos[triangle_idx];
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

    fn traceRayRecursive(scene: Scene, ray: Ray, depth: u16, ignore_triangle: usize, random: std.rand.Random) Vec3 {
        if (depth == 0) {
            return Vec3.zeroes();
        }

        const hit = scene.traceRay(ray, ignore_triangle);

        if (hit.t == std.math.inf(f32)) {
            return getEnvColor(ray);
        }

        const triangle = scene.triangles_data[hit.triangle_idx];
        const material = scene.materials[triangle.material_idx];

        const texcoord = triangle.interpolate("texcoord", hit.u, hit.v); // TODO: get texcoord channel from Texture
        const albedo = material.base_color.sample(texcoord[0], texcoord[1]);
        const emissive = material.emissive.sample(texcoord[0], texcoord[1]);
        const transparency = material.transparency.sample(texcoord[0], texcoord[1]);
        const normal = triangle.interpolate("normal", hit.u, hit.v);
        if (random.float(f32) > transparency) {
            const new_ray = .{
                .orig = ray.at(hit.t),
                .dir = ray.dir, // TODO!!!!
            };
            return scene.traceRayRecursive(new_ray, depth-1, hit.triangle_idx, random);
        }
        const scattered_dir = normal.add(Vec3.randomUnitVector(random)).normalize();
        const new_ray = .{
            .orig = ray.at(hit.t),
            .dir = scattered_dir,
        };
        return emissive.add(albedo.mul(scene.traceRayRecursive(new_ray, depth-1, hit.triangle_idx, random)));
    }

    fn renderWorker(self: Scene, thread_idx: usize, thread_num: usize, camera: Camera, img: []RGB) void {
        const inv_num_samples = Vec3.ones().div(Vec3.splat(@floatFromInt(main.config.num_samples)));

        var prng = std.rand.DefaultPrng.init(thread_idx);
        const random = prng.random();

        const pixels_per_thread = (img.len + thread_num - 1) / thread_num;
        var i = pixels_per_thread * thread_idx;
        for (0..pixels_per_thread) |_| {
            if (i >= img.len) {
                break;
            }
            const x: f32 = @floatFromInt(@mod(i, camera.w));
            const y: f32 = @floatFromInt(i / camera.w);
            var pixel = Vec3.zeroes();
            for (0..main.config.num_samples) |_| {
                const ray = camera.getRay(x + random.float(f32), y + random.float(f32));
                const ray_color = self.traceRayRecursive(ray, main.config.max_bounce, std.math.maxInt(usize), random);
                pixel = pixel.add(ray_color);
            }
            img[i] = pixel.mul(inv_num_samples).toRGB();
            i += 1;
        }
    }

    pub fn render(self: Scene, threads: []std.Thread, camera: Camera, img: []RGB) !void {
        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, renderWorker, .{
                self, i, threads.len, camera, img
            });
        }
        for (threads) |*thread| {
            thread.join();
        }
    }

    pub fn init(allocator: std.mem.Allocator) Scene {
        var scene: Scene = undefined;
        scene.arena = std.heap.ArenaAllocator.init(allocator);
        return scene;
    }

    pub fn deinit(self: *Scene) void {
        self.arena.deinit();
    }
};
