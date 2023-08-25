const std = @import("std");
const zigimg = @import("zigimg");

pub fn Vec(comptime size: usize, comptime T: type) type {
    return struct {
        const Self = @This();
        const Data = @Vector(size, T);

        data: Data,

        pub fn x(self: Self) T {
            return self.data[0];
        }
        pub fn y(self: Self) T {
            return self.data[1];
        }
        pub fn z(self: Self) T {
            return self.data[2];
        }

        pub fn zeroes() Self {
            return .{ .data = .{0, 0, 0}};
        }

        pub fn ones() Self {
            return .{ .data = .{1, 1, 1}};
        }

        pub fn init(_x: T, _y: T, _z: T) Self {
            return .{ .data = .{_x, _y, _z}};
        }

        pub fn fromArray(array: [size]T) Self {
            return .{ .data = array};
        }

        pub fn fromScalar(s: T) Self {
            return .{ .data = .{s, s, s}};
        }

        pub fn max(a: Self, b: Self) Self {
            return .{ .data = @max(a.data, b.data) };
        }

        pub fn min(a: Self, b: Self) Self {
            return .{ .data = @min(a.data, b.data) };
        }

        pub fn clamp(self: Self, min_value: T, max_value: T) Self {
            return min(self, max(fromScalar(min_value), fromScalar(max_value)));
        }

        pub fn scale(self: Self, s: T) Self {
            return .{ .data = self.data * @as(Data, @splat(s)) };
        }

        pub usingnamespace if (T == f32 and size == 3) struct {
            pub fn toRGB(self: Self) zigimg.color.Rgb24 {
                const rgb = self.clamp(0.0, 0.999999).scale(256);
                return .{
                    .r = @intFromFloat(rgb.data[0]),
                    .g = @intFromFloat(rgb.data[1]),
                    .b = @intFromFloat(rgb.data[2]),
                };
            }
            pub fn toInt(self: Self, comptime U: type) Vec(size, U) {
                return .{
                    .data = .{
                        @as(U, @intFromFloat(self.data[0])),
                        @as(U, @intFromFloat(self.data[1])),
                        @as(U, @intFromFloat(self.data[2])),
                    }
                };
            }
            pub fn length(self: Self) T {
                return @sqrt(@reduce(.Add, self.data * self.data));
            }

            pub fn normalize(self: Self) Self {
                return self.scale(1.0 / self.length());
            }
        } else struct {};

        pub usingnamespace if (T == bool) struct {
            pub fn select(self: Self, a: anytype, b: anytype) @TypeOf(a, b) {
                const Result = @TypeOf(a, b);
                const Elem = std.meta.Child(std.meta.FieldType(Result, .data));
                return .{.data = @select(Elem, self.data, a.data, b.data)};
            }
        } else struct {};

        pub usingnamespace if (size == 3) struct {
            pub fn cross(a: Self, b: Self) Self {
                const tmp0 = @shuffle(f32, a.data, a.data ,@Vector(3, i32){1,2,0});
                const tmp1 = @shuffle(f32, b.data, b.data ,@Vector(3, i32){2,0,1});
                const tmp2 = @shuffle(f32, a.data, a.data ,@Vector(3, i32){2,0,1});
                const tmp3 = @shuffle(f32, b.data, b.data ,@Vector(3, i32){1,2,0});
                return .{ .data = tmp0*tmp1-tmp2*tmp3 };
            }
        } else struct {};

        pub fn add(self: Self, b: Self) Self {
            return .{.data = self.data + b.data};
        }

        pub fn subtract(self: Self, b: Self) Self {
            return .{.data = self.data - b.data};
        }

        pub fn dot(a: Self, b: Self) T {
            return @reduce(.Add, a.data * b.data);
        }

        pub fn mul(a: Self, b: Self) Self {
            return .{ .data = a.data * b.data };
        }

        pub fn div(a: Self, b: Self) Self {
            return .{ .data = a.data / b.data };
        }

        pub fn inc(self: Self) Self {
            return self.add(Self.fromScalar(1));
        }

        pub fn dec(self: Self) Self {
            return self.subtract(Self.fromScalar(1));
        }

        pub fn lessThan(a: Self, b: Self) Vec(size, bool) {
            return .{.data = a.data < b.data};
        }
    };
}

pub const Vec3 = Vec(3, f32);
pub const Vec3u = Vec(3, u32);

const vec3 = Vec3.init;
const vec3u = Vec3u.init;

test "cross product" {
    const a = vec3(1,-8,12);
    const b = vec3(4,6,3);
    const result = vec3(-96,45,38);
    try std.testing.expectEqual(Vec3.cross(a,b), result);
}

test "vector length" {
    const v = vec3(1.5, 100.0, -21.1);
    try std.testing.expectApproxEqAbs(v.length(), 102.21281720019266, 0.0001);
}

pub const Mat4 = struct {
    data: [4][4]f32,

    fn get(self: Mat4, row: usize, column: usize) f32 {
        return self.data[column][row];
    }

    fn set(self: *Mat4, row: usize, column: usize, val: f32) void {
        self.data[column][row] = val;
    }

    pub fn col3(self: Mat4, column: usize) Vec3 {
        return vec3(
            self.get(0, column),
            self.get(1, column),
            self.get(2, column),
        );
    }

    pub fn transformPosition(self: Mat4, v: Vec3) Vec3 {
        return .{ .data =
            self.col3(0).scale(v.x()).data +
            self.col3(1).scale(v.y()).data +
            self.col3(2).scale(v.z()).data +
            self.col3(3).data
        };
    }

    pub fn transformDirection(self: Mat4, v: Vec3) Vec3 {
        return .{ .data =
            self.col3(0).scale(v.x()).data +
            self.col3(1).scale(v.y()).data +
            self.col3(2).scale(v.z()).data
        };
    }
};

pub const Ray = struct {
    orig: Vec3,
    dir: Vec3,

    pub fn at(self: Ray, t: f32) Vec3 {
        return self.orig.add(self.dir.scale(t));
    }
};

const dot = Vec3.dot;
const cross = Vec3.cross;
const subtract = Vec3.subtract;
const add = Vec3.add;

pub const Bbox = struct {
    min: Vec3 = Vec3.zeroes(),
    max: Vec3 = Vec3.zeroes(),

    pub fn extendBy(self: *Bbox, pos: Vec3) void {
        self.min = Vec3.min(self.min, pos);
        self.max = Vec3.max(self.max, pos);
    }

    pub fn size(self: Bbox) Vec3 {
        return subtract(self.max, self.min);
    }

    pub fn rayIntersection(self: Bbox, ray: Ray, t: *f32) bool
    {
        const sign = ray.dir.lessThan(Vec3.zeroes());

        const min = sign.select(self.max, self.min).subtract(ray.orig).div(ray.dir);
        const max = sign.select(self.min, self.max).subtract(ray.orig).div(ray.dir);

        var tmin = min.x();
        var tmax = max.x();

        if ((tmin > max.y()) or (tmax < min.y()))
            return false;

        tmin = @max(tmin, min.y());
        tmax = @min(tmax, max.y());

        if ((tmin > max.z()) or (tmax < min.z()))
            return false;

        tmin = @max(tmin, min.z());
        tmax = @min(tmax, max.z());

        t.* = tmin;

        return true;
    }
};

test "bbox rayIntersection 1" {
    const bbox = Bbox{
        .min = Vec3.fromScalar(-1),
        .max = Vec3.fromScalar(1),
    };
    const ray = Ray{
        .orig = vec3(0, 0, 5),
        .dir = vec3(0, 0, -1),
    };
    var t: f32 = undefined;
    try std.testing.expect(bbox.rayIntersection(ray, &t));
    try std.testing.expectApproxEqAbs(@as(f32, 4), t, 0.0001);
}

test "bbox rayIntersection 2" {
    const bbox = Bbox{
        .min = Vec3.fromScalar(1),
        .max = Vec3.fromScalar(2),
    };
    const ray = Ray{
        .orig = vec3(0, 0, 0),
        .dir = vec3(1, 1, 1).normalize(),
    };
    var t: f32 = undefined;
    try std.testing.expect(bbox.rayIntersection(ray, &t));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f32, 3)), t, 0.0001);
}

test "bbox rayIntersection 3 (if orig is inside bbox => t < 0)" {
    const bbox = Bbox{
        .min = Vec3.fromScalar(-1),
        .max = Vec3.fromScalar(3),
    };
    const ray = Ray{
        .orig = vec3(0, 0, 0),
        .dir = vec3(1, 1, 0).normalize(),
    };
    var t: f32 = undefined;
    try std.testing.expect(bbox.rayIntersection(ray, &t));
    try std.testing.expectApproxEqAbs(-@sqrt(@as(f32, 2)), t, 0.0001);
}

test "bbox rayIntersection 4 (miss)" {
    const bbox = Bbox{
        .min = Vec3.fromScalar(-1),
        .max = Vec3.fromScalar(3),
    };
    const ray = Ray{
        .orig = vec3(5, 5, 5),
        .dir = vec3(1, 1, 0).normalize(),
    };
    var t: f32 = undefined;
    try std.testing.expect(bbox.rayIntersection(ray, &t) == false);
}

pub const Grid = struct {
    bbox: Bbox,
    resolution: Vec3u,
    cell_size: Vec3,

    pub fn getCellPos(grid: Grid, point: Vec3) Vec3u {
        const pos = point.subtract(grid.bbox.min).div(grid.cell_size).toInt(u32);
        return Vec3u.min(pos, grid.resolution.dec());
    }
};

pub const Triangle = struct {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,

    pub fn init(v0: Vec3, v1: Vec3, v2: Vec3) Triangle {
        return .{
            .v0 = v0,
            .e1 = subtract(v1, v0),
            .e2 = subtract(v2, v0),
        };
    }

    pub fn rayIntersection(self: Triangle, ray: Ray, t: *f32, _u: *f32, _v: *f32) bool
    {
        const pvec = cross(ray.dir, self.e2);
        const det = dot(self.e1, pvec);

        const epsilon = 0.00000001; // TODO

        // // if the determinant is negative, the triangle is 'back facing'
        // // if the determinant is close to 0, the ray misses the triangle
        if (det < epsilon) return false;

        const inv_det = 1.0 / det;

        const tvec = subtract(ray.orig, self.v0);
        const u = dot(tvec, pvec) * inv_det;
        if (u < 0 or u > 1) return false;

        const qvec = cross(tvec, self.e1);
        const v = dot(ray.dir, qvec) * inv_det;
        if (v < 0 or u+v > 1) return false;

        t.* = dot(self.e2, qvec) * inv_det;
        _u.* = u;
        _v.* = v;

        return true;
    }
};
