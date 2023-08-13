const std = @import("std");


pub const RGB = struct {
    r: u8,
    g: u8,
    b: u8,
};

test "RGB size should be 3" {
    try std.testing.expect(@sizeOf(RGB) == 3);
}

pub const Vec3 = struct {
    data: @Vector(3, f32),

    pub fn x(self: Vec3) f32 {
        return self.data[0];
    }
    pub fn y(self: Vec3) f32 {
        return self.data[1];
    }
    pub fn z(self: Vec3) f32 {
        return self.data[2];
    }

    pub fn zeroes() Vec3 {
        return .{ .data = .{0.0, 0.0, 0.0}};
    }

    pub fn ones() Vec3 {
        return .{ .data = .{1.0, 1.0, 1.0}};
    }

    pub fn init(_x: f32, _y: f32, _z: f32) Vec3 {
        return .{ .data = .{_x, _y, _z}};
    }

    pub fn sqrt(self: Vec3) Vec3 {
        return .{ .data = @sqrt(self.data) };
    }

    pub fn clamp(self: Vec3, min: f32, max: f32) Vec3 {
        return .{ .data = @min(@max(self.data, @as(@Vector(3, f32), @splat(min))), @as(@Vector(3, f32), @splat(max))) };
    }

    pub fn scale(self: Vec3, s: f32) Vec3 {
        return .{ .data = self.data * @as(@Vector(3, f32), @splat(s)) };
    }

    pub fn toRGB(self: Vec3) RGB {
        const rgb = self.clamp(0.0, 0.999999).scale(256);
        return .{
            .r = @intFromFloat(rgb.data[0]),
            .g = @intFromFloat(rgb.data[1]),
            .b = @intFromFloat(rgb.data[2]),
        };
    }

    pub fn length(self: Vec3) f32 {
        return @sqrt(@reduce(.Add, self.data * self.data));
    }

    pub fn normalize(self: Vec3) Vec3 {
        return self.scale(1.0 / self.length());
    }

    pub fn add(self: Vec3, b: Vec3) Vec3 {
        return .{.data = self.data + b.data};
    }

    pub fn subtract(self: Vec3, b: Vec3) Vec3 {
        return .{.data = self.data - b.data};
    }

    pub fn dot(a: Vec3, b: Vec3) f32 {
        return @reduce(.Add, a.data * b.data);
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        const tmp0 = @shuffle(f32, a.data, a.data ,@Vector(3, i32){1,2,0});
        const tmp1 = @shuffle(f32, b.data, b.data ,@Vector(3, i32){2,0,1});
        const tmp2 = @shuffle(f32, a.data, a.data ,@Vector(3, i32){2,0,1});
        const tmp3 = @shuffle(f32, b.data, b.data ,@Vector(3, i32){1,2,0});
        return .{ .data = tmp0*tmp1-tmp2*tmp3 };
    }

};

const vec3 = Vec3.init;

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
    data: [16]f32,

    fn get(self: Mat4, row: usize, column: usize) f32 {
        return self.data[column*4+row];
    }

    fn set(self: *Mat4, row: usize, column: usize, val: f32) void {
        self.data[column*4+row] = val;
    }

    pub fn identity() Mat4 {
        var result: Mat4 = .{ .data = [_]f32{0} ** 16 };
        for (0..4) |i| {
            result.data[i*4+i] = 1;
        }
        return result;
    }

    pub fn translation(t: [3]f32) Mat4 {
        return .{ .data = .{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            t[0], t[1], t[2], 1,
        }};
    }

    pub fn rotation(q: [4]f32) Mat4 {
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

    pub fn scale(s: [3]f32) Mat4 {
        return .{ .data = .{
            s[0], 0, 0, 0,
            0, s[1], 0, 0,
            0, 0, s[2], 0,
            0, 0, 0, 1,
        }};
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

    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        var result: Mat4 = undefined;
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
};
