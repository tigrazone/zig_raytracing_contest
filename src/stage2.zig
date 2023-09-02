
const std = @import("std");

const main = @import("main.zig");
const linalg = @import("linalg.zig");
const stage3 = @import("stage3.zig");

const Vec3 = linalg.Vec3;
const Grid = linalg.Grid;
const Bbox = linalg.Bbox;

// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

pub const Triangle = struct {
	pos: [3]Vec3,
	data: stage3.Triangle.Data,
};

pub const Geometry = struct {
	triangles: std.MultiArrayList(Triangle),
	arena: std.heap.ArenaAllocator,

	pub fn init(allocator: std.mem.Allocator) Geometry {
	    var geometry: Geometry = undefined;
	    geometry.arena = std.heap.ArenaAllocator.init(allocator);
	    return geometry;
	}

	pub fn deinit(self: *Geometry) void {
	    self.arena.deinit();
	}

	fn initGrid(geometry: Geometry) !Grid {
	    var mean_triangle_size = Vec3.zeroes();
	    var bbox: Bbox = .{};

	    for (geometry.triangles.items(.pos)) |pos| {
            var triangle_bbox: Bbox = .{};
            for (0..3) |i| {
                triangle_bbox.extendBy(pos[i]);
            }
            mean_triangle_size = Vec3.add(mean_triangle_size, triangle_bbox.size());
            bbox.unionWith(triangle_bbox);
	    }

	    mean_triangle_size = mean_triangle_size.div(Vec3.fromScalar(@floatFromInt(geometry.triangles.len)));
	    const mean_triangle_count = bbox.size().div(mean_triangle_size);

	    std.log.info("Mean triangle count: {d:.1}", .{mean_triangle_count.data});
	    const resolution = main.config.grid_resolution orelse mean_triangle_count.div(Vec3.fromScalar(4)).ceil().toInt(u32);
	    std.log.info("Grid resolution: {}", .{resolution.data});

	    return Grid.init(bbox, resolution);
	}

	fn initCells(geometry: Geometry, cells: []stage3.Scene.Cell, grid: Grid) !usize {
	    @memset(cells, .{.first_triangle = 0, .num_triangles = 0});
	    for (geometry.triangles.items(.pos)) |pos| {
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

	fn initTriangles(geometry: Geometry, triangles: *std.MultiArrayList(stage3.Triangle), cells: []stage3.Scene.Cell, grid: Grid) !void {
	    for (geometry.triangles.items(.pos), geometry.triangles.items(.data)) |pos, data| {
            const triangle = stage3.Triangle{
                .pos = stage3.Triangle.Pos.init(pos[0], pos[1], pos[2]),
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
};

pub fn compileGeometry(geometry: Geometry, scene: *stage3.Scene) !void {
    const allocator = scene.arena.allocator();

    const grid = try geometry.initGrid();
    const cells = try allocator.alloc(stage3.Scene.Cell, grid.resolution.reduceMul());
    const total_triangles_count = try geometry.initCells(cells, grid);

    var triangles = std.MultiArrayList(stage3.Triangle){};
    try triangles.resize(allocator, total_triangles_count);
    try geometry.initTriangles(&triangles, cells, grid);

    std.log.info("Unique triangle count: {}/{} ({d:.2}%)",
        .{geometry.triangles.len, total_triangles_count,
            @as(f32, @floatFromInt(geometry.triangles.len)) / @as(f32, @floatFromInt(total_triangles_count)) * 100});

    scene.grid = grid;
    scene.cells = cells;
    scene.triangles_pos = triangles.items(.pos);
    scene.triangles_data = triangles.items(.data);
}