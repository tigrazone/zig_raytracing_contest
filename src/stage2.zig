
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

const Cell = struct {
    first_triangle: u32,
    num_triangles: u32,
};

pub const Geometry = struct {
	triangles: std.MultiArrayList(Triangle).Slice,
	grid: Grid,
	cells: []Cell,
	indices: []usize,
	arena: std.heap.ArenaAllocator,

	pub fn init(allocator: std.mem.Allocator) Geometry {
	    var geometry: Geometry = undefined;
	    geometry.arena = std.heap.ArenaAllocator.init(allocator);
	    return geometry;
	}

	pub fn deinit(self: *Geometry) void {
	    self.arena.deinit();
	}

	fn initGrid(geometry: *Geometry) !void {
	    var bbox: Bbox = .{};

	    for (geometry.triangles.items(.pos)) |pos| {
            var triangle_bbox: Bbox = .{};
            for (0..3) |i| {
                triangle_bbox.extendBy(pos[i]);
            }
            bbox.unionWith(triangle_bbox);
	    }

	    const resolution = main.config.grid_resolution;
	    std.log.info("Grid resolution: {}", .{resolution.data});

	    geometry.grid = Grid.init(bbox, resolution);
	}

	fn initCells(geometry: *Geometry) !usize {
		const grid = &geometry.grid;
		const num_cells = grid.resolution.reduceMul();
		const cells = try geometry.arena.allocator().alloc(Cell, num_cells);
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
	    geometry.cells = cells;
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

	fn initIndices(geometry: *Geometry, total_triangles_count: usize) !void {
		const grid = &geometry.grid;
		const indices = try geometry.arena.allocator().alloc(usize, total_triangles_count);
	    for (geometry.triangles.items(.pos), 0..) |pos, triangle_index| {
            const min = grid.getCellPos(Vec3.min(pos[0], Vec3.min(pos[1], pos[2])));
            const max = grid.getCellPos(Vec3.max(pos[0], Vec3.max(pos[1], pos[2])));

            for (min.z()..max.z()+1) |z| {
                for (min.y()..max.y()+1) |y| {
                    for (min.x()..max.x()+1) |x| {
                        const cell_index = grid.getCellIdx(x,y,z);
                        var cell = &geometry.cells[cell_index];
                        indices[cell.first_triangle + cell.num_triangles] = triangle_index;
                        cell.num_triangles += 1;
                    }
                }
            }
	    }
	    geometry.indices = indices;
	    std.log.info("Unique triangle count: {}/{} ({d:.2}%)",
	        .{geometry.triangles.len, total_triangles_count,
	            @as(f32, @floatFromInt(geometry.triangles.len)) / @as(f32, @floatFromInt(total_triangles_count)) * 100});
	}

	pub fn build(self: *Geometry) !void {
	    try self.initGrid();
	    const total_triangles_count = try self.initCells();
	    try self.initIndices(total_triangles_count);
	}

	pub fn bakeInto(geometry: Geometry, scene: *stage3.Scene) !void {
	    const allocator = scene.arena.allocator();

	    const cells = try allocator.alloc(stage3.Cell, geometry.cells.len);
	    for (geometry.cells, cells) |src, *dst| {
	    	dst.* = .{
	    		.begin = src.first_triangle,
	    		.end = src.first_triangle + src.num_triangles,
	    	};
	    }

	    var triangles = std.MultiArrayList(stage3.Triangle){};
	    try triangles.resize(allocator, geometry.indices.len);

	    for (0..geometry.indices.len) |i| {
	    	const src = geometry.triangles.get(geometry.indices[i]);
	    	const dst = stage3.Triangle{
	    	    .pos = stage3.Triangle.Pos.init(src.pos[0], src.pos[1], src.pos[2]),
	    	    .data = src.data,
	    	};
	    	triangles.set(i, dst);
	    }

	    scene.grid = geometry.grid;
	    scene.cells = cells;
	    scene.triangles_pos = triangles.items(.pos);
	    scene.triangles_data = triangles.items(.data);
	}
};
