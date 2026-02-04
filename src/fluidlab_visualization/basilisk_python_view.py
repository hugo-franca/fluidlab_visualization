import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.spatial.distance import cdist
from matplotlib.collections import LineCollection, PolyCollection
from typing import Callable
from cmap import Colormap
import struct
import os

def ReadLine_Binary(file):
	line = ""
	read_stuff = file.read(1).decode('UTF-8')
	index = 0
	while( read_stuff!='\n' ):
		index += 1
		line += read_stuff
		read_stuff = file.read(1).decode('UTF-8')
		if(index>500):
			return None
	return line

def ReadPoint_Float(file):
	x = struct.unpack('>f', file.read(4))[0]
	y = struct.unpack('>f', file.read(4))[0]
	z = struct.unpack('>f', file.read(4))[0]
	return np.array([x, y, z])

def ReadInt(file):
	return struct.unpack('>i', file.read(4))[0]

def ReadDouble(file):
	return struct.unpack('>d', file.read(8))[0]

def ReadFloat(file):
	return struct.unpack('>f', file.read(4))[0]

# def Find_VTK_Timesteps(vtk_folder):
# 	temp_list_files = os.listdir(vtk_folder)
# 	list_files = []
# 	for file in temp_list_files:
# 		if( file.startswith("Interface") ):
# 			list_files.append( int(file[11:-4]) )
# 	list_files = np.sort(list_files)
# 	return list_files


def ReadFieldsVTK(file_name : str, 
				  color_fields : str | Callable[[dict], float], cmap_fields, crange=[None, None], 
				  color_fields_mirror : str | Callable[[dict], float] | None = None, cmap_fields_mirror = None, crange_mirror=[None, None], mirror_direction = "y", 
				  cell_edges_color : str | None = None, cell_edges_width : float = 0.5,
				  rasterize=False):
	"""
	PARAMETERS
	file_name: name of the mesh.vtk file to read
	color_fields: how to colour the cells of the mesh. Can be one of two things:
		1) A string, with the name of the property you want to visualize in the mesh
		2) A callable function that makes some operation on the properties and returns a value
	cmap: colormap
	crange: range of the colormap. Automatic if left at [None, None]
	
	OUTPUT
	cells: list of cells in the mesh. Each cell is a square and contains:
			x, y: coordinates of the cell center
			Delta: size of the cell (its a square of size Delta by Delta)
			p: pressure
			vel_v, vel_v: velocity components
			tau_xx, tau_xy, tau_yy: stress components
			f: volume fraction
	collection: matplotlib collection to plot the mesh
	"""

	try:
		file = open(file_name, 'rb')
	except FileNotFoundError:
		return None, None, None, None, None, None
	
	line = ReadLine_Binary(file) # vtk DataFile Version 2.0
	line = ReadLine_Binary(file) # MESH. time X
	line = ReadLine_Binary(file) # BINARY
	line = ReadLine_Binary(file) # DATASET UNSTRUCTURED_GRID
	line = ReadLine_Binary(file) # FIELD FieldData 1
	line = ReadLine_Binary(file) # time 1 1 float
	simulation_time = ReadFloat(file)

	line = ReadLine_Binary(file) # \n
	line = ReadLine_Binary(file) # POINTS %d float	
	num_points = int(line.split(" ")[1])
	num_cells = np.rint(num_points/4).astype(int)
	
	array_cell_centers = np.zeros(shape=(num_cells, 2))
	array_cell_sizes = np.zeros(shape=(num_cells, ))
	array_cell_vertices = np.zeros(shape=(num_cells, 4, 2))

	for i in range(num_cells):

		# Discarding the third coordinate for now... only 2D
		p1 = ReadPoint_Float(file)[:2]
		p2 = ReadPoint_Float(file)[:2]
		p3 = ReadPoint_Float(file)[:2]
		p4 = ReadPoint_Float(file)[:2]
		
		# Adding a new nico_cell to the list
		array_cell_vertices[i] = [p1, p2, p3, p4]
		
		array_cell_centers[i] = 0.25*(p1 + p2 + p3 + p4)
		array_cell_sizes[i] = p2[0] - p1[0]
		# cell_delta = 2.0*np.abs(cell_center[0] - p1[0])

	# Reading the ordering of the points in the cells
	# Doesnt actually matter, because I know the ordering I used...
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	for i in range(num_cells):
		read_stuff = ReadInt(file)
		read_stuff = ReadInt(file)
		read_stuff = ReadInt(file)
		read_stuff = ReadInt(file)
		read_stuff = ReadInt(file)

	# Reading CELL_TYPES
	# Doesnt matter because Im assuming its always squares
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	for i in range(num_cells):
		read_stuff = ReadInt(file)

	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)

	# Starting to read printed field data
	dict_values = {}
	line = ReadLine_Binary(file) # SCALARS
	while( (line is not None) and line.split(" ")[0]=="SCALARS" ):
		field_name = line.split(" ")[1] 
		line = ReadLine_Binary(file) # LOOKUP_TABLE
		dict_values[field_name] = np.zeros(shape=(num_cells, ))
		for i in range(num_cells):
			dict_values[field_name][num_cells - i - 1] = ReadFloat(file)
		line = ReadLine_Binary(file)
		line = ReadLine_Binary(file)

	# Finished reading the file
	file.close()


	# Setting cell colours and making the PolyCollection that will be visualized
	values = dict_values[color_fields] if (color_fields in dict_values.keys()) else color_fields(dict_values)
	min_values, max_values = np.min(values), np.max(values)
	norm_colors = mpl.colors.Normalize(vmin=crange[0] if crange[0] else min_values, vmax=crange[1] if crange[1] else max_values) 
	array_colors = cmap_fields( norm_colors( values ) )
	poly_collection = PolyCollection(array_cell_vertices, facecolors=array_colors, edgecolor=cell_edges_color, linewidth=cell_edges_width)

	# Setting cell colours for the mirror part (if requested) and making the PolyCollection
	if( color_fields_mirror ):
		values_mirror = dict_values[color_fields_mirror] if (color_fields_mirror in dict_values.keys()) else color_fields_mirror(dict_values)
		min_values_mirror, max_values_mirror = np.min(values_mirror), np.max(values_mirror)
		norm_colors_mirror = mpl.colors.Normalize(vmin=crange_mirror[0] if crange_mirror[0] else min_values_mirror, vmax=crange_mirror[1] if crange_mirror[1] else max_values_mirror) 
		array_colors_mirror = cmap_fields_mirror( norm_colors_mirror( values_mirror ) )
		mirror_direction = 0 if mirror_direction=="x" else 1 if mirror_direction=="y" else None
		array_cell_vertices[:, :, mirror_direction] *= -1.0
		poly_collection_mirror = PolyCollection(array_cell_vertices, facecolors=array_colors_mirror, edgecolor=cell_edges_color, linewidth=cell_edges_width)
		return array_cell_centers, array_cell_sizes, dict_values, poly_collection, poly_collection_mirror
	
	return array_cell_centers, array_cell_sizes, dict_values, poly_collection


# Reading the reconstructed interface between the water and air
# Input: filename is the name of the interface.vtk file
# The output is a sequence of line segments
# Output points: array of points
# Output segments: each segment has two integers. These integers are the indices of the points in the points array
# Output collection: matplotlib LineCollection ready to be plotted

def read_polydata_legacy(filename : str, rotate : float = 0.0, flip_x : bool = False, color = "black", rasterized : bool = False):
	file = open(filename, "rt")
	line = file.readline()
	line = file.readline()
	simulation_time = float(line.split(" ")[4])
	line = file.readline()
	line = file.readline()
	line = file.readline()

	sin_angle = np.sin(np.pi*rotate/(180.0))
	cos_angle = np.cos(np.pi*rotate/(180.0))
	
	flip_mult = [-1.0, 1.0] if flip_x else [1.0, 1.0]

	num_points = int( line.split(" ")[1] )
	points = np.zeros(shape=(num_points, 2))
	for i in range(num_points):
		line = file.readline()
		line = line.split(" ")
		x = float(line[0])
		y = float(line[1])

		if( rotate!=0.0 ):
			points[i, 0] = x*cos_angle - y*sin_angle
			points[i, 1] = x*sin_angle + y*cos_angle
		else:
			points[i, 0] = x
			points[i, 1] = y



	line = file.readline()
	num_polys = int(line.split(" ")[1])

	collection = []
	flipped_collection = []

	segments = []
	for i in range(num_polys):
		line = file.readline()
		line = line.split(" ")
		num_vertices = int(line[0])

		if( num_vertices<2 ):
			continue

		p1 = int(line[1])
		p2 = int(line[2])
		segments.append( [p1, p2] )

		collection.append(np.array([points[p1, :], points[p2, :]]))
		
		if( flip_x ):
			flipped_collection.append(flip_mult*np.array([points[p1, :], points[p2, :]]))

	file.close()

	line_collection = LineCollection(collection, colors=color, linewidth=1.5, rasterized=rasterized)
	flipped_line_collection = LineCollection(flipped_collection, colors=color, linewidth=1.5, rasterized=rasterized)

	# ===== Now I try to obtain a connected polygon to describe the interface
	mid_points = np.array( [0.5*(points[segment[0], :] + points[segment[1], :]) for segment in segments] )

    # Calculating distance between all points to each other
	distance_matrix = cdist(mid_points, mid_points)

	# Attempting to obtain a mid_point
	idx_initial = np.argwhere(np.abs(mid_points[:, 1])<0.08).flatten()
	selected_points = mid_points[idx_initial, :]
	min_x_center = np.min(selected_points[:, 0])
	max_x_center = np.max(selected_points[:, 0])
	com_point = [0.5*(min_x_center + max_x_center), 0.0]

	idx_initial = np.argwhere(mid_points[:, 0]<com_point[0]).flatten()
	selected_points = mid_points[idx_initial, :]
	idx_initial = np.argmin(np.abs(selected_points[:, 1])).flatten()[0]
	initial_point = selected_points[idx_initial, :]
	index_first_point = np.argwhere( ((mid_points[:]==initial_point).all(1)) == True ).flatten()[0]

	# print(initial_index, mid_points[initial_index, :])
	polygon_points = np.zeros_like(mid_points)
	polygon_points[0, :] = mid_points[index_first_point, :]
	index_previous_point = index_first_point
	for i in range(len(mid_points) - 1):
		distance_matrix[index_previous_point, index_previous_point] = 1e+10

		index_next_point = np.argmin(distance_matrix[index_previous_point, :]).flatten()
		polygon_points[i+1, :] = mid_points[index_next_point, :]

		distance_matrix[index_previous_point, :] = distance_matrix[:, index_previous_point] = 1e+10        
		index_previous_point = index_next_point

	return simulation_time, points, polygon_points, segments, line_collection, flipped_line_collection

def read_polydata_3D(filename : str, rotate : float = 0.0, flip_y : bool = False, color="black", ignore_z : bool = True):
	file = open(filename, "rb")
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	time = line.split(" ")[-1]
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	
	num_points = int( str(line).split(" ")[1] )
	points = np.zeros(shape=(num_points, 3))
	for i in range(num_points):
		points[i, :] = ReadPoint_Float(file)

	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	num_polys = int(str(line).split(" ")[1])

	segments = []
	for i in range(num_polys):
		num_vertices = ReadInt(file)
		
		indices_points = []
		for i in range(num_vertices):
			indices_points.append( ReadInt(file) )

		if( num_vertices<2 ):
			continue
		
		segments.append( indices_points )

	file.close()

	return float(time), points, segments


def read_polydata(filename : str, only_2D : bool = True, rotate_2D : float = 0.0, flip_y : bool = False, 
				  color : str = "black", color_range = [None, None], colormap = None):
	"""
	This function reads a VTK file "Interface_XXXX.vtk" coming from one of our Basilisk simulations.

	Parameters
		filename: the name of the interface file
		2D_only: if this file comes from a 2D simulation, set this as true. If 3D set as false. The output of the function is different in each case.
		rotate_2D: only relevant if 2D_only=True. You can rotate the visualization by a certain angle. This parameter is that angle (in degrees).
		flip_y: generate a second data set which is the same as the original, but with the y-coordinates flipped. Cute for axissymetric simulations where only the top half is simulated
		color: color of the lines of the interface. Can be a matplotlib solid color or the name of a property contained in the VTK file. In this case we will color based on that property with a colormap
	"""

	file = open(filename, "rb")
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	time = line.split(" ")[-1]
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	
	# Rotation matrix, which will be used later if requested
	sin_angle = np.sin(np.pi*rotate_2D/(180.0))
	cos_angle = np.cos(np.pi*rotate_2D/(180.0))
	rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

	flip_mult = [1.0, -1.0, 1.0] if flip_y else [1.0, 1.0, 1.0]

	# Loading all the points into a numpy [n x 3] array
	num_points = int( str(line).split(" ")[1] )
	points = np.zeros(shape=(num_points, 3))
	for i in range(num_points):
		points[i, :] = ReadPoint_Float(file)
	flipped_points = flip_mult*points if flip_y else None

	# Reading how many lines (in 2D) or polygons we will have
	line = ReadLine_Binary(file)
	line = ReadLine_Binary(file)
	num_polys = int(str(line).split(" ")[1])

	# If 2D: Discarding the third coordinate and applying rotation
	if( only_2D ):
		points = np.dot(points[:, :2], rotation_matrix.T)
		flipped_points = np.dot(flipped_points[:, :2], rotation_matrix.T) if flip_y else None


	# Looping over each line (in 2D) or polygon (in 3D) and reading which vertices define it
	collection = []
	flipped_collection = []
	segments = []
	degenerate_indices = []
	for i in range(num_polys):
		num_vertices = ReadInt(file) # How many vertices in this line/polygon
		
		indices_points = [] # List of vertices in this line/polygon
		for i in range(num_vertices):
			indices_points.append( ReadInt(file) )

		# Discarding degenerate cases that have less than 2 vertices
		if( num_vertices<2 ):
			degenerate_indices.append( i )
			continue

		segments.append( indices_points )

		if( only_2D ):
			collection.append(np.array([points[indices_points[0], :2], points[indices_points[1], :2]]))
			if( flip_y ):
				flipped_collection.append(np.array([flipped_points[indices_points[0], :], flipped_points[indices_points[1], :]]))
            

	# Now we are going to read the scalar data associated to each polygon
	line = ReadLine_Binary(file) # New line
	line = ReadLine_Binary(file) # CELL_DATA X
	dict_values = {}
	line = ReadLine_Binary(file) # SCALARS
	while( (line is not None) and line.split(" ")[0]=="SCALARS" ):
		field_name = line.split(" ")[1] 
		line = ReadLine_Binary(file) # LOOKUP_TABLE
		dict_values[field_name] = np.zeros(shape=(len(segments), ))
		i_segment = 0
		for i in range(num_polys):
			if( not(i in degenerate_indices) ):
				dict_values[field_name][i_segment] = ReadFloat(file)
				i_segment += 1
		line = ReadLine_Binary(file)
		line = ReadLine_Binary(file)

	file.close()

	if( color in dict_values.keys() ):
		values = dict_values[color]
		min_values, max_values = np.min(values), np.max(values)
		norm_colors = mpl.colors.Normalize(vmin=color_range[0] if color_range[0] else min_values, vmax=color_range[1] if color_range[1] else max_values) 
		colormap = Colormap('bids:plasma') if colormap is None else colormap
		color = colormap( norm_colors( values ) )

	if( only_2D ):
		line_collection = LineCollection(collection, colors=color)
		flipped_line_collection = LineCollection(flipped_collection, colors=color) if len(flipped_collection)>0 else None

		return float(time), dict_values, points, segments, line_collection, flipped_line_collection
	
	print("ERROR. read_polydata: need to implement 3D version.")
	exit()

def droplet_properties(filename):
	file = open(filename, "rt")
	line = file.readline()
	line = file.readline()
	line = file.readline()
	line = file.readline()
	line = file.readline()

	droplet_radius = -1e+10

	num_points = int( line.split(" ")[1] )
	# points = np.zeros(shape=(num_points, 2))
	for i in range(num_points):
		line = file.readline()
		line = line.split(" ")
		x = float(line[0])
		y = float(line[1])
		# points[i, 0] = x
		# points[i, 1] = y

		if( y>droplet_radius ):
			droplet_radius = y

	file.close()
	return droplet_radius

def find_all_vtk_files_in_folder(folder_name):
	temp_list_files = os.listdir(folder_name)
	list_files = []
	for file in temp_list_files:
		if( file.startswith("Interface") and file.endswith(".vtk") ):
			list_files.append( file[10:-4] )
	list_files = np.sort(np.array(list_files).astype("float"))

	return list_files


# def Read_VTK_As_Uniform_Grid(filename):

# 	# Loading the original VTK using pyvista
# 	grid = pv.UnstructuredGrid(filename)

# 	# Converting the cell data to point data
# 	grid_new = grid.cell_data_to_point_data()

# 	# Cell size of this grid
# 	dx = (grid.get_cell(0).points[1] - grid.get_cell(0).points[0])[0]
# 	# dx = (grid.cells[0].points[1] - grid.cells[0].points[0])[0]

# 	# Limits of the domain
# 	grid_bounds = grid_new.bounds
# 	x_lims = grid_bounds[:2]
# 	y_lims = grid_bounds[2:4]

# 	# Number of cells in x and y directions
# 	cells_x = int(np.ceil((x_lims[1] - x_lims[0])/dx))
# 	cells_y = int(np.ceil((y_lims[1] - y_lims[0])/dx))

# 	# Initializing empty matrix
# 	matrix_ux = np.zeros(shape=(cells_x+1, cells_y+1))
# 	matrix_uy = np.zeros(shape=(cells_x+1, cells_y+1))
# 	matrix_p = np.zeros(shape=(cells_x+1, cells_y+1))
# 	matrix_ux[:] = np.nan
# 	matrix_uy[:] = np.nan
# 	matrix_p[:] = np.nan

# 	# Looping over the points and bringing the data to the matrix
# 	for p, ux, uy, pressure in zip(grid_new.points, grid_new.get_array("ux"), grid_new.get_array("uy"), grid_new.get_array("a")):
# 		p_i = int( np.round( (-x_lims[0] + p[0])/dx) )
# 		p_j = int( np.round( (-y_lims[0] + p[1])/dx) )
# 		matrix_ux[p_i, p_j] = ux
# 		matrix_uy[p_i, p_j] = uy
# 		matrix_p[p_i, p_j] = pressure

# 	x_points = np.linspace(x_lims[0], x_lims[1], cells_x+1)
# 	y_points = np.linspace(y_lims[0], y_lims[1], cells_y+1)
# 	x_points, y_points = np.meshgrid(x_points, y_points)


# 	# matrix_f, matrix_p, matrix_vel_u, matrix_vel_v, matrix_txx, matrix_txy, matrix_tyy

# 	return x_points.T, y_points.T, matrix_ux, matrix_uy, matrix_p