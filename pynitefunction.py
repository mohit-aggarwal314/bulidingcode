import ezdxf
from PyNite import FEModel3D
from PyNite.Visualization import Renderer
import itertools
import numpy as np

doc = ezdxf.readfile('testfile3.dxf')
msp = doc.modelspace()


def generate_beam_pairs(column_coordinates, slmax):
    beam_pairs = []

    sorted_columns = sorted(column_coordinates, key=lambda p: (p[1], p[0]))  # Sort top to bottom, left to right

    for i in range(len(sorted_columns)):
        for j in range(i + 1, len(sorted_columns)):
            point1 = sorted_columns[i]
            point2 = sorted_columns[j]

            distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

            if distance > slmax:
                num_secondary_points = int(distance / slmax)
                secondary_points = [(point1[0] + (point2[0] - point1[0]) * k / num_secondary_points,
                                    point1[1] + (point2[1] - point1[1]) * k / num_secondary_points)
                                   for k in range(1, num_secondary_points)]

                for k in range(len(secondary_points) - 1):
                    beam_pairs.append((secondary_points[k], secondary_points[k+1]))

            beam_pairs.append((point1, point2))

    return beam_pairs

def filter_orthogonal_pairs(beam_pairs):
    orthogonal_beam_pairs = []

    for start, end in beam_pairs:
        if start[0] == end[0] or start[1] == end[1]:
            orthogonal_beam_pairs.append((start, end))

    return orthogonal_beam_pairs

def remove_overlapping_pairs(beam_pairs):
    unique_beam_pairs = []

    y_coordinates = set([point[1] for pair in beam_pairs for point in pair])
    
    for y in y_coordinates:
        markers = {}  # Mark distances of each point on the y-axis
        sorted_pairs = sorted([pair for pair in beam_pairs if pair[0][1] == y or pair[1][1] == y],
                              key=lambda pair: min(pair[0][0], pair[1][0]))

        for pair in sorted_pairs:
            x_start = pair[0][0]
            x_end = pair[1][0]

            if x_start not in markers or markers[x_start] < y:
                markers[x_start] = y
                unique_beam_pairs.append(pair)

            if x_end not in markers or markers[x_end] < y:
                markers[x_end] = y
                unique_beam_pairs.append(pair)

    return unique_beam_pairs

def distance(node1, node2):
    return ((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)**0.5

def make_orthogonal(node1, node2):
    x_diff = abs(node2[0] - node1[0])
    y_diff = abs(node2[1] - node1[1])
    
    if x_diff > y_diff:
        return ((node1[0], node2[1]), node2)
    else:
        return (node2, (node1[0], node2[1]))

def generate_members(node_coordinates):
    pairs = []
    
    for pair in itertools.combinations(node_coordinates, 2):
        no_overlap = True
        
        for existing_pair in pairs:
            if any(coord in existing_pair for coord in pair):
                no_overlap = False
                break
        
        if no_overlap:
            pairs.append(pair)
        
    return pairs

def get_rectangle_blocks_info(block_name):
    
    
    block_info = []  # List to store block information
    
    for block in doc.blocks:
        if block.name == block_name:
            print(f"Found block with name '{block_name}'")
            for entity in block:
                print(f"Entity type: {entity.dxftype()}")
                if entity.dxftype() == 'LWPOLYLINE' and entity.is_closed:
                    points = list(entity.get_points('xy'))
                    if len(points) == 4:
                        print("Found closed lightweight polyline with 4 points")
                        
                        # Calculate the base point of the block
                        base_point = points[0]
                        
                        # Calculate lengths based on the two adjacent corners
                        length_x = points[1][0] - points[0][0]
                        length_y = points[3][1] - points[0][1]
                        
                        block_info.append({
                            'base_point': base_point,
                            'length_x': length_x,
                            'length_y': length_y
                        })
    
    return block_info


def draw_circles_within_rectangle(rectangle_corner, rectangle_length_x, rectangle_length_y, layer_name, circle_radius=1.0):
    

    coordinates_by_hatch = {}

    for entity in msp:
        if entity.dxftype() == 'HATCH' and entity.dxf.layer == layer_name:
            hatch_coordinates = []
            
            for path in entity.paths:
                if isinstance(path, ezdxf.entities.PolylinePath):
                    for vertex in path.vertices:
                        x, y, _ = vertex  # Get x, y, and ignore the bulge
                        
                        # Check if the vertex is within the rectangle's boundaries
                        if (
                            rectangle_corner[0] <= x <= rectangle_corner[0] + rectangle_length_x and
                            rectangle_corner[1] <= y <= rectangle_corner[1] + rectangle_length_y
                        ):
                            hatch_coord = (x - rectangle_corner[0], y - rectangle_corner[1])  # Adjust coordinates based on rectangle's origin
                            # msp.add_circle(center=hatch_coord, radius=circle_radius)
                            hatch_coordinates.append(hatch_coord)
            
            if hatch_coordinates:  # Only add non-empty coordinate lists to the dictionary
                coordinates_by_hatch[entity] = hatch_coordinates
        
    return coordinates_by_hatch  #


# dxf_file_path = "testfile3.dxf"
# doc = ezdxf.readfile(dxf_file_path)
# msp = doc.modelspace()
target_layer = "A- HATCH"
circle_radius = 1000
circle_layer = "Circles"


block_name = 'planrect1'



# Initialize block reference as None
block_ref = None

# Iterate through the INSERT entities in modelspace
for insert in msp.query('INSERT'):
    if insert.dxf.name == block_name:
        block_ref = insert
        break

if block_ref is not None:
    # Extract insert point (location)
    insert_point = block_ref.dxf.insert

    # Get the block definition associated with the reference
    block_def = doc.blocks[block_ref.dxf.name]

    # Initialize bounding box extents
    min_point = None
    max_point = None

    # Iterate through the entities in the block definition
    for entity in block_def:
        if entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points('xy')  # Get the 2D vertices of LWPOLYLINE
            for point in points:
                if min_point is None:
                    min_point = point
                    max_point = point
                else:
                    min_point = (min(min_point[0], point[0]), min(min_point[1], point[1]))
                    max_point = (max(max_point[0], point[0]), max(max_point[1], point[1]))

    # Calculate lengths in X and Y directions
    length_x = max_point[0] - min_point[0]
    length_y = max_point[1] - min_point[1]

    rectangle_corner = insert_point
    rectangle_length_x = length_x
    rectangle_length_y = length_y

    print("Insert Point:", insert_point)
    print("Length X:", length_x)
    print("Length Y:", length_y)
else:
    print("Block not found.")


# # Call the function with user inputs and other parameters
hatch_coordinates = draw_circles_within_rectangle(rectangle_corner, rectangle_length_x, rectangle_length_y, target_layer, circle_radius)



coord_list = []
# Assuming hatch_coordinates is the dictionary you provided
for hatch, coordinates in hatch_coordinates.items():
    
    for coord in coordinates:
        coord_list.append(coordinates)
        
    

# Initialize FE model and add nodes
beam = FEModel3D()
nodeCounter = 0 
avg_coords = []
for coord in coord_list:
    if not coord:
        continue  # Skip empty coordinate sets
    
    avg_x = sum(x for x, y in coord) / len(coord)
    avg_y = sum(y for x, y in coord) / len(coord)
    avg_coord = (avg_x, avg_y)

    if avg_coord in avg_coords:
        continue
    
    avg_coords.append(avg_coord)

for i in range(len(avg_coords)):
    msp.add_circle(center=avg_coords[i], radius=circle_radius)

print(len(avg_coords))    

member_pairs = generate_members(avg_coords)

# for i in range(len(member_pairs)):
#     msp.add_lwpolyline([member_pairs[i][0],member_pairs[i][1]], dxfattribs={'color': 2})

slmax = 5000

# Generate beam pairs and filter for orthogonality
beam_pairs = generate_beam_pairs(avg_coords, slmax)
orthogonal_beam_pairs = filter_orthogonal_pairs(beam_pairs)

# Remove overlapping pairs
non_overlapping_beam_pairs = remove_overlapping_pairs(orthogonal_beam_pairs)

for i in range(len(non_overlapping_beam_pairs)):
    msp.add_lwpolyline([non_overlapping_beam_pairs[i][0],non_overlapping_beam_pairs[i][1]], dxfattribs={'color': 2})

    
print(non_overlapping_beam_pairs)

# Print the generated non-overlapping orthogonal beam pairs
for start, end in non_overlapping_beam_pairs:
    print(f"Beam: {start} to {end}")


# all_nodes = set()
# for pair in member_pairs:
#     all_nodes.update(pair)

# unique_node_coordinates = list(all_nodes)

# print(len(unique_node_coordinates))

# # Define maximum span length
# lengthmax = 20000  # Specify the maximum span length

# # Sort the coordinates to start from the top-left
# unique_node_coordinates.sort(key=lambda coord: (coord[1], coord[0]))

# Adding nodes
nodes = {}
for i, (x, y) in enumerate(non_overlapping_beam_pairs):
    node_name = f'N{i+1}'
    nodes[node_name] = (x, y, 0)
    beam.add_node(node_name, x, y, 0)


# # Define material properties and add members
# # col prop
# iycol = 30568.33102
# izcol =15284.11746 
# Jcol =438.2525265 
# Acol =30580.51518
# E =  3150.002112909767 #29000       # Modulus of elasticity (ksi)
# G =   1346.1547119175743  #11200       # Shear modulus of elasticity (ksi)
# nu =  0.17 #0.3        # Poisson's ratio
# rho = 0.008679 # 2.836e-4  # Density (kci)
# beam.add_material('Steel', E, G, nu, rho)

# # Adding members
# for i in range(len(unique_node_coordinates)):
#     for j in range(i + 1, len(unique_node_coordinates)):
#         node1 = f'N{i+1}'
#         node2 = f'N{j+1}'
#         x1, y1, _ = nodes[node1]
#         x2, y2, _ = nodes[node2]

#         dx = abs(x2 - x1)
#         dy = abs(y2 - y1)

#         if (dx <= lengthmax and y1 == y2) or (dy <= lengthmax and x1 == x2):
#             beam.add_member(f'M{i+1}_{j+1}', node1, node2, 'Steel', iycol, izcol, Jcol, Acol)

#             msp.add_lwpolyline([member_pairs[i][0],member_pairs[i][1]], dxfattribs={'color': 2})    

# for i in range(len(unique_node_coordinates)):
#     msp.add_circle(center=unique_node_coordinates[i], radius=circle_radius)
    
# # Structural analysis
# # beam.analyze()

# Visualization
renderer = Renderer(beam)
renderer.annotation_size = 6
renderer.deformed_shape = False
renderer.deformed_scale = 1
renderer.render_loads = False
renderer.render_model()   

# Save the modified DXF file
doc.saveas('newfile3.dxf')


