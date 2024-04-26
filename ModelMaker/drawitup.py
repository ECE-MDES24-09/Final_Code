## Import all relevant libraries
import math
from datetime import datetime
from tqdm import tqdm
import bpy
import sys

bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.compute_device = 'CUDA_MULTI_2'
import numpy as np
import math as m
import random
import psutil

MAX_LIGHTS = 3
TOTAL_LIGHT_POWER = 30


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'


class Render:

    def __init__(self, filepath):

        # Set rendering to use GPU
        self.last_max_gamma = None
        self.last_min_gamma = None
        self.last_max_beta = None
        self.last_min_beta = None
        self.make_dim = False
        bpy.context.scene.cycles.device = 'GPU'
        self.counter = 0
        # Configure GPU settings
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences

        # Attempt to set GPU device types if available
        cprefs.compute_device_type = 'CUDA'  # or 'OPTIX' for RTX cards
        devices = cprefs.devices
        # Enable all CPU and GPU devices
        for device in cprefs.devices:
            device.use = True

        ## Scene information
        # Define the scene information
        self.Origlight_names = ['Light1', 'Light2']
        self.light_names = []
        for i in range(MAX_LIGHTS):
            # Create light data
            lightName = "light_" + str(i)
            self.light_names.append(lightName)
        self.total_light_power = TOTAL_LIGHT_POWER
        self.scene = bpy.data.scenes['OriginalScene']
        # Define the information relevant to the <bpy.data.objects>
        self.camera = bpy.data.objects['Camera']
        self.last_camera_location = (0, 0, 3)  # Initialize with default or current camera location
        self.last_camera_rotation = (0, 0, 0)  # Initialize with default or current camera rotation
        self.axis = bpy.data.objects['Main Axis']
        self.light_1 = bpy.data.objects['Light1']
        self.light_2 = bpy.data.objects['Light2']
        self.obj_names = {'BigBox': 0, 'Nozzle': 1, 'Rocket': 2, 'SmallBox': 3, 'StartZone': 4, 'RedZone': 5,
                          'BlueZone': 6, 'GreenZone': 7, 'YellowLine': 8, 'WhiteLine': 9, 'Button': 10}
        self.objects = self.create_objects()  # Create list of bpy.data.objects from bpy.data.objects[1] to
        # bpy.data.objects[N]
        self.floorandtableobjs = self.create_floor_and_table()
        self.lights = []
        self.light_types = ['AREA', 'POINT', 'SUN', 'SPOT']
        self.light_colors = [(random.random(), random.random(), random.random(), random.random()) for i in range(40)]

        # Set the origin of each object to its geometric center
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        for obj in self.objects:
            obj.select_set(True)  # Select the object
            bpy.context.view_layer.objects.active = obj  # Set the active object to the current object
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')  # Set origin to geometric center
            obj.select_set(False)  # Deselect the object

        self.max = 10

        ## Render information
        self.camera_d_limits = [0.2, 0.8]  # Define range of heights z in m that the camera is going to pan through
        self.beta_limits = [80, -80]  # Define range of beta angles that the camera is going to pan through
        self.gamma_limits = [0, 360]  # Define range of gamma angles that the camera is going to pan through

        ## Output information
        # Input your own preferred location for the images and labels
        self.overall_filepath = filepath
        self.images_filepath = f'{filepath}/images'
        self.labels_filepath = f'{filepath}/labels'
        self.set_image_background()
        self.set_hdri_background()

    def set_image_background(self):
        # Load new image
        new_image_path = "BlenderStuffs/Woodtable.jpg"  # Replace with your new image path
        new_image = bpy.data.images.load(new_image_path)

        # Select object
        obj = bpy.data.objects["Floor"]  # Replace with your object's name

        # Check if the object has material
        if obj.data.materials:
            mat = obj.data.materials[0]  # Get the first material

            # Check if material uses nodes
            if mat.use_nodes:
                nodes = mat.node_tree.nodes

                # Find Image Texture node
                for node in nodes:
                    if node.type == 'TEX_IMAGE':
                        # Replace the image in the Image Texture node
                        node.image = new_image
                        break

    def set_camera(self):
        self.axis.rotation_euler = (0, 0, 0)
        self.axis.location = (0, 0, 0)
        self.camera.location = (0, 0, 3)

    def drop_objects_onto_table(self):
        total_frames = 250  # Total number of frames in the simulation
        print("Starting simulation...")

        # Run the simulation
        for frame in tqdm(range(1, total_frames + 1)):
            bpy.context.scene.frame_set(frame)

        rot_x = math.radians(90)  # Rotate 90 degrees on X
        rot_y = math.radians(0)  # Rotate 0 degrees on Y
        rot_z = math.radians(0)  # Rotate 0 degrees on Z
        zn_coord = 0.005

        rocket = bpy.data.objects["Rocket"]
        rocket.rotation_euler = (rot_x, rot_y, rot_z)
        n = bpy.data.objects["Nozzle"]
        n.rotation_euler = (rot_x, rot_y, rot_z)
        n.location.z = zn_coord
        n1 = bpy.data.objects["Nozzle.001"]
        n1.rotation_euler = (rot_x, rot_y, rot_z)
        n1.location.z = zn_coord

        print("Simulation completed.")

    def apply_transformations(self):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in self.objects:
            if obj.name != 'table':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                obj.select_set(False)

    def adjust_total_light_power(self):
        # Randomly decide to reduce TOTAL_LIGHT_POWER
        if random.random() < 0.5:  # 50% chance to reduce total light power
            return random.uniform(5, 15)  # Lower total light power
        else:
            return self.total_light_power

    def adjust_light_intensity(self, light, min_intensity=0, max_intensity=15):
        # Randomly choose a new intensity within the given range
        new_intensity = random.uniform(min_intensity, max_intensity)
        light.data.energy = new_intensity

    def move_light(self, light, x_range=(-5, 5), y_range=(-5, 5), z_range=(1, 10)):
        # Generate new coordinates within the specified ranges
        new_x = random.uniform(*x_range)
        new_y = random.uniform(*y_range)
        new_z = random.uniform(*z_range)

        # Update light position
        light.location = (new_x, new_y, new_z)

    def random_lighting(self):
        # Create lights if they don't exist
        if not self.lights:
            for i in range(MAX_LIGHTS):
                # Create light data
                light = bpy.data.lights.new(name="light_" + str(i), type=random.choice(self.light_types))

                try:
                    # Set random properties
                    light.color = random.choice(self.light_colors)
                except Exception as e:
                    light.color = (random.random(), random.random(), random.random())
                light.energy = random.uniform(0, 15)

                # Create light object
                light_obj = bpy.data.objects.new(name="light_" + str(i), object_data=light)

                # Link object to scene
                bpy.context.collection.objects.link(light_obj)

                # Randomize light position
                self.move_light(light_obj)

                # Add to lights list
                self.lights.append(light_obj)

        # Modify existing lights
        else:
            total_power = self.total_light_power
            for light in self.lights:
                # Adjust light intensity
                self.adjust_light_intensity(light, 0, 15)

                # Move light
                self.move_light(light)

                # Randomize light rotation
                light.rotation_euler = (random.random(), random.random(), random.random())

                # Adjust light color
                try:
                    # Set random properties
                    light.color = random.choice(self.light_colors)
                except Exception as e:
                    light.color = (random.random(), random.random(), random.random())

                # Accumulate total power
                total_power += light.data.energy

            # Normalize light power
            for light in self.lights:
                if total_power > 0:
                    light.data.energy *= (TOTAL_LIGHT_POWER / total_power)
                    self.total_light_power = total_power
        if self.make_dim:
            # Set dim lighting conditions
            for light in self.lights:
                light.data.energy = random.uniform(0, 5)  # Much lower intensity
                light.color = (0.1, 0.1, 0.1, 0.1)  # Darker color

    def random_exposure(self):
        expose_range = (0.05, .6)
        expose = random.uniform(expose_range[0], expose_range[1])

        self.scene.view_settings.exposure = expose

    def create_spotlight(self, name, location, rotation, energy=1000, color=(1, 1, 1), spot_size=0.5, spot_blend=0.1):
        # Create a new lamp data-block
        light_data = bpy.data.lights.new(name=name, type='SPOT')
        light_data.energy = energy
        light_data.color = color
        light_data.spot_size = spot_size
        light_data.spot_blend = spot_blend

        # Create a new object with our light data-block
        light_object = bpy.data.objects.new(name=name, object_data=light_data)
        light_object.location = location
        light_object.rotation_euler = rotation

        # Link light object to the current collection
        bpy.context.collection.objects.link(light_object)
        return light_object

    def vary_spotlight_angle(self, spotlight, angle_range=math.radians(45)):
        # Randomly change the rotation within the given range
        delta_rot = (random.uniform(-angle_range, angle_range),
                     random.uniform(-angle_range, angle_range),
                     random.uniform(-angle_range, angle_range))

        spotlight.rotation_euler = (
            spotlight.rotation_euler[0] + delta_rot[0],
            spotlight.rotation_euler[1] + delta_rot[1],
            spotlight.rotation_euler[2] + delta_rot[2]
        )

    def adjust_spotlight_beam(spotlight, spot_size=None, spot_blend=None):
        if spot_size is not None:
            spotlight.data.spot_size = spot_size
        if spot_blend is not None:
            spotlight.data.spot_blend = spot_blend

    def is_overlapping(self, obj1, obj2):
        # Simple distance-based check
        distance = (obj1.location - obj2.location).length
        return distance < (obj1.dimensions.length / 1.5 + obj2.dimensions.length / 1.5)

    def adjust_position(self, obj, bounds, z_fixed):
        # Check if the object is outside the bounds and reposition it randomly within the bounds

        newx = random.uniform(bounds['min_x'], bounds['max_x'])
        while newx == obj.location.x:
            newx = random.uniform(bounds['min_x'], bounds['max_x'])
        obj.location.x = newx

        newy = random.uniform(bounds['min_y'], bounds['max_y'])
        while newy == obj.location.y:
            newy = random.uniform(bounds['min_y'], bounds['max_y'])
        obj.location.y = newy

        if not z_fixed:
            newz = random.uniform(bounds['min_z'], bounds['max_z'])
            while newz == obj.location.z:
                newz = random.uniform(bounds['min_z'], bounds['max_z'])
            obj.location.z = newz


    def get_bounds(self):
        # Object names
        wall_names = ['Wall0', 'Wall1', 'Wall2', 'Wall3', 'Lid']
        table_name = 'Table'
        # Initialize boundaries
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

        # Calculate boundaries
        for name in wall_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                # Calculate local boundaries
                loc = obj.location
                dim = obj.dimensions

                local_min_z, local_max_z = loc.z - dim.z / 2, loc.z + dim.z / 2

                min_z, max_z = min(min_z, local_min_z), max(max_z, local_max_z)
            else:
                print(f"Object {name} not found")

        table = bpy.data.objects.get(table_name)
        if table is not None:
            loc = table.location
            dim = table.dimensions
            min_x, max_x = min(min_x, loc.x - dim.x / 2), max(max_x, loc.x + dim.x / 2)
            min_y, max_y = min(min_y, loc.y - dim.y / 2), max(max_y, loc.y + dim.y / 2)
        else:
            print(f"Table object {table_name} not found")

        max_z = max_z - 0.075
        min_z = min_z + 0.075
        max_y = max_y - 0.075
        min_y = min_y + 0.075
        max_x = max_x - 0.075
        min_x = min_x + 0.075
        bounds = {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y, 'min_z': min_z, 'max_z': max_z}
        return bounds

    def reposition_objects_on_table(self):
        bounds = self.get_bounds()
        table = bpy.data.objects['Table']

        z_fixed_objects = ['Nozzle', 'Nozzle.001']

        fixed_objects = ['StartZone', 'RedZone', 'BlueZone', 'GreenZone', 'YellowLine', 'WhiteLine', 'StartZone.001',
                           'RedZone.001', 'BlueZone.001', 'GreenZone.001', 'YellowLine.001', 'WhiteLine.001',
                           'BlueZone.002', 'BlueZone.003', 'GreenZone.002', 'YellowLine.002', 'WhiteLine.002',
                           'RedZone.002', 'StartZone.002', 'StartZone.003']
        z_fixed = False
        for obj in self.objects:
            if obj.name != 'Table':

                if obj.name in z_fixed_objects:
                    z_fixed = True
                else:
                    z_fixed = False

                # Check for and resolve overlaps
                for other_obj in self.objects:
                    if other_obj != obj and other_obj.name != 'Table':
                        if obj.name not in fixed_objects:
                            if other_obj.name in z_fixed_objects and obj.name in z_fixed_objects:
                                while self.is_overlapping(obj, other_obj):
                                    self.adjust_position(obj, bounds, z_fixed)

                            self.adjust_position(obj, bounds, z_fixed)

    def set_hdri_background(self):
        hdri_files = [
            "BlenderStuffs/Backgrounds/aft_lounge_8k.hdr",
            "BlenderStuffs/Backgrounds/blender_institute_8k.hdr",
            "BlenderStuffs/Backgrounds/combination_room_8k.hdr",
            "BlenderStuffs/Backgrounds/dancing_hall_8k.hdr",
            "BlenderStuffs/Backgrounds/fireplace_8k.hdr",
            "BlenderStuffs/Backgrounds/kiara_interior_8k.hdr",
            "BlenderStuffs/Backgrounds/lebombo_8k.hdr",
            "BlenderStuffs/Backgrounds/lythwood_lounge_8k.hdr",
            "BlenderStuffs/Backgrounds/music_hall_02_8k.hdr",
            "BlenderStuffs/Backgrounds/photo_studio_loft_hall_8k.hdr",
            "BlenderStuffs/Backgrounds/sculpture_exhibition_8k.hdr",
            "BlenderStuffs/Backgrounds/st_fagans_interior_8k.hdr",
            "BlenderStuffs/Backgrounds/thatch_chapel_8k.hdr",
            "BlenderStuffs/Backgrounds/wooden_lounge_8k.hdr",
        ]

        # Randomly select an HDRI file
        hdri_path = random.choice(hdri_files)

        # Ensure the use of nodes in world settings
        bpy.context.scene.world.use_nodes = True

        # Get world nodes
        world = bpy.context.scene.world
        nodes = world.node_tree.nodes

        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)

        # Add Environment Texture node
        env_texture_node = nodes.new('ShaderNodeTexEnvironment')
        # Load the HDRI image
        env_texture_node.image = bpy.data.images.load(hdri_path)

        # Add Background node
        background_node = nodes.new('ShaderNodeBackground')

        # Add Output node
        output_node = nodes.new('ShaderNodeOutputWorld')

        # Link nodes
        links = world.node_tree.links
        link = links.new(env_texture_node.outputs['Color'], background_node.inputs['Color'])
        link = links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

    def main_rendering_loop(self, rot_step, iteration, calc=True, n_renders=0):
        """
        This function represent the main algorithm explained in the Tutorial, it accepts the
        rotation step as input, and outputs the images and the labels to the above specified locations.
        """

        # Calculate the number of images and labels to generate
        if calc:
            n_renders = self.calculate_n_renders(rot_step)  # Calculate number of images
        print('Number of renders to create:', n_renders)

        accept_render = 'Y'  # Ask whether to procede with the data generation

        if accept_render == 'Y':  # If the user inputs 'Y' then procede with the data generation
            object_rotation_step = m.radians(random.randint(-10, 10))  # 5 degrees per camera step, adjust as needed

            # Create .txt file that record the progress of the data generation
            report_file_path = self.labels_filepath + '/progress_report.txt'
            report = open(report_file_path, 'w')
            # Multiply the limits by 10 to adapt to the for loop
            dmin = int(self.camera_d_limits[0] * 11)
            dmax = int(self.camera_d_limits[1] * 11)
            # Define a counter to name each .png and .txt files that are outputted
            render_counter = 0
            i = 0
            # Define the step with which the pictures are going to be taken
            rotation_step = rot_step

            min_gamma = self.gamma_limits[0]
            max_gamma = self.gamma_limits[1]
            for i in tqdm(range(n_renders)):
                # Begin nested loops
                for d in range(dmin, dmax + 1, 2):  # Loop to vary the height of the camera
                    ## Update the height of the camera

                    self.camera.location = (0, 0, d / 10)

                    randomnum = random.random()
                    # Refactor the beta limits for them to be in a range from 0 to 360 to adapt the limits to the for loop

                    min_beta = (-1) * self.beta_limits[0] + 90
                    max_beta = (-1) * self.beta_limits[1] + 90

                    for beta in range(min_beta, max_beta + 1, rotation_step):  # Loop to vary the angle beta
                        beta_r = (-1) * beta + 90  # Re-factor the current beta

                        for gamma in range(min_gamma, max_gamma + 1,
                                           rotation_step):  # Loop to vary the angle gamma

                            ## Update the rotation of the axis
                            axis_rotation = (m.radians(beta_r), 0, m.radians(gamma))
                            self.axis.rotation_euler = axis_rotation  # Assign rotation to <bpy.data.objects['Empty']> object
                            # Display demo information - Location of the camera

                            # Rotate each object
                            for obj in self.objects:
                                if obj.name not in ['YellowLine', 'WhiteLine', 'YellowLine.001', 'WhiteLine.001']:
                                    if obj.name in ['Button', 'Button.001']:
                                        obj.rotation_euler.z = m.radians(gamma)
                                    else:
                                        obj.rotation_euler.z += m.radians(random.randint(-20, 20))

                            if (render_counter > 1 and random.random() > 0.4):
                                ## Configure lighting for predefined lights
                                energy1 = random.uniform(0, 30)  # Random light intensity
                                self.light_1.data.energy = energy1
                                try:
                                    self.light_1.data.color = random.choice(self.light_colors)  # Random light color
                                except Exception as e:
                                    self.light_1.data.color = (random.random(), random.random(), random.random())

                                energy2 = random.uniform(4, 20)  # Random light intensity
                                self.light_2.data.energy = energy2
                                try:
                                    self.light_2.data.color = random.choice(self.light_colors)  # Random light color
                                except Exception as e:
                                    self.light_2.data.color = (random.random(), random.random(), random.random())

                                # Then call this new method in the main_rendering_loop or random_lighting
                            TOTAL_LIGHT_POWER = self.adjust_total_light_power()

                            if randomnum < 0.5:
                                # Update light colors list
                                self.light_colors = [(random.random(), random.random(), random.random(), 1) for i in
                                                     range(20)]
                                self.random_exposure()
                                # Remove and recreate lights
                                for light in self.lights:
                                    bpy.data.objects.remove(light)
                                self.lights = []
                                self.random_lighting()  # This will recreate the lights with new properties



                            elif randomnum > 0.6:
                                # Adjust intensity and color for predefined lights
                                self.adjust_light_intensity(self.light_1, 0, 15)
                                self.adjust_light_intensity(self.light_2, 0, 15)
                                self.set_hdri_background()
                                if randomnum > 0.8:
                                    self.scene.view_settings.exposure = 0.5
                                    self.light_1.color = random.choice(self.light_colors)
                                    self.random_exposure()
                                    self.set_hdri_background()


                            else:
                                # Update light colors list and adjust exposure and lighting
                                self.light_colors = [(random.random(), random.random(), random.random(), 1) for i in
                                                     range(40)]
                                self.random_exposure()
                                self.random_lighting()

                            now = datetime.now()
                            dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")

                            with open('blenderlog.log', 'w') as f:
                                # Save the current stdout and stderr
                                original_stdout = sys.stdout
                                original_stderr = sys.stderr

                                # Redirect stdout and stderr to the file
                                sys.stdout = f
                                sys.stderr = f

                                try:

                                    ## Generate render
                                    self.render_blender(
                                        self.counter,
                                        dt_string)  # Take photo of current scene and ouput the render_counter.png file
                                    # Display demo information - Photo information
                                    pass
                                finally:
                                    # Reset stdout and stderr
                                    sys.stdout = original_stdout
                                    sys.stderr = original_stderr

                            ## Output Labels
                            text_file_name = self.labels_filepath + '/' + 'button_board_nm_' + dt_string + '.txt'  # Create label file name
                            text_file = open(text_file_name, 'w+')  # Open .txt file of the label
                            # Get formatted coordinates of the bounding boxes of all the objects in the scene
                            # Display demo information - Label construction

                            text_coordinates = self.get_all_coordinates()
                            splitted_coordinates = text_coordinates.split('\n')[:-1]  # Delete last '\n' in coordinates
                            text_file.write('\n'.join(
                                splitted_coordinates))  # Write the coordinates to the text file and output the render_counter.txt file
                            text_file.close()  # Close the .txt file corresponding to the label

                            print('Progress = ' + str(render_counter) + '/' + str(n_renders))
                            report.write(
                                'Progress: ' + str(render_counter) + ' Rotation: ' + str(
                                    axis_rotation) + ' z_d: ' + str(
                                    d / 10) + '\n')
                            render_counter += 1  # Update counter
                            self.counter += 1
                            if render_counter > n_renders:
                                print(f"Reached render limit of {n_renders}. Stopping.")
                                self.last_camera_location = self.camera.location
                                self.last_camera_rotation = self.camera.rotation_euler
                                self.last_min_beta = beta
                                self.last_max_beta = max_beta
                                self.last_min_gamma = gamma
                                self.last_max_gamma = max_gamma
                                return  # Break out of all loops and end function
            report.close()  # Close the .txt file corresponding to the report
        else:  # If the user inputs anything else, then abort the data generation
            print('Aborted rendering operation')
        pass

    def get_all_coordinates(self):
        '''
        This function takes no input and outputs the complete string with the coordinates
        of all the objects in view in the current image
        '''
        main_text_coordinates = ''  # Initialize the variable where we'll store the coordinates
        for objct in self.objects:  # Loop through all of the objects
            base_name = objct.name.split('.')[0]  # Get the base name
            class_index = self.obj_names.get(base_name, -1)  # Get the class index
            if class_index == -1:  # If the object is not recognized, skip it
                continue

            b_box = self.find_bounding_box(objct)  # Get current object's coordinates
            if b_box:  # If find_bounding_box() doesn't return None

                text_coordinates = self.format_coordinates(b_box, class_index)  # Reformat coordinates to YOLOv3 format

                main_text_coordinates += text_coordinates  # Update main_text_coordinates

        return main_text_coordinates  # Return all coordinates

    def format_coordinates(self, coordinates, classe):
        '''
        This function takes as inputs the coordinates created by the find_bounding box() function, the current class,
        the image width and the image height and outputs the coordinates of the bounding box of the current class
        '''
        # If the current class is in view of the camera
        if coordinates:
            ## Change coordinates reference frame
            x1 = (coordinates[0][0])
            x2 = (coordinates[1][0])
            y1 = (1 - coordinates[1][1])
            y2 = (1 - coordinates[0][1])

            ## Get final bounding box information
            width = abs(x2 - x1)  # Calculate the absolute width of the bounding box
            height = abs(y2 - y1)  # Calculate the absolute height of the bounding box
            # Calculate the absolute center of the bounding box
            cx = abs(x1 + (width / 2))
            cy = abs(y1 + (height / 2))

            ## Formulate line corresponding to the bounding box of one class
            txt_coordinates = str(classe) + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(width) + ' ' + str(height) + '\n'

            return txt_coordinates
        # If the current class isn't in view of the camera, then pass
        else:
            pass

    def find_bounding_box(self, obj):
        """
        Returns camera space bounding box of the mesh object.

        Gets the camera frame bounding box, which by default is returned without any transformations applied.
        Create a new mesh object based on self.carre_bleu and undo any transformations so that it is in the same space as the
        camera frame. Find the min/max vertex coordinates of the mesh visible in the frame, or None if the mesh is not in view.

        :param scene:
        :param camera_object:
        :param mesh_object:
        :return:
        """

        """ Get the inverse transformation matrix. """
        matrix = self.camera.matrix_world.normalized().inverted()
        """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
        mesh = obj.to_mesh(preserve_all_data_layers=True)
        mesh.transform(obj.matrix_world)
        mesh.transform(matrix)

        """ Get the world coordinates for the camera frame bounding box, before any transformations. """
        frame = [-v for v in self.camera.data.view_frame(scene=self.scene)[:3]]

        lx = []
        ly = []

        for v in mesh.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                """ Vertex is behind the camera; ignore it. """
                continue
            else:
                """ Perspective division """
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        """ Image is not in view if all the mesh verts were ignored """
        if not lx or not ly:
            return None

        min_x = np.clip(min(lx), 0.0, 1.0)
        min_y = np.clip(min(ly), 0.0, 1.0)
        max_x = np.clip(max(lx), 0.0, 1.0)
        max_y = np.clip(max(ly), 0.0, 1.0)

        """ Image is not in view if both bounding points exist on the same side """
        if min_x == max_x or min_y == max_y:
            return None

        """ Figure out the rendered image size """
        render = self.scene.render
        fac = render.resolution_percentage * 0.01
        dim_x = render.resolution_x * fac
        dim_y = render.resolution_y * fac

        ## Verify there's no coordinates equal to zero
        coord_list = [min_x, min_y, max_x, max_y]
        if min(coord_list) == 0.0:
            indexmin = coord_list.index(min(coord_list))
            coord_list[indexmin] = coord_list[indexmin] + 0.0000001

        return (min_x, min_y), (max_x, max_y)

    def render_blender(self, count_f_name, dt_string):
        # Define random parameters
        random.seed(random.randint(1, 1000))
        self.xpix = 640
        self.ypix = 640
        self.percentage = random.randint(90, 100)
        self.samples = random.randint(25, 50)
        # Render images
        image_name = 'button_board_nm_' + dt_string + '.png'
        self.export_render(self.xpix, self.ypix, self.percentage, self.samples, self.images_filepath, image_name)

    def export_render(self, res_x, res_y, res_per, samples, file_path, file_name):
        # Set all scene parameters
        bpy.context.scene.cycles.samples = samples
        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = res_per
        self.scene.render.filepath = file_path + '/' + file_name

        # Take picture of current visible scene
        bpy.ops.render.render(write_still=True)

    def adjust_color(self, base_color, variation=1):
        """
        Adjusts an RGB color by a random amount within the given variation range.

        :param base_color: A tuple of (R, G, B) values.
        :param variation: Maximum variation for each color channel.
        :return: Adjusted color.
        """
        return tuple(
            min(max(channel + random.uniform(-variation, variation), 0.0), 1.0)
            for channel in base_color
        )

    def calculate_n_renders(self, rotation_step):
        zmin = int(self.camera_d_limits[0] * 10)
        zmax = int(self.camera_d_limits[1] * 10)

        render_counter = 0
        rotation_step = rotation_step

        for d in range(zmin, zmax + 1, 2):
            camera_location = (0, 0, d / 10)
            min_beta = (-1) * self.beta_limits[0] + 90
            max_beta = (-1) * self.beta_limits[1] + 90

            for beta in range(min_beta, max_beta + 1, rotation_step):
                beta_r = 90 - beta

                for gamma in range(self.gamma_limits[0], self.gamma_limits[1] + 1, rotation_step):
                    render_counter += 1
                    axis_rotation = (beta_r, 0, gamma)

        return render_counter

    def create_floor_and_table(self):
        objs = []
        floorandtable = ['Floor', 'Table']
        for obj_name in floorandtable:
            for obj in bpy.data.objects:
                if obj.name.startswith(obj_name):
                    objs.append(obj)
        return objs

    def create_objects(self):
        objs = []
        for obj_name in self.obj_names:
            for obj in bpy.data.objects:
                if obj.name.startswith(obj_name):
                    objs.append(obj)
        return objs

    def create_lights(self):
        lights = []
        for light_name in self.light_names:
            for light in bpy.data.objects:
                if light.name.startswith(light_name):
                    lights.append(light)
        return lights


def load_blend_file(filepath):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.wm.open_mainfile(filepath=filepath)


def drawitup(blend_file_path, rotation_step, calculate, number_of_renders, input_dir, iterations):
    load_blend_file(blend_file_path)
    r = Render(input_dir)
    r.set_camera()
    r.reposition_objects_on_table()  # Reposition objects for next iteration
    for _ in range(iterations):
        rotation_step = random.randint(137, 316)
        r.reposition_objects_on_table()  # Reposition objects for next iteration
        print(f"{Color.RED}Starting Iteration {_}{Color.END}")
        print(f"{Color.RED}Running Simulation{Color.END}")
        # r.apply_transformations()  # Apply transformations after simulation
        r.drop_objects_onto_table()  # Run simulation
        #
        print(f"{Color.RED}Rendering. Rotation Step {rotation_step}. Creating {number_of_renders} renders.{Color.END}")
        r.main_rendering_loop(rotation_step, _, calculate, number_of_renders)  # Render images
        print(f"{Color.RED}Finished Rendering{Color.END}")
        print(f"{Color.RED}Preparing for Iteration {_ + 1}. Resetting Board.{Color.END}")
        r.reposition_objects_on_table()  # Reposition objects for next iteration
        r.set_hdri_background()


# Example usage
if __name__ == '__main__':
    blend_file_path = "MDETestw_BoardColors.blend"
    rot_step = 9
    drawitup(blend_file_path, rot_step, False, 5001, 5)  # Adding '5' for iterations
