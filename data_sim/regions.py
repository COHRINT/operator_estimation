import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

#  from dynamics_models import DynamicsModelGenerator
from DynamicsProfiles import *
import sys
sys.path.append('../../db_builder/')
sys.path.append('../db_builder/')
#  from database import *

class Region(object):
    def __init__(self, name, img_path, init_coordinates):
        # Initialize
        self.name = name
        self.img_path = img_path
        self.init_coordinates = init_coordinates

        self.final_coordinates = None

        # Set flag for construction visualizations
        self.VISUALIZE_MODEL = False

        # Set the h5_group_name
        self.h5_group_name = self.name.lower().replace(" ", "_")

        # Set img dimensions
        self.img_dimensions = {"nFrames" : 100, "nRows" : 100, "nCols" : 100}
        self.img_shape = (100, 100, 100)

        # Initialize empty list of point targets: dynamics models will be initialized into this list
        self.model_types = ["constant_velocity", "constant_acceleration", "circular"]
        self.point_targets = []

        self.target_types=[]
        # Core Data Layers
        self.ground_truth_layers = []
        self.shake_base_layers = []
        self.pixel_bleed_layers = []

        # NOISE LAYERS
        self.noise_layer = None
        self.shotgun_noise_layer = None
        self.structured_noise_layer = None


    def __repr__(self):
        return "\n{}\n{}\n\tPath: {}\n\tCoordinates: {}\n\tTargets: {}\n".format(self.name, "-"*len(self.name), self.img_path, self.init_coordinates, (self.point_targets))

    def getIntensity(self, time_step):
    	return np.sqrt(abs(time_step - 50))

    def getRandomModel(self):
        #random_model_name = np.random.choice(self.model_types)
        #return DynamicsModelGenerator(random_model_name, self.img_dimensions["nFrames"])
        random_model = np.random.randint(0,5); 
        start = [np.random.randint(5,95),np.random.randint(5,95)];

        '''
        if(random_model == 0):
            return Cirrus(startPose = start);
        elif(random_model == 1):
            return Stratus(startPose = start); 
        elif(random_model == 2):
            return Cumulus(startPose = start); 
        elif(random_model == 3):
            return Nimbostratus(startPose = start); 
        elif(random_model == 4):
            return Cumulonimbus(startPose = start);
        '''
        return Cumuliform(startPose = start);  

    def initPointTargets(self):
        # Set min and maximum targets in region
        min_targets, max_targets = 1, 4

        # Randomly initialize number of targets generated in region
        #  self.num_point_targets = np.random.randint(min_targets, max_targets)
        self.num_point_targets = 2

        # Initialize the point targets
        self.point_targets = [ self.getRandomModel() for _ in range(self.num_point_targets) ]
        #self.point_targets = [Cirrus() for _ in range(self.num_point_targets)]

        # Initialize plot if necessary
        if self.VISUALIZE_MODEL:
            self.f, self.axarr = plt.subplots(self.num_point_targets, 1, sharex=False, squeeze=False)

    def generateLayers(self):
        self.generateCoreLayers()
        self.generateNoiseLayers()

    def generateCoreLayers(self):

        for model_i, point_target in enumerate(self.point_targets):
            
            #ground_truth_data = point_target.model_data
            #shake_data = point_target.shake_data
            ground_truth_data = []; 
            shake_data = []; 
            intense_data = []; 

            for i in range(0,100):
                x,ints,target_type,s = point_target.update(shake=True); 
                ground_truth_data.append(x); 
                shake_data.append(s); 
                intense_data.append(ints); 

            self.target_types.append(target_type)
            # generate GROUND TRUTH and SHAKE BASE
            self.generate_TRUTH_and_SHAKE(ground_truth_data,shake_data,intense_data)

            # Generate PIXEL BLEED
            self.generate_PIXEL_BLEED(ground_truth_data, shake_data)

            if self.VISUALIZE_MODEL:
                self.plotModel(point_target, ground_truth_data, model_i)

        if self.VISUALIZE_MODEL:
            self.axarr[-1][0].set_xlabel('Distance East (km)')
            self.axarr[0][0].set_ylabel('Distance North (km)')
            plt.show()


    def generateNoiseLayers(self):
        # Generate base Noise
    	self.generate_NOISE()

    	# Generate Shotgun noise
        self.generate_SHOTGUN_NOISE()

    	# Generate Structured Noise
        self.generate_STRUCTURED_NOISE()


    def generate_TRUTH_and_SHAKE(self, ground_truth_data, shake_data,intense_data):
        allIntensity = []

        nRows = self.img_dimensions["nRows"]
        nCols = self.img_dimensions["nCols"]
        nFrames = self.img_dimensions["nFrames"]

        # Initialize ground truth and shake base containers
        self.ground_truth = np.zeros(shape=(self.img_shape))
        self.shake_base = np.zeros(shape=(self.img_shape))


        for i in range(nFrames):
            X_current = ground_truth_data[i]
            shake_current = shake_data[i]

            #allIntensity.append(self.getIntensity(i))
            allIntensity.append(intense_data[i]); 

            # Update Ground Truth
            # if (abs(X_current[0]) < nRows and abs(X_current[1]) < nCols):
            #     self.ground_truth[i][int(X_current[0])][int(X_current[1])] = allIntensity[i]
            if (X_current[0] < nRows and X_current[1] < nCols and X_current[0] > 0 and X_current[1] > 0):
                self.ground_truth[i][int(X_current[0])][int(X_current[1])] = allIntensity[i]

            # Update Shake Base
            if (abs(X_current[0]+shake_current[0]) < nRows and
                abs(X_current[1]+shake_current[1]) < nCols and
                abs(X_current[0]+shake_current[0]) >=0 and
                abs(X_current[1]+shake_current[1]) >= 0):

                self.shake_base[i][int(X_current[0]+shake_current[0])][int(X_current[1]+shake_current[1])] = allIntensity[i]

        # Append to object's ground truth and shake base layers array
        self.ground_truth_layers.append(self.ground_truth.tolist())
        self.shake_base_layers.append(self.shake_base.tolist())


    def generate_PIXEL_BLEED(self, ground_truth_data, shake_data):

        nRows = self.img_dimensions["nRows"]
        nCols = self.img_dimensions["nCols"]
        nFrames = self.img_dimensions["nFrames"]

        self.pixel_bleed = np.zeros(shape=(self.img_shape))
        for i in range(nFrames):
            x_data = ground_truth_data[i][0] + shake_data[i][0]
            y_data = ground_truth_data[i][1] + shake_data[i][1]
            for x_neighbor in range(-3, 4):
                for y_neighbor in range(-3, 4):
                    if (not (x_neighbor == 0 and y_neighbor == 0) and
                        x_data+x_neighbor < nRows and
                        y_data+y_neighbor < nCols and
                        x_data+x_neighbor >= 0 and
                        y_data+y_neighbor >= 0):
                        self.pixel_bleed[i][int(x_data)+x_neighbor][int(y_data)+y_neighbor] = np.random.randint(2, 4) * (1 / (abs(x_neighbor) + abs(y_neighbor)))

        # Append to object's pixel bleed layers
        self.pixel_bleed_layers.append(self.pixel_bleed.tolist())

    def generate_NOISE(self):
        self.noise_layer = np.random.uniform(0, 0.3, size=self.img_shape)

    def generate_SHOTGUN_NOISE(self):
        nRows = self.img_dimensions["nRows"]
        nCols = self.img_dimensions["nCols"]
        nFrames = self.img_dimensions["nFrames"]

        self.shotgun_noise_layer = np.zeros(shape=self.img_shape)
        x_dim = range(0, nRows)
        y_dim = range(0, nCols)

        # Static Noise
        percent_of_frame_static = .002
        for j in range(0, int(round(nRows * nCols * percent_of_frame_static))):
            x_pixel = x_dim.index(np.random.choice(x_dim))
            y_pixel = y_dim.index(np.random.choice(y_dim))
            self.shotgun_noise_layer[:, x_pixel, y_pixel] = 5


        # Blinking Noise
        percent_of_frame_blink = .001
        for j in range(0, int(round(nRows * nCols * percent_of_frame_blink))):
            x_pixel = x_dim.index(np.random.choice(x_dim))
            y_pixel = y_dim.index(np.random.choice(y_dim))
            for i in range(0, nFrames):
                if np.random.choice([True, False]):
                    self.shotgun_noise_layer[i, x_pixel, y_pixel] = 5


    def adjustCoordinates(self, height, width, left_extent, right_extent, top_extent, bottom_extent):
        # Get lat and lon extent
        latmax = self.init_coordinates['latmax']    # upper-most coordinate
        latmin = self.init_coordinates['latmin']    # lower-most coordinate
        lonmax = self.init_coordinates['lonmax']    # right-most coordinate
        lonmin = self.init_coordinates['lonmin']    # left-most coordinate

        # Calculate lat and lon degree per pixel
        lat_deg_per_pixel = np.abs(latmax - latmin) / height
        lon_deg_per_pixel = np.abs(lonmax - lonmin) / width


        # Readjust longitude
        # ------------------------------------
        new_lonmin = lonmin + (left_extent * lon_deg_per_pixel)
        new_lonmax = lonmin + (right_extent * lon_deg_per_pixel)
        # ------------------------------------

        # Readjust latitude
        # ------------------------------------
        new_latmin = latmin + (top_extent * lat_deg_per_pixel)
        new_latmax = latmax - (top_extent * lat_deg_per_pixel)
        # ------------------------------------

        # Encode as to 6 sig figs
        new_lonmin = float('%.6f' % round(new_lonmin, 6))
        new_lonmax = float('%.6f' % round(new_lonmax, 6))
        new_latmin = float('%.6f' % round(new_latmin, 6))
        new_latmax = float('%.6f' % round(new_latmax, 6))

        # Set new final coords
        self.final_coordinates = {"latmax" : new_latmax, "latmin" : new_latmin, "lonmax" : new_lonmax, "lonmin" : new_lonmin}



    def generate_STRUCTURED_NOISE(self):

        # Declare dimensions of img
        nRows = self.img_dimensions["nRows"]
        nCols = self.img_dimensions["nCols"]
        nFrames = self.img_dimensions["nFrames"]

        # Get shake data
        #shake_data = self.point_targets[0].shake_data

        # Initialize structured noise layer
        self.structured_noise_layer = np.zeros(shape=self.img_shape)

        # Obtain and process the sat image to use for structurd noise
        img = Image.open(self.img_path)

        # Get height and width of original image (pixels)
    	width, height = img.size

        # Set the new height and width to the minimum dimension of origin image (square it)
    	new_width = min([width, height])
    	new_height = min([width, height])

        # Get the new extent in order to center new image within previous space
    	left_extent = (width - new_width) // 2
        right_extent = (width + new_width) // 2
    	top_extent = (height - new_height) // 2
    	bottom_extent = (height + new_height) // 2

        extents = [left_extent, right_extent, top_extent, bottom_extent]
        ex = ["left", "right", "top", "bottom"]
        for name, extent in zip(ex, extents):
            print("{}: {}".format(name, extent))

        # Adjust coordinates
        # --------------------------------------------------------------------------------
        self.adjustCoordinates(height, width, left_extent, right_extent, top_extent, bottom_extent)
        # --------------------------------------------------------------------------------


        # Crop the image within the extent bounds
    	img = img.crop((left_extent, top_extent, right_extent, bottom_extent))

        # Set thumbnail and additional processing
    	img.thumbnail((110, 110), Image.ANTIALIAS)
    	img = np.array(img)
    	img = img[:,:,2] / 255.0 * 12
    	img = np.resize(img, (110, 110))
    	img = np.flip(img, 0)


        # Create structured noise layer
        for i in range(nFrames):
            #self.structured_noise_layer[i] = img[5+shake_data[i][0] : 5+nRows+shake_data[i][0], 5+shake_data[i][1] : 5+nCols+shake_data[i][1]]
            shakex = np.random.randint(-0.4, 0.4+1);
            shakey = np.random.randint(-0.4, 0.4+1); 
            self.structured_noise_layer[i] = img[5+shakex : 5+nRows+shakex, 5+shakey : 5+nCols+shakey]


    def writeRegionH5(self, H5_FILE, into_db=False):
        # Create region group on H5 File
        region_group = H5_FILE.create_group(self.h5_group_name)

        # Add attributes to group
        for key, value in self.final_coordinates.items():
            region_group.attrs.create(key, value)

        # Create core layer subgroups
        ground_truth_subgroup = region_group.create_group("GroundTruth")
        shake_base_subgroup = region_group.create_group("ShakeBase")
        pixel_bleed_subgroup = region_group.create_group("PixelBleed")

        # Write Core layers
        total_targets=np.zeros(shape=self.img_shape)
        for i, (gt_layer, sb_layer, pb_layer) in enumerate(zip(self.ground_truth_layers, self.shake_base_layers, self.pixel_bleed_layers)):
            ground_truth_dataset=ground_truth_subgroup.create_dataset('GroundTruth_{}'.format(i), data=gt_layer)
            ground_truth_dataset.attrs.create('True Type','Cumuliform'+str(self.target_types[i]))
            shake_base_subgroup.create_dataset('ShakeBase_{}'.format(i), data=sb_layer)
            pixel_bleed_subgroup.create_dataset('PixelBleed_{}'.format(i), data=pb_layer)
            if into_db:
                total_targets=total_targets+gt_layer+sb_layer+pb_layer

        # Write Noise Layers
        region_group.create_dataset('Noise', data=self.noise_layer)
        region_group.create_dataset('StructuredNoise', data=self.structured_noise_layer)
        region_group.create_dataset('ShotgunNoise', data=self.shotgun_noise_layer)

        if into_db:
            # note 1: right now the profile names are not stored in self
            # note 2: intensities come from the detectoin script or the profile generator
            # in the future, events will be stored from the front end after running this script
            # note 3: 
            database_data=[self.structured_noise_layer,self.shotgun_noise_layer+self.noise_layer+total_targets]
            store_event(database_data,self.final_coordinates,"Altostratus")



    def plotModel(self, point_target, ground_truth_data, model_i):

        nFrames = self.img_dimensions["nFrames"]

        for i in range(nFrames):
            mark_i_color = '#%02x%02x%02x' % (i * 255 / nFrames, 255 - i * 255 / nFrames, 0)
            x = np.asarray(ground_truth_data[i][0]).reshape(-1)
            y = np.asarray(ground_truth_data[i][1]).reshape(-1)
            self.axarr[model_i][0].scatter(x, y, color=mark_i_color)

        self.axarr[model_i][0].set_title('Model: {}'.format(point_target.model_type), y=0.75)







def module_tester():
    """
        FOR TESTING ONLY
    """
    # Define region names
    region_names = ["Boulder", "North Korea", "Israel"]
    #  region_names = ["Boulder"]

    # Define region image paths
    region_img_paths = ['../imgs/boulder.png', '../imgs/north_korea.png', '../imgs/israel.png']

    # Define region coordinates
    region_coordinates = [{'latmin' : 39.971308, 'latmax' : 39.990298, 'lonmin' : -105.296981, 'lonmax' : -105.265918},
            			  {'latmin' : 39.489147, 'latmax' : 39.514605, 'lonmin' :  127.230443, 'lonmax' :  127.265342},
            			  {'latmin' : 32.075752, 'latmax' : 32.102536, 'lonmin' :  34.756838,  'lonmax' :  34.791335}]

    # Initialize REGIONS
    REGIONS = [Region(region_name, img_path, coordinates) for region_name, img_path, coordinates
                    in zip(region_names, region_img_paths, region_coordinates)]

    # Generate Region Data
    for region in REGIONS:
        region.initPointTargets()
        region.generateLayers()

        h5_filepath = '../{}_test.hdf5'.format(region.h5_group_name)
        print h5_filepath
        with h5py.File(h5_filepath, 'w') as outFile:
            region.writeRegionH5(outFile,into_db=False)

if __name__ == '__main__':
    module_tester()
