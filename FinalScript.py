# Import modules

import cartopy.crs as ccrs
import geopandas as gpd
from cartopy.feature import ShapelyFeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------------------------------------------------
# Define functions for the code

def generate_handles(labels, colors, edge='k', alpha=1):
    """Creates handles using Matplotlib to create a legend of the features on the map.

                Args:
                    labels: assigns the features' names to the handle.
                    colors: assigns the features' colours to the handle.
                    edge: sets the color of the handle edge.
                    alpha: sets the transparency of the handle.

                Returns:

                    handles: generates handles created of features for placement on the legend of the map.

    """
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

def leg():
    """Creates a legend for the map.
    Arg: null

    Returns:
        a legend for the map.
    """
    ax.legend(handles, labels, title='Legend', title_fontsize=15,
                    fontsize=11, loc='upper left', frameon=True, framealpha=1)
    print(leg)


def scale_bar(ax,length,location=(0.92, 0.95)):
    """
    Creates a labelled km scale bar placed on the upper right hand corner of the map.
             Args:
                ax: the axes to draw the scale bar on.
                length: the length of the scale bar in km.
                location: the centre of the scale bar in axis coordinates.

             Return:
                 A km scale bar placed in the upper right hand corner of the map.
         """
    # Get the limits of the axis in lat long.
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc horizontally centred on the middle of the map.
    # vertically at scale bar location.
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres.
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres.
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length.
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)

        length = scale_number(length)

        # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot scalebar and label.
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=3)
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')

def countyoutlines():

    """Creates a  feature of Northern Ireland county borders.
    Args: Null

    Return: A feature of Northern Ireland's county outlines.
    """
    county_outlines = ShapelyFeature(counties['geometry'], myCRS, edgecolor='g', facecolor='none')
    ax.add_feature(county_outlines)
    print(countyoutlines)

def waterfeature():
    """Creates a shapefile feature of lakes of Northern Ireland.
    Args: Null

    Return: A feature of the lakes of Northern Ireland."""
    waterr = ShapelyFeature(water['geometry'], myCRS, edgecolor='blue', facecolor='mediumblue', linewidth=0.5)
    ax.add_feature(waterr)
    print(waterfeature)

def scatterplot(dataset, column1, column2, markersize, colour, namefile):
    """Creates Scatterplot of column from the same geodataset.
    Args:
        dataset: geodataset containing the two columns of data
        column1: the independent variable
        column2: the dependent variable
        markersize: size of markers
        colour: colour of markers
        namefile: saved name of .png file of scatter plot

    Return:
         A scatterplot defined by markers.
        """
    df = dataset
    df.plot(kind='scatter', x=column1, y=column2, s=markersize, color=colour)
    plt.savefig(namefile)
    return scatterplot

def histogram(dataset, column, title ,filename):
    """Creates histogram of illustrating the distribution of a column of data.
        Args:
            dataset: geodataset containing the column
            column: column name of the above dataset
            title: title of the histogram centred above the histogram
            filename: saved name of .png file of histogram

        Return: A histogram depicting the distribution of a column of data.
            """
    df = dataset
    ax = df.hist(column=column)
    ax = ax[0]
    for x in ax:
    # set title
        x.set_title(title)
    # save histogram
        plt.savefig(filename)
    return histogram


def max_dataset(dataset, column):
    """Calculates the maximum value of a column of data.
            Args:
                dataset: geodataset containing the column of data
                column: column of data that the max is calculated from

            Return:
                the maximum value in a column of data from a selected geodataset
                """
    mx = dataset[[column]].max()
    return mx


def min_dataset(dataset, column):
    """Calculates the minimum value of a column of data.
        Args:
            dataset: geodataset containing the column of data
            column: column of data that the minimum value is calculated from

        Return:
            the minimum value in a column of data from a selected geodataset
            """
    mn = dataset[[column]].min()
    return mn


def mean_dataset(dataset, column):
    """Calculates the mean of a column of data.
        Args:
            dataset: geodataset containing the column of data
            column: variable that the mean is calculated from

        Return:
            Calculates the average value in a column of data from a selected geodataset
            """
    m = dataset[[column]].mean()
    return m

plt.ion()
# ---------------------------------------------------------------------------------------------------------------------
# Load the datasets.
health_issues = gpd.read_file('Datafiles/Health_Problems.shp')
deprivation_indicator = gpd.read_file('Datafiles/Deprivation(SA2011).shp')
water = gpd.read_file('Datafiles/Water.shp')
counties = gpd.read_file('Datafiles/Counties.shp')

# ---------------------------------------------------------------------------------------------------------------------
# print the health issues header.
health_issues.head()

# Display the column headers of the long-term health problem GeoDataset.
print(health_issues.columns.values)

 # Create new column that displays the % of the population has a long-term health issue per Small Area.
for i, row in health_issues.iterrows():# iterate over each row in the GeoDataFrame.
    health_issues["PC_LTHP"] = (health_issues["LTHP_littl"] + row["LTHP_lot"]) / row["residents"] * 100

# print the updated GeoDataFrame to view the new column.
print(health_issues.head())

# Find the mean, max and min % LTHP per Small Area using functions and print in sentence form.

max_LTHP = max_dataset(health_issues,"PC_LTHP")
min_LTHP = min_dataset(health_issues,"PC_LTHP")
mean_LTHP = mean_dataset(health_issues,"PC_LTHP")

print("the maximum percentage of long-term health problems per small area population is",max_LTHP)
print("the minimum percentage of long-term health problems per small area population is",min_LTHP)
print("the mean percentage of long-term health problems per small area population is",mean_LTHP)

# Create a scatter plot comparing Multiple Deprivation Measure and Long-term Health Problems

histogram(health_issues, "PC_LTHP", 'Percentage of Long-Term Health Problems per Small Area','LTHPhist.png')

# ---------------------------------------------------------------------------------------------------------------------
# create a map illustrating % of population with long term health problems per Small Area in Northern Ireland.

# Set both features to the same projection before adding to the map.
counties = counties.to_crs(epsg=32629)
health_issues = health_issues.to_crs(epsg=32629)

# create a Universal Transverse Mercator reference system to transform our data.
myCRS = ccrs.UTM(29)

# set the figure size and assign the projection.
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

# add a color bar.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)

# add the LTHP data to the map.
LTHP_plot = health_issues.plot(column='PC_LTHP', ax=ax, vmin=12, vmax=80, cmap='cividis',
                            legend=True, cax=cax, legend_kwds={'label': '% of Long-Term Health Problems'})

# add county outlines to the map.
countyoutlines()
county_handles = generate_handles([''], ['none'], edge='g')

# add water features to the map.
waterfeature()
water_handle = generate_handles(['Lakes'], ['mediumblue'])

# add legend.
handles = water_handle + county_handles
labels = ['Lakes','County Boundaries']
leg()

# add a scale bar.
scale_bar(ax, 20)

# save the figure.
fig.savefig("LTHPmap1.png", dpi=300, bbox_inches='tight')

# ------------------------------------------------------------------------------------------------------------
# join the deprivation and health issues GeoDataframe objects in order to assess the variables' relationship.
deprivation_indicator = deprivation_indicator.to_crs(epsg=32629)

# ensure that both GeoDataframe objects have the same CRS before joining.
print(health_issues.crs == deprivation_indicator.crs)

# now join the two objects.
join = gpd.sjoin(health_issues, deprivation_indicator, how='inner', lsuffix='left', rsuffix='right')
print(join)
print(join.columns.values)

# Now find the mean, max and min percentage of income deprived per Small Area using functions.
max_MDMIncome = max_dataset(join, "MDM_PC_inc")
min_MDMIncome = min_dataset(join, "MDM_PC_inc")
mean_MDMIncome = mean_dataset(join, "MDM_PC_inc")


# and print in sentence form.
print("the maximum percentage of the population that is income deprived per small area is",max_MDMIncome)
print("the minimum percentage of the population that is income deprived per small area is",min_MDMIncome)
print("the mean percentage of the population that is income deprived per small area is",mean_MDMIncome)


# create a histogram to illustrate the spread of income deprived data across small areas.
histogram(join, "MDM_PC_inc", 'Multiple Deprivation Measure per Small Area','MDMpchist.png')
print(join[['MDM_rank', 'PC_LTHP']])

# Find the mean, max and min MDM rank per Small Area using functions.
# note: MDM rank indicates how deprived an area is relative to another, with 1= most deprived.
max_MDM = max_dataset(deprivation_indicator,"MDM_rank")
min_MDM = min_dataset(deprivation_indicator,"MDM_rank")
mean_MDM = mean_dataset(deprivation_indicator,"MDM_rank")

# and print in sentence form.
print("the maximum Multiple Deprivation Measure ranking per Small Area is",max_MDM)
print("the minimum Multiple Deprivation Measure ranking per Small Area is",min_MDM)
print("the mean Multiple Deprivation Measure ranking per Small Area is",mean_MDM)

# clean the data using the Pandas dropna function to remove values with NaN values from the dataframe.
join.dropna()

# Create a scatter plot comparing Multiple Deprivation Measure and Long-term Health Problems data.
scatterplot(join, 'MDM_PC_inc', 'PC_LTHP', .05, 'b', 'scatter.png')
# ---------------------------------------------------------------------------------------------------------------------
# create a map of the Multiple Deprivation Measure rankings per Small Area in Northern Ireland.

# create a figure of size 10x10 (representing the page size in inches)
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

# make a colorbar that stays in line with our map
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
divider = make_axes_locatable(ax)

# plot the MDM data on the map
MDM_plot = deprivation_indicator.plot(column="MDM_rank", ax=ax, vmin=1, vmax=4540, cmap='plasma',
                      legend=True, cax=cax, legend_kwds={'label': 'Multiple Deprivation Rank'})

# add county outlines to the map.
countyoutlines()
county_handles = generate_handles([''], ['none'], edge='g')

waterfeature()
water_handle = generate_handles(['Lakes'], ['mediumblue'])

# add legend.
handles = water_handle + county_handles
labels = ['Lakes', 'County Boundaries']
leg()

# add a scale bar.
scale_bar(ax, 40)

# save the figure.
fig.savefig("MDR_map.png", dpi=300, bbox_inches='tight')

# ------------------------------------------------------------------------------------------------------------
