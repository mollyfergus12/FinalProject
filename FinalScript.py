# Import modules
from typing import Union
import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
from cartopy.feature import ShapelyFeature
from shapely.geometry import Point
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
import matplotlib.lines as mlines


# ---------------------------------------------------------------------------------------------------------------------
# Define functions for the code

def generate_handles(labels, colors, edge='k', alpha=1):
    """Creates handles using Matplotlib to create a legend of the features on the map.

                Args:
                    labels: assigns the features' names to the headle.
                    colors: assigns the features' colours to the handle.
                    edge: sets the color of the handle edge.
                    alpha: sets the transparency of the hande.

                Returns:
                    handles: generates handles created of features for placement on the legend of the map.

    """
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

def leg():
    """Creates a legend for the map"""
    ax.legend(handles, labels, title='Legend', title_fontsize=15,
                    fontsize=11, loc='upper left', frameon=True, framealpha=1)
    print(leg)


def gridlines():
    """Creates gridlines for the map"""
    g = ax.gridlines(draw_labels=True,
                     xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                     ylocs=[54, 54.5, 55, 55.5])
    gridlines.right_labels = True
    gridlines.bottom_labels = True
    print(g)


def colourbar():
    """Creates a colour bar that stays in line with the map"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    print(cax)

def scale_bar(ax,length,location=(0.92, 0.95)):
    """
    Creates a labelled km scale bar placed on the upper right hand corner of the map.
             Args:
                ax: the axes to draw the scale bar on.
                length: the length of the scale bar in km.
                location: the centre of the scale bar in axis coordinates.

             Return:
                 A km scale bar placed in the upper right hand corner of the map
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

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
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
    # Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=3)
    # Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')

def countyoutlines():
    """Creates a  feature of Northern Ireland county borders."""
    county_outlines = ShapelyFeature(counties['geometry'], myCRS, edgecolor='g', facecolor='none')
    ax.add_feature(county_outlines)
    print(countyoutlines)

def waterfeature():
    """Creates a shapefile feature of lakes of Northern Ireland."""
    waterr = ShapelyFeature(water['geometry'], myCRS, edgecolor='blue', facecolor='mediumblue', linewidth=0.5)
    ax.add_feature(waterr)
    print(waterfeature)

def scatterplot(dataset, column1, column2):
    """Creates Scatterplot of two variables from the same geodataset.
    Args:
        dataset: geodataset containing the two variables
        column1: the independent variable
        column2: the dependent variable

    Return: A scatterplot defined by red markers.
        """
    df = dataset
    df.plot(kind='scatter', x=column1, y=column2, color='r')
    return scatterplot


"""Plots the data for a choropleth map"""
#def main_data(dataset, columnname, mmin, mmax, color, labelname):
    #md = dataset.plot(column=columnname, ax='ax', vmin=mmin, vmax=mmax, cmap=color, legend=True, cax='cax', legend_kwds={'label': labelname})
    #print(md)

#health_issues = gpd.read_file('Datafiles/Health_Problems.shp')

# plot the LTHP data on the map
#main_data(health_issues, "PC_LTHP", 12, 80, 'viridis', 'SA Percentage with Long-term Health Issues')


def max_dataset(dataset, column):
    """Calculates the maximum value of a variable.
            Args:
                dataset: geodataset containing the variable
                column: variable that the mean is calculated from
            Return:
                Calculates the maximum value in a column of data from a selected geodataset
                """
    mx = dataset[[column]].max()
    return mx


def min_dataset(dataset, column):
    """Calculates the minimum value of a variable.
        Args:
            dataset: geodataset containing the variable
            column: variable that the minimum value is calculated from
        Return:
            Calculates the minimum value in a column of data from a selected geodataset
            """
    mn = dataset[[column]].min()
    return mn


def mean_dataset(dataset, column):
    """Calculates the mean of a variable.
        Args:
            dataset: geodataset containing the variable
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
towns = gpd.read_file('Datafiles/Towns.shp')
water = gpd.read_file('Datafiles/Water.shp')
rivers = gpd.read_file('Datafiles/Rivers.shp')
counties = gpd.read_file('Datafiles/Counties.shp')
outline = gpd.read_file('Datafiles/NI_outline.shp')

# ---------------------------------------------------------------------------------------------------------------------
health_issues.head()

# Display the column headers of the long-term health problem GeoDataset.
print(health_issues.columns.values)

 # Create new column that displays the % of the population has a long-term health issue per Small Area.
for i, row in health_issues.iterrows():# iterate over each row in the GeoDataFrame.
    health_issues["PC_LTHP"] = (health_issues["LTHP_littl"] + row["LTHP_lot"]) / row["residents"] * 100

print(health_issues.head()) # print the updated GeoDataFrame to view the new column.

# Find the mean, max and min % LTHP per Small Area and print using the format string method.

max_LTHP = max_dataset(health_issues,"PC_LTHP")
min_LTHP = min_dataset(health_issues,"PC_LTHP")
mean_LTHP = mean_dataset(health_issues,"PC_LTHP")

print("the maximum percentage of long-term health problems per small area population is",max_LTHP)
print("the minimum percentage of long-term health problems per small area population is",min_LTHP)
print("the mean percentage of long-term health problems per small area population is",mean_LTHP)

# ---------------------------------------------------------------------------------------------------------------------
# Set both features to the same projection before adding to the map.
counties = counties.to_crs(epsg=32629)
health_issues = health_issues.to_crs(epsg=32629)

myCRS = ccrs.UTM(29)  # create a Universal Transverse Mercator reference system to transform our data.

fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

# add a color bar that aligns with the map boundaries.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)

# add the LTHP data to the map.
MDM_plot = health_issues.plot(column='PC_LTHP', ax=ax, vmin=12, vmax=80, cmap='cividis',
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
fig.savefig("LTHPmap.png", dpi=300, bbox_inches='tight')

# ------------------------------------------------------------------------------------------------------------
# join the deprivation and health issues GeoDataframe objects in order to assess the variables' relationship
deprivation_indicator = deprivation_indicator.to_crs(epsg=32629)
# Ensure that both GeoDataframe objects have the same CRS before joining
print(health_issues.crs == deprivation_indicator.crs)

join = gpd.sjoin(health_issues, deprivation_indicator, how='inner', lsuffix='left', rsuffix='right')
print(join)
print(join.columns.values)

# Find the mean, max and min MDM rank per Small Area and print using the format string method.
max_MDM = max_dataset(deprivation_indicator,"MDM_rank")
min_MDM = min_dataset(deprivation_indicator,"MDM_rank")
mean_MDM = mean_dataset(deprivation_indicator,"MDM_rank")

max_MDM = join["MDM_rank"].max()
min_MDM = join["MDM_rank"].min()
mean_MDM = join["MDM_rank"].mean()

print("{:.2f} is the maximum multiple deprivation rank per small area population.".format(max_MDM))
print("{:.2f} is the minimum multiple deprivation rank per small area population.".format(min_MDM))
print("{:.2f} is the mean multiple deprivation rank per small area population.".format(mean_MDM))

# A scatter plot comparing MDM and Long-term Health Problems
df = join
df.plot(kind='scatter',x='MDM_rank',y='PC_LTHP',color='r')
plt.savefig('scatter.png')

# ---------------------------------------------------------------------------------------------------------------------
# create a figure of size 10x10 (representing the page size in inches)
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

# make a colorbar that stays in line with our map
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)

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

# save the figure
fig.savefig("MDR_map.png", dpi=300, bbox_inches='tight')

# ------------------------------------------------------------------------------------------------------------