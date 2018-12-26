# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
# import matplotlib.pyplot as plt
from shapely.geometry import Point


def get_locality(df_storm):
    """
    Returns localities defined by center and radius.

    :param df_storm:
    :return:
    """

    # df_storm = pd.DataFrame(
    #     {
    #         'Radius': [.5],
    #         'Latitude': [4.6310727],
    #         'Longitude': [-74.0686756]
    #     }
    # )

    df_storm['Coordinates'] = list(zip(df_storm['Longitude'], df_storm['Latitude']))
    df_storm['Coordinates'] = df_storm['Coordinates'].apply(Point)

    gdf_storm = gpd.GeoDataFrame(df_storm, geometry='Coordinates')
    gdf_storm['Buffer'] = gdf_storm.buffer(gdf_storm['Radius'])

    gdf_storm.set_geometry('Buffer', inplace=True)

    path_localidades = '../gis/Bog_Localidades.shp'
    gdf_localidades = gpd.read_file(path_localidades, encoding='utf-8')
    # ax = gdf_localidades.plot()
    # gdf_storm.plot(color='green', alpha=.5, ax=ax)
    # plt.show()

    gdf_loc_int = gpd.overlay(gdf_localidades, gdf_storm, how='intersection')
    # gdf_loc_int.to_clipboard()
    # gdf_loc_int.plot()
    # plt.show()

    localities = set(gdf_loc_int['LocNombre'].values)

    return localities


if __name__ == '__main__':
    df_storm = pd.DataFrame(
        {
            'Radius': [.05],
            'Latitude': [4.6310727],
            'Longitude': [-74.0686756]
        }
    )

    print(get_locality(df_storm))
    pass
